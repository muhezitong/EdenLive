# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-generation loop and session shared by camera-to-video apps."""

from __future__ import annotations

import inspect
import time
import os
import threading
from collections import deque
from contextlib import nullcontext
from concurrent.futures import Future
import numpy as np
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.nn import functional as F

from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.core.distributed import is_distributed_initialized
from flashdreams.runtime_v2.distributed_sync import broadcast_step_gate
from flashdreams.runtime_v2.event_buffer import EventBuffer
from flashdreams.infra.postprocess import VideoPostprocessStream
from flashdreams.infra.postprocess.base import from_bvtchw, to_bvtchw
from flashdreams.infra.runner_io import ResizeInterpolation
from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
from flashdreams.runtime_v2.keyboard_input import KeyboardStateTrack
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.recent_frame_rate import RecentFrameRateTracker
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.user_input_event import (
    ClientActionUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .controls import CameraPoseIntegrator, KeyboardResampler
from .game_hud import GAME_UI_ACTION_CHANNEL, envelope, is_dialog_open
from .game_timeline import GameChunkPacket, GameTimeline
from .game_renderer import game_static_prompt
from .defaults import Cam2VConditioning, Cam2VGenerateStep, generate_camera_step
from .postprocess_comparison import compose_postprocess_comparison
from .ui import (
    RECENT_MODEL_FPS_WINDOW_SECONDS,
    Cam2VPostprocessComparisonSlangPyUILoop,
    Cam2VSlangPyUILoop,
    Cam2VUIState,
    Cam2VUIStatus,
)

GAME_EVENT_READER_ID = 2
"""EventBuffer reader used by the capture worker while a chunk is in flight.

session_runner registers UI=0 and MODEL=1. This extra reader lets the
Playwright thread drain keys during generate without stealing the model
loop's step-boundary batch (reset/close still go through MODEL).
"""

_DEFAULT_RENDER_PACE_S = 0.4
"""Wall-clock stretch for the first successor, before a generate has been timed."""

_DIALOG_POLL_SECONDS = 0.05
"""Gap between page polls while a blocking client card holds the rollout."""

PROMPT_MESSAGE_TYPE = "prompt"
"""Control-channel message carrying the text the chunk was conditioned with."""

_DEFAULT_PROMPT_EVENT_CHUNKS = 1
"""Chunks an event line stays in the prompt after the chunk it happened in."""


@dataclass(kw_only=True, slots=True)
class CameraControlInput:
    """Model-neutral per-step camera payload."""

    intrinsics: torch.Tensor
    """Per-frame intrinsics shaped ``[T, 4]``."""

    poses: torch.Tensor
    """Per-frame camera-to-world matrices shaped ``[T, 4, 4]``."""

    world_scale: float
    """Scale applied to camera translations by the model camera encoder."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Game2VSessionConfig:
    """Resolved immutable settings for one camera-to-video rollout."""

    conditioning: Cam2VConditioning
    """Prompt, first frame, and calibrated camera values."""

    total_blocks: int
    """Number of model steps generated before the rollout completes."""

    device: torch.device
    """Device holding model inputs and cache state."""

    model_video_width: int | None = None
    """Raw model width when presentation is postprocessed to another size."""

    model_video_height: int | None = None
    """Raw model height when presentation is postprocessed to another size."""

    first_frame_dtype: torch.dtype
    """Tensor dtype required by the model's first-frame input."""

    first_frame_interpolation: ResizeInterpolation
    """Resize interpolation required by the model's image preprocessor."""

    warmup_blocks: int
    """Leading blocks excluded from steady-state FPS."""

    generate_step: Cam2VGenerateStep = generate_camera_step
    """Integration hook that adapts camera payloads and generates one step."""

    pose_integrator_factory: Callable[[], CameraPoseIntegrator] = CameraPoseIntegrator
    """Construct the integration-calibrated camera pose integrator."""

    log_model_timing: bool = False
    """Write one synchronized wall-time record for each AR model step."""

    install_hint: str = ""
    """Optional first-frame loader hint for missing integration dependencies."""

    postprocess_enabled: bool = False
    """Whether post-processed frames are selected at session start."""

    postprocess_comparison: bool = False
    """Whether output shows original and post-processed frames side by side."""

    fixed_camera: bool = True
    """Ignore browser camera controls for this game-to-video shell."""

    game_slug: str = "freeze"
    """Local WebGameVault game used for the conditioning timeline."""

    control_stream_factory: (
        Callable[["Game2VSessionConfig", SessionDesc], Any] | None
    ) = None
    """Build the overlapped condition encoder for frame-conditioned models.

    ``None`` (the camera-control default) leaves
    :attr:`Game2VModelState.control_stream` unset, so steps pass a camera
    payload. When set, the returned stream encodes each chunk's rendered
    frames a step ahead of their use.
    """

    compile_control_encoder: bool = True
    """``torch.compile`` the condition encoder.

    ``--no-compile`` turns this off together with the pipeline's own compile
    and CUDA-graph switches. The encoder's trace runs on the first chunk, so
    leaving it on makes that chunk wait on the compiler even when everything
    else was asked to run eager.
    """

    prompt_event_chunks: int = _DEFAULT_PROMPT_EVENT_CHUNKS
    """How many generated chunks an event line survives in the prompt.

    The chunk a line is first folded into counts as one, so the default of one
    shows an event only on the chunk generated right after the one that
    simulated it. The same count governs the text sent to the client, so the
    panel and the conditioning stay in step.
    """

    dynamic_prompt: bool = False
    """Re-encode the prompt for every chunk instead of once per rollout.

    The prompt text is composed either way; this decides whether the text
    encoder sees it. Only a pipeline that exposes ``encode_text`` and
    ``replace_text_embeddings`` can do it, and the encoder stays resident on
    rank zero for the whole rollout.
    """

    wait_for_start: bool = False
    """Hold the first chunk until the web panel picks a game.

    The workbench page uses this so nothing renders before the visitor has
    built a game (or chosen an existing one) and pressed start.
    """

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("Game2VSessionConfig.total_blocks must be > 0.")
        if self.warmup_blocks < 0:
            raise ValueError("Game2VSessionConfig.warmup_blocks must be >= 0.")
        if not isinstance(self.log_model_timing, bool):
            raise TypeError("Game2VSessionConfig.log_model_timing must be bool.")
        if not isinstance(self.postprocess_enabled, bool):
            raise TypeError("Game2VSessionConfig.postprocess_enabled must be bool.")
        if not isinstance(self.postprocess_comparison, bool):
            raise TypeError("Game2VSessionConfig.postprocess_comparison must be bool.")
        if not isinstance(self.fixed_camera, bool):
            raise TypeError("Game2VSessionConfig.fixed_camera must be bool.")
        if (
            isinstance(self.prompt_event_chunks, bool)
            or not isinstance(self.prompt_event_chunks, int)
            or self.prompt_event_chunks <= 0
        ):
            raise ValueError(
                "Game2VSessionConfig.prompt_event_chunks must be a positive integer."
            )
        if not isinstance(self.dynamic_prompt, bool):
            raise TypeError("Game2VSessionConfig.dynamic_prompt must be bool.")
        if (self.model_video_width is None) != (self.model_video_height is None):
            raise ValueError("Cam2V model dimensions must be set together.")
        if self.model_video_width is not None:
            assert self.model_video_height is not None
            if self.model_video_width <= 0 or self.model_video_height <= 0:
                raise ValueError("Cam2V model dimensions must be positive.")


@dataclass(slots=True)
class Game2VModelState:
    """Mutable rollout state owned exclusively by the model-generation-thread."""

    pipeline: Any
    """Application-owned, loaded model pipeline."""

    session_desc: SessionDesc
    """Output shape, layout, and rates accepted for this session."""

    config: Game2VSessionConfig
    """Resolved inputs and rollout controls."""

    postprocess_stream: VideoPostprocessStream | None = None
    """Session-owned postprocess stream retained while the rollout runs."""

    postprocess_enabled: bool = False
    """Whether post-processed chunks are selected for presentation."""

    comparison_pending_generated_frames: torch.Tensor | None = None
    """Original tchw frames awaiting their post-processed counterparts."""

    keyboard_resampler: KeyboardResampler | None = None
    """Legacy combined input view retained for construction compatibility."""

    input_timeline: RealtimeInputTimeline = field(init=False)
    """Session-relative sampling windows owned by the model thread."""

    keyboard_track: KeyboardStateTrack = field(init=False)
    """Timestamped held-key state projected into camera-control segments."""

    game_timeline: GameTimeline = field(init=False)
    last_game_chunk: GameChunkPacket | None = None
    game_capture: Any = None
    """Rank-zero worker that renders every frame of one chunk."""

    pending_game_chunk: Any = None
    """In-flight render of the chunk this step will generate from."""

    control_stream: Any = None
    """Overlapped condition encoder for models conditioned on rendered frames.

    ``None`` for camera-control models, which take poses rather than frames.
    When set, each chunk's encode is chained onto its render future so the
    cost lands inside the previous chunk's denoising.
    """

    dialog_held: bool = False
    """Rank-zero flag that a client card is open; kept so a re-entered pause is not re-logged."""

    pending_game_frames: Any = field(default_factory=deque)

    event_buffer: EventBuffer | None = None
    """Session EventBuffer; rank zero's capture worker reads it mid-generate."""

    last_generate_wall_s: float | None = None
    """Wall time of the previous ``generate_step``, used to pace the next render."""

    static_prompt: str = ""
    """Scene description from the game's registry entry.

    Resolved once, on rank zero, the first time a chunk's prompt is composed.
    The text encoder is not fed yet; the composed prompt is recorded on
    :attr:`active_prompt` for that hand-off.
    """

    active_prompt: str = ""
    """Prompt composed for the chunk being generated.

    The static scene description plus the event lines still inside their
    configured lifetime. With ``dynamic_prompt`` this is what the text encoder
    conditioned the chunk on; otherwise the rollout keeps the prompt its cache
    was built from and only the client reads this.
    """

    active_event_lines: tuple[str, ...] = ()
    """Event lines inside :attr:`active_prompt`, in the order they were folded in.

    Kept apart from the prompt text so the client can show them on their own and
    drop them when the chunk's frames run out, rather than as a growing history.
    """

    encoded_prompt: str = ""
    """The text currently in the cross-attention cache.

    Empty until the first chunk, which records the static prompt the cache was
    built from. A chunk whose composed text differs is re-encoded; one whose
    text matches is not.
    """

    prompt_encode_s: float = 0.0
    """Wall time spent re-encoding the prompt for the chunk about to be generated.

    Zero when the composed text matches what is already in the cache. Counted
    into the chunk's generate time, since the chunk cannot start denoising until
    the swap has landed.
    """

    prompt_events: list = field(default_factory=list)
    """Event lines in the prompt, oldest chunk first.

    Each entry is ``(chunks_remaining, event_type, text)``. A line is dropped
    once its chunk count reaches zero, so the count is the line's lifetime on
    both the conditioning side and the client panel.
    """

    cache: Any | None = None
    """Session-local autoregressive model cache."""

    ar_index: int = 0
    """Autoregressive index the pipeline is addressed with.

    Distinct from the loop's ``step_index``, which counts steps for the whole
    session and never goes back. A game restart begins a new rollout, so this
    returns to zero while ``step_index`` keeps climbing. Index zero is not
    interchangeable with the others: the decoder applies causal padding there and
    produces a different frame count.
    """

    game_scene: str | None = None
    """``scene`` the rollout is currently anchored on, or ``None`` for the cover.

    A game that publishes ``scene`` in its state (a key-shot id such as
    ``overhead``) and ships ``<slug>/shots/<scene>/cover.png`` gets its rollout
    re-anchored on that frame when the scene changes; see :func:`_reanchor_rollout`.
    """

    game_round_id: int | None = None
    """``roundId`` of the game round the rollout is currently generating.

    Rank zero compares this against each rendered chunk to notice a restart. The
    games bump ``roundId`` in their own ``reset()``, so it changes for an R-key
    restart, a result-card button, and a self-restarting round alike.
    """

    start_requested: bool = False
    """Rank-zero flag set by the first panel switch; releases ``wait_for_start``."""

    started: bool = False
    """Whether the ``wait_for_start`` gate has been passed on this rank."""

    pending_home: bool = False
    """Rank-zero request to return to the home screen and re-arm the gate.

    Written by the HTTP extension's handler thread; consumed at the top of
    :meth:`Game2VModelLoop.step`, where every rank hears the same decision.
    """

    pending_game_restart: bool = False
    """Ask for a full rewind even when the requested slug is the current one.

    Set by "play again" so the same game restarts from its first frame instead
    of the switch being treated as a no-op.
    """

    pending_game_slug: str | None = None
    """Game the rank-zero web panel asked the next step to switch to.

    Written by the HTTP extension's handler thread; consumed and cleared at the
    top of :meth:`Game2VModelLoop.step`, where every rank hears the same slug.
    """

    blocks_generated: int = 0
    """Number of completed autoregressive model steps."""

    frames_generated: int = 0
    """Number of generated video frames on the virtual camera clock."""

    anchor_frame: torch.Tensor | None = None
    """The i2v anchor frame, loaded once and reused by every restarted rollout.

    The proxy-DINO pipeline reuses its one-shot encoder outputs only when a
    later ``initialize_cache`` sees this same tensor, so reloading the image
    on each restart would both repeat the decode and defeat that reuse.
    """

    pose_integrator: CameraPoseIntegrator = field(default_factory=CameraPoseIntegrator)
    """Session-local continuous camera state."""

    steady_started_at: float | None = None
    """Wall-clock origin immediately after excluded warmup blocks."""

    steady_frames_generated: int = 0
    """Frames generated since :attr:`steady_started_at`."""

    steady_held_s: float = 0.0
    """Seconds spent paused on a client card since :attr:`steady_started_at`.

    Subtracted from the steady-state window so a pause does not read as the
    model having slowed down.
    """

    _recent_model_frame_rate_tracker: RecentFrameRateTracker = field(
        default_factory=lambda: RecentFrameRateTracker(
            window_seconds=RECENT_MODEL_FPS_WINDOW_SECONDS
        )
    )
    """Trailing post-warmup AR-step throughput shown by the UI."""

    ui_loop: Cam2VSlangPyUILoop | None = None
    """Registered UI-loop handle used only through ``invoke_async``."""

    browser_metrics_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )
    browser_metrics: dict[str, float | int | str] | None = None
    browser_metrics_epoch: int = 0
    """Small, locked snapshot for the HTTP dashboard; never expose model state."""

    def __post_init__(self) -> None:
        """Expose decomposed input state behind the legacy constructor field."""
        keyboard_resampler = self.keyboard_resampler
        if keyboard_resampler is None:
            keyboard_resampler = KeyboardResampler(
                fps=self.session_desc.frames_per_second_for_step,
            )
            self.keyboard_resampler = keyboard_resampler
        self.input_timeline = keyboard_resampler.input_timeline
        self.keyboard_track = keyboard_resampler.keyboard_track
        self.game_timeline = GameTimeline(self.config.game_slug)

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Select generated or post-processed frames without resetting the stream."""
        enabled = bool(enabled)
        if enabled and self.postprocess_stream is None:
            raise RuntimeError(
                "Cannot enable post-processing without --postprocess-preset."
            )
        if enabled == self.postprocess_enabled:
            return
        self.postprocess_enabled = enabled


class Game2VModelLoop(IModelLoop[Game2VModelState]):
    """Generate one camera-controlled video chunk per model-loop iteration."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Apply new keyboard edges and generate one autoregressive block."""
        state = self.state
        if _enter_start_gate(state):
            self.exclude_from_step_time(_await_start(state, self._shutdown_event))
            state.started = True
        _apply_pending_game_switch(state)
        step_started_at = time.perf_counter()
        if state.blocks_generated == state.config.warmup_blocks:
            state.steady_started_at = step_started_at

        conditioning = state.config.conditioning
        ar_index = state.ar_index
        # The anchor frame is only needed when the cache is (re)built: the first
        # step, and any step a game restart rewound. A restart regenerates from it
        # exactly as the session started, since the rendered game frames condition
        # nothing yet and the fresh rollout is plain i2v.
        if state.cache is None:
            state.cache = state.pipeline.initialize_cache(
                # No chunk has been simulated yet, so the rollout opens on the
                # registry's scene description alone.
                text=[_opening_prompt(state, conditioning.prompt)],
                image=_anchor_frame(state),
                **_rollout_cache_kwargs(state),
                **_text_encoder_kwargs(state),
            )
        assert state.cache is not None

        frame_count = int(state.pipeline.get_num_output_frames(ar_index))
        if frame_count <= 0:
            raise ValueError(
                "Cam2V pipelines must generate at least one frame per step."
            )
        final_chunk = state.blocks_generated + 1 == state.config.total_blocks
        # The chunk generated below was begun during the previous step and has
        # been rasterizing — and eating live keys — while that step ran. The
        # first step has no predecessor, so it renders with no pacing wait.
        if state.pending_game_chunk is None:
            # Either the first step, or the step after a card paused the
            # pipeline. Both start their render here; the pause is released
            # first so no simulation runs while the card is still up.
            held_s = _await_dialog_dismissed(state, self._shutdown_event)
            # Waiting on the client is not generation cost. Left in the step
            # time it would be read as this chunk taking seconds to produce,
            # and presentation would crawl for seconds after the card closed.
            self.exclude_from_step_time(held_s)
            state.steady_held_s += held_s
            if state.blocks_generated == 0:
                # The browser launches lazily inside this first capture, and the
                # first denoising step traces the compiled DiT, so both are far
                # slower than anything that follows.
                logger.info(
                    "Game2V: rendering the opening chunk ({} frames); the "
                    "browser starts here",
                    frame_count,
                )
            state.pending_game_chunk = _begin_game_chunk(
                state, ar_index, duration_s=0.0
            )
        capture_started_at = time.perf_counter()
        state.last_game_chunk = _collect_game_chunk(state)
        game_capture_wall_s = time.perf_counter() - capture_started_at
        # A restart resets the game in place, so this chunk ends on a new round
        # and its renders belong to that round. They cannot condition the rollout
        # that has been generating the previous one, so the rollout starts over:
        # the cache is rebuilt from the anchor frame and generation resumes at
        # autoregressive index zero, conditioned on a fresh render of the reset
        # round. The chunk just collected is dropped; its frames straddle the
        # reset, so none of it is one round's opening.
        if _restart_rollout(state) or _reanchor_rollout(state):
            ar_index = state.ar_index
            frame_count = int(state.pipeline.get_num_output_frames(ar_index))
            state.cache = state.pipeline.initialize_cache(
                text=[_opening_prompt(state, conditioning.prompt)],
                image=_anchor_frame(state),
                **_rollout_cache_kwargs(state),
                **_text_encoder_kwargs(state),
            )
            state.pending_game_chunk = _begin_game_chunk(
                state, ar_index, duration_s=0.0
            )
            capture_started_at = time.perf_counter()
            state.last_game_chunk = _collect_game_chunk(state)
            game_capture_wall_s += time.perf_counter() - capture_started_at
        if len(state.last_game_chunk.frames) != frame_count:
            raise RuntimeError(
                f"Game2V rendered {len(state.last_game_chunk.frames)} frames for a "
                f"{frame_count}-frame model step; the pipelined chunk and the "
                "model cadence disagree."
            )
        # Submitted before generating so the browser renders the next chunk while
        # the GPU works on this one, stretched to the last generate so keys that
        # arrive during this generate land in the in-flight sim. The final step
        # has no successor to render.
        # A blocking card in this chunk (win, loss, or an opened help overlay)
        # holds the successor: nothing is submitted, so no simulation runs ahead,
        # and the next step waits before starting one. This chunk is still
        # generated and published below, which is how the card reaches the
        # client that has to dismiss it.
        final_chunk = state.blocks_generated + 1 == state.config.total_blocks
        if not final_chunk and not _hold_for_dialog(state):
            state.pending_game_chunk = _begin_game_chunk(
                state,
                ar_index + 1,
                duration_s=_pace_duration_s(state, ar_index + 1),
            )
        keyboard_events = (
            [] if state.config.fixed_camera else state.keyboard_track.ingest(events)
        )
        input_window = state.input_timeline.next_window(
            frame_count,
            input_times_s=(
                result.timestamp_s for result in keyboard_events if result.tracked
            ),
        )
        # The pose integrator still needs a time interval in fixed-camera mode;
        # an empty segment list is invalid even though the held-key state is empty.
        segments = (
            [
                (
                    input_window.start_s,
                    input_window.end_s,
                    frozenset(),
                )
            ]
            if state.config.fixed_camera
            else state.keyboard_track.segments(input_window)
        )
        frame_times = list(input_window.sample_times_s)
        poses = state.pose_integrator.integrate_chunk(
            segments=segments,
            frame_times=frame_times,
        )
        camera_input = CameraControlInput(
            intrinsics=conditioning.base_intrinsics.repeat(frame_count, 1).to(
                device=state.config.device,
                dtype=torch.float32,
            ),
            poses=torch.from_numpy(poses).to(
                device=state.config.device,
                dtype=torch.float32,
            ),
            world_scale=conditioning.world_scale,
        )
        # Frame-conditioned models take the encoded render instead of a camera
        # payload. The encode was started when this chunk's render was
        # submitted, one step ago, so in steady state this is a handoff rather
        # than a wait.
        step_input = camera_input
        if state.control_stream is not None:
            step_input = state.control_stream.take(ar_index)
        # The chunk just collected is the previous chunk's simulation, so its
        # events belong to the chunk generated below. Composed before the
        # generate rather than after it, because a per-chunk prompt has to be
        # in the cross-attention cache while the chunk denoises, not after.
        _compose_chunk_prompt(state)
        if state.blocks_generated == 0:
            logger.info(
                "Game2V: generating the first chunk; the compiled DiT traces "
                "its input shape here, so this step is much slower than later ones"
            )
        browser_metrics_lock = getattr(state, "browser_metrics_lock", None)
        with (
            browser_metrics_lock if browser_metrics_lock is not None else nullcontext()
        ):
            browser_metrics_epoch = getattr(state, "browser_metrics_epoch", 0)
        generate_started_at = time.perf_counter()
        frames = state.config.generate_step(
            state.pipeline,
            ar_index,
            state.cache,
            step_input,
        )
        finalize_metrics = state.pipeline.finalize(
            autoregressive_index=ar_index,
            cache=state.cache,
        )
        metrics = _numeric_metrics(finalize_metrics)
        _synchronize_output(frames)
        model_completed_at = time.perf_counter()
        generate_wall_s = model_completed_at - generate_started_at
        # The prompt swap lands before denoising starts, so its cost belongs to
        # this chunk's generate time rather than to the gap between log lines.
        generate_wall_s += state.prompt_encode_s
        state.prompt_encode_s = 0.0
        model_step_wall_s = generate_wall_s
        state.last_generate_wall_s = generate_wall_s
        # The same renders that conditioned this step are what the client sees.
        # Only rank zero renders, so only rank zero has frames to pair and to
        # drain below; elsewhere the queue must stay empty rather than grow.
        if state.game_capture is not None:
            state.pending_game_frames.extend(state.last_game_chunk.frames)
        postprocess_started_at = time.perf_counter()

        generated_frames = frames.detach()
        output_frames = generated_frames
        postprocess_stream = state.postprocess_stream
        postprocess_output_frames = generated_frames
        if state.config.postprocess_comparison and postprocess_stream is None:
            raise RuntimeError(
                "Post-processing comparison requires a configured postprocessor."
            )
        if postprocess_stream is not None:
            postprocess_output_frames = postprocess_stream.process(
                generated_frames,
                autoregressive_index=ar_index,
            )
            if final_chunk:
                tail = postprocess_stream.finish()
                if tail is not None:
                    postprocess_output_frames = _concatenate_video(
                        postprocess_output_frames,
                        tail,
                        layout=state.session_desc.output_layout,
                    )
            # Presentation pacing observes the complete model-loop wall time.
            # Do not let asynchronous post-processing spill into the next step.
            _synchronize_output(postprocess_output_frames)
            if state.config.postprocess_comparison:
                (
                    output_frames,
                    state.comparison_pending_generated_frames,
                ) = compose_postprocess_comparison(
                    pending_generated_frames=state.comparison_pending_generated_frames,
                    generated_frames=generated_frames,
                    postprocessed_frames=postprocess_output_frames,
                    final_chunk=final_chunk,
                    output_layout=state.session_desc.output_layout,
                )
                _synchronize_output(output_frames)
            elif state.postprocess_enabled:
                output_frames = postprocess_output_frames
            else:
                output_frames = _resize_raw_for_presentation(
                    generated_frames,
                    layout=state.session_desc.output_layout,
                    height=state.session_desc.video_height,
                    width=state.session_desc.video_width,
                )
        step_completed_at = time.perf_counter()
        postprocess_step_wall_s = step_completed_at - postprocess_started_at
        postprocess_output_frame_count = _video_frame_count(
            postprocess_output_frames,
            layout=state.session_desc.output_layout,
        )
        output_frame_count = _video_frame_count(
            output_frames,
            layout=state.session_desc.output_layout,
        )
        if (
            state.postprocess_enabled or state.config.postprocess_comparison
        ) and output_frame_count <= 0:
            raise RuntimeError(
                "Post-processing produced no frames for the Cam2V presentation. "
                "Use a postprocess chunk size no larger than the model cadence."
            )

        state.blocks_generated += 1
        state.ar_index += 1
        state.frames_generated += frame_count
        metrics.update(
            {
                "model_step_wall_s": model_step_wall_s,
                "game_capture_wall_s": game_capture_wall_s,
                "generate_wall_s": generate_wall_s,
                "chunk_fps": frame_count / model_step_wall_s,
                "model_loop_wall_s": step_completed_at - step_started_at,
                "postprocess_step_wall_s": postprocess_step_wall_s,
                "postprocess_enabled": int(state.postprocess_enabled),
                "postprocess_comparison": int(state.config.postprocess_comparison),
                "postprocess_output_frames": postprocess_output_frame_count,
            }
        )
        if state.steady_started_at is not None:
            state.steady_frames_generated += frame_count
            steady_elapsed_s = (
                step_completed_at - state.steady_started_at - state.steady_held_s
            )
            if steady_elapsed_s > 0:
                metrics["steady_state_fps"] = (
                    state.steady_frames_generated / steady_elapsed_s
                )
            metrics["recent_model_fps"] = (
                state._recent_model_frame_rate_tracker.observe(
                    completed_at=step_completed_at,
                    frame_count=frame_count,
                    elapsed_s=model_step_wall_s,
                )
            )
        if state.config.log_model_timing:
            phase = "warmup" if ar_index < state.config.warmup_blocks else "steady"
            evaluate_s = getattr(state.game_capture, "last_evaluate_s", 0.0)
            decode_s = getattr(state.game_capture, "last_decode_s", 0.0)
            # Pipelined, so `capture` is the leftover wait after the overlap, not
            # the render cost; `eval`/`decode` are that render's own timings.
            logger.info(
                "Cam2V AR {} [{}] | {} frames | capture {:.1f} ms "
                "(eval {:.1f}, decode {:.1f}) | generate {:.1f} ms | {:.2f} fps",
                ar_index,
                phase,
                frame_count,
                game_capture_wall_s * 1_000.0,
                evaluate_s * 1_000.0,
                decode_s * 1_000.0,
                generate_wall_s * 1_000.0,
                metrics["chunk_fps"],
            )
        _publish_ui_status(state, metrics)
        _publish_browser_metrics(state, metrics, epoch=browser_metrics_epoch)
        # The text was composed, and encoded when per-chunk prompts are on,
        # before the generate above. The event lines still travel on their own
        # so the client shows exactly this chunk's set, replaced wholesale when
        # the next chunk's message arrives.
        prompt_message = (
            {
                "type": PROMPT_MESSAGE_TYPE,
                "static": state.static_prompt,
                "events": list(state.active_event_lines),
            },
        )
        results = [
            StepResult(
                step_index=step_index,
                output=output_frames,
                frame_count=output_frame_count,
                output_layout=state.session_desc.output_layout,
                metrics=finalize_metrics,
                client_messages=prompt_message,
            )
        ]

        if state.game_capture is not None:
            if output_frame_count > len(state.pending_game_frames):
                raise RuntimeError(
                    "Game2V requires a postprocessor that preserves frame cadence."
                )
            paired = [
                state.pending_game_frames.popleft() for _ in range(output_frame_count)
            ]
            results.append(
                StepResult(
                    step_index=step_index,
                    output=torch.from_numpy(
                        np.stack([f.ui_frame for f in paired])
                    ).permute(0, 3, 1, 2),
                    frame_count=output_frame_count,
                    output_layout=VideoTensorLayout.tchw,
                    frame_metadata=tuple(
                        envelope(f.game_slug, f.game_frame_id, f.ui_state)
                        for f in paired
                    ),
                )
            )
        return results

    def attach_event_buffer(self, event_buffer: EventBuffer, reader_id: int) -> None:
        """Let the rank-zero capture worker drain keys during generate."""
        if int(os.environ.get("RANK", "0")) != 0:
            return
        self.state.event_buffer = event_buffer
        event_buffer.register(GAME_EVENT_READER_ID)

    def is_finished(self) -> bool:
        """Return whether this rollout generated its requested blocks."""
        return self.state.blocks_generated >= self.state.config.total_blocks

    def reset(self) -> None:
        """Discard model and camera state for a new session generation."""
        state = self.state
        if state.postprocess_stream is not None:
            state.postprocess_stream.reset()
        state.cache = None
        state.ar_index = 0
        state.game_round_id = None
        state.game_scene = None
        state.dialog_held = False
        state.blocks_generated = 0
        state.frames_generated = 0
        state.input_timeline.reset(start_s=0.0)
        state.keyboard_track.reset()
        _discard_pending_game_chunk(state)
        if state.control_stream is not None:
            state.control_stream.reset()
        if state.game_capture is not None:
            state.game_capture.close()
            state.game_capture = None
        state.pending_game_frames.clear()
        state.game_timeline.reset()
        state.last_game_chunk = None
        state.last_generate_wall_s = None
        # A restarted rollout re-resolves the registry text and starts with no
        # event lines; the previous round's events must not outlive it.
        state.static_prompt = ""
        state.active_prompt = ""
        state.active_event_lines = ()
        state.encoded_prompt = ""
        state.prompt_events.clear()
        if state.event_buffer is not None:
            state.event_buffer.read(GAME_EVENT_READER_ID)
        state.pose_integrator.reset()
        state.steady_started_at = None
        state.steady_frames_generated = 0
        state.steady_held_s = 0.0
        state._recent_model_frame_rate_tracker.reset()
        state.comparison_pending_generated_frames = None

    def close(self) -> None:
        """Release session-owned tensors and finish post-processing."""
        _discard_pending_game_chunk(self.state)
        if self.state.control_stream is not None:
            self.state.control_stream.close()
            self.state.control_stream = None
        if self.state.game_capture is not None:
            self.state.game_capture.close()
            self.state.game_capture = None
        self.state.pending_game_frames.clear()
        if self.state.event_buffer is not None:
            self.state.event_buffer.unregister(GAME_EVENT_READER_ID)
            self.state.event_buffer = None
        self.state.cache = None
        self.state.comparison_pending_generated_frames = None
        if self.state.postprocess_stream is not None:
            self.state.postprocess_stream.finish()


class Game2VSession(ISession):
    """One camera-controlled rollout sharing its application's loaded model."""

    def __init__(
        self,
        *,
        pipeline: Any,
        postprocess_stream: VideoPostprocessStream | None = None,
        config: Game2VSessionConfig,
        session_desc: SessionDesc,
        use_ui: bool = True,
    ) -> None:
        """Configure one rollout without initializing model or UI resources.

        Args:
            pipeline: Application-owned model pipeline.
            postprocess_stream: Generated-video processor owned by this session.
            config: Resolved inputs and rollout controls.
            session_desc: Output dimensions, layout, and loop rates.
            use_ui: Whether to register the shared Cam2V overlay.
        """
        if config.postprocess_comparison and postprocess_stream is None:
            raise ValueError(
                "Cam2V postprocess comparison requires a configured postprocessor."
            )
        self._pipeline = pipeline
        self._postprocess_stream = postprocess_stream
        self._config = config
        self._session_desc = session_desc
        self._use_ui = use_ui
        self._panel_registered = False

    @property
    def session_desc(self) -> SessionDesc:
        """Return the resolved output dimensions and loop rates."""
        return self._session_desc

    @cached_property
    def _presentation_manager(self) -> PresentationManager:
        """Return a frame manager initialized on the model device."""
        return PresentationManager(device=self._config.device)

    def _build_control_stream(self) -> Any:
        """Build the condition encoder this model needs, if any.

        Camera-control presets leave the factory unset, so this is ``None``
        and steps are driven by poses. Read defensively because the field is
        optional wiring, not part of every caller's config contract.
        """
        factory = getattr(self._config, "control_stream_factory", None)
        if factory is None:
            return None
        return factory(self._config, self._session_desc)

    def init(self) -> None:
        """Register the UI and model-generation loops with isolated state."""
        from .game_ui import Game2VUILoop

        self.register_ui_loop(Game2VUILoop, state=None)
        ui_loop = None
        # The proxy-DINO encoder is built lazily, on the first chunk that needs
        # it, which reads as a stall right before the first frame. Build it now
        # so that cost lands here, during startup, with a log line in front.
        control_stream = self._build_control_stream()
        prepare = getattr(control_stream, "prepare", None)
        if callable(prepare):
            logger.info("Game2V: loading the condition encoder")
            prepare_started = time.perf_counter()
            prepare()
            logger.info(
                "Game2V: condition encoder ready in {:.1f}s",
                time.perf_counter() - prepare_started,
            )
        self.register_model_loop(
            Game2VModelLoop,
            state=Game2VModelState(
                pipeline=self._pipeline,
                postprocess_stream=self._postprocess_stream,
                postprocess_enabled=self._config.postprocess_enabled,
                session_desc=self._session_desc,
                config=self._config,
                keyboard_resampler=KeyboardResampler(
                    fps=self._session_desc.frames_per_second_for_step,
                ),
                pose_integrator=self._config.pose_integrator_factory(),
                ui_loop=ui_loop,
                control_stream=control_stream,
            ),
        )
        if int(os.environ.get("RANK", "0")) == 0:
            from .panel import register_game2v_panel
            from .workbench_proxy import register_workbench

            register_game2v_panel(self)
            register_workbench()
            self._panel_registered = True

    def close(self) -> None:
        """Drop the web panel, then release session-owned tensors."""
        if self._panel_registered:
            from .panel import unregister_game2v_panel
            from .workbench_proxy import unregister_workbench

            unregister_game2v_panel()
            unregister_workbench()
            self._panel_registered = False
        super().close()


MOUSE_MOVE_SENSITIVITY = 0.5
"""Browser pointer-lock movement is amplified: one full swipe across the video
(1.0 in video-width units) turns the camera half the screen width."""


def _game_input_payloads(events: UserInputEvents) -> list[dict[str, str]]:
    """Turn keyboard edges, mouse moves and HUD clicks into Chromium payloads.

    Overlay actions are click pulses rather than held keys, so each client action
    is a down immediately followed by an up. Reset/close stay with the model
    reader; they are not game keys. The runtime does not interpret the action
    name; Chromium's ``handleUiAction`` owns that vocabulary.

    Mouse move events carry a position, not a delta, so the delta comes from
    comparing with the previous event. Only moves are forwarded; buttons and
    the wheel have no game use here. Deltas are scaled down: generated frames
    trail input by seconds, so a slow, steady turn reads better than a fast
    flick.
    """
    payloads: list[dict[str, str]] = []
    last_mouse: tuple[float, float] | None = None
    for event in events.get_events():
        if isinstance(event, KeyboardUserInputEvent):
            payloads.append(
                {
                    "action": event.key.lower(),
                    "phase": (
                        "down" if event.state is KeyboardInputState.PRESSED else "up"
                    ),
                }
            )
        elif isinstance(event, MouseUserInputEvent) and event.action == "move":
            x, y = event.x, event.y
            if last_mouse is not None:
                dx, dy = x - last_mouse[0], y - last_mouse[1]
                if dx or dy:
                    payloads.append(
                        {
                            "action": "mouse-move",
                            "phase": "move",
                            "dx": round(dx * MOUSE_MOVE_SENSITIVITY, 4),
                            "dy": round(dy * MOUSE_MOVE_SENSITIVITY, 4),
                        }
                    )
            last_mouse = (x, y)
        elif isinstance(event, ClientActionUserInputEvent):
            if event.channel in ("", GAME_UI_ACTION_CHANNEL) and event.action:
                payloads.append({"action": event.action, "phase": "down"})
                payloads.append({"action": event.action, "phase": "up"})
    return payloads


def _drain_live_game_events(state: Game2VModelState) -> list[dict[str, str]]:
    """Read keys that arrived since the last capture frame, on the worker thread."""
    buffer = state.event_buffer
    if buffer is None:
        return []
    events, _generation = buffer.read(GAME_EVENT_READER_ID)
    return _game_input_payloads(events)


def _chunk_event_lines(chunk: GameChunkPacket | None) -> list[tuple[str, str]]:
    """Event lines simulated during one chunk, in the order they happened.

    Every frame of the chunk contributes. The game keeps only the last few
    dozen events, so a line that scrolled out of the final frame's log is still
    visible on an earlier frame. Dedup is by the sentence, not the event type:
    two coins collected produce one line, but a lane change to the left and one
    to the right are different sentences and both stay. Repeated sentences keep
    their earliest occurrence.
    """
    if chunk is None:
        return []
    seen: set[str] = set()
    lines: list[tuple[str, str]] = []
    indexed: list[tuple[int, str, str]] = []
    for frame in chunk.frames:
        state = frame.ui_state
        events = state.get("events") if isinstance(state, Mapping) else None
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            text = event.get("prompt")
            event_id = event.get("id")
            if (
                not isinstance(event_type, str)
                or not event_type
                or not isinstance(text, str)
                or not text.strip()
                or isinstance(event_id, bool)
                or not isinstance(event_id, int)
            ):
                continue
            indexed.append((event_id, event_type, text.strip()))
    for event_id, event_type, text in sorted(indexed, key=lambda item: item[0]):
        del event_id
        if text in seen:
            continue
        seen.add(text)
        lines.append((event_type, text))
    return lines


def _compose_chunk_prompt(state: Game2VModelState) -> tuple[str, tuple[str, ...]]:
    """Fold the chunk just simulated into the prompt for the one being generated.

    Event lines live for ``prompt_event_chunks`` generated chunks, counting the
    one composed here. Rank zero owns the rendered frames, so it alone
    composes; the result is broadcast because a rank that conditioned on
    different text would diverge the rollout. The first chunk of a rollout has
    no predecessor and carries the static prompt alone.

    Returns:
        The text the encoder hand-off reads, and the event lines inside their
        lifetime. The lines travel separately so the client can show them on
        their own and drop them when the chunk finishes, rather than appending
        them to a growing history.
    """
    static = ""
    event_lines: tuple[str, ...] = ()
    if state.game_capture is not None:
        if not state.static_prompt:
            state.static_prompt = game_static_prompt(state.config.game_slug)
        lifetime = state.config.prompt_event_chunks
        retained = [
            (remaining - 1, event_type, text)
            for remaining, event_type, text in state.prompt_events
            if remaining > 1
        ]
        # A sentence that recurs replaces its own earlier copy with a fresh
        # lifetime. A different sentence of the same event type stays: shifting
        # left and then right are two lines, not one.
        incoming = _chunk_event_lines(state.last_game_chunk)
        restated = {text for _, text in incoming}
        retained = [entry for entry in retained if entry[2] not in restated]
        for event_type, text in incoming:
            retained.append((lifetime, event_type, text))
        state.prompt_events = retained
        static = state.static_prompt
        event_lines = tuple(text for _, _, text in retained)
    if is_distributed_initialized():
        from flashdreams.runtime_v2.distributed_sync import dist

        payload = [(static, event_lines)]
        dist.broadcast_object_list(payload, src=0)
        static, event_lines = payload[0]
        event_lines = tuple(event_lines)
        # Only rank zero resolves the registry text, yet every rank's
        # ``_encode_chunk_prompt`` decides from it whether to join the
        # embeddings broadcast. A peer left with an empty copy would see the
        # opening text as changed and broadcast alone, deadlocking the next
        # collective.
        state.static_prompt = static
    state.active_prompt = " ".join(part for part in (static, *event_lines) if part)
    state.active_event_lines = event_lines
    _encode_chunk_prompt(state)
    return static, event_lines


def _encode_chunk_prompt(state: Game2VModelState) -> None:
    """Put the composed prompt into the cross-attention cache, when enabled.

    Rank zero owns the text encoder and encodes; the embeddings are broadcast
    and every rank swaps them into its own cache. The self-attention KV cache
    is left alone, so the rollout horizon survives the swap. A prompt that has
    not changed since the chunk before is skipped, which is what keeps a quiet
    stretch of the game from re-encoding the same text. The opening chunk is
    that case too: its text is the static prompt the cache was just built from,
    and nothing has been recorded as encoded yet, so the comparison is against
    that startup text rather than against empty.
    """
    if not getattr(state.config, "dynamic_prompt", False):
        return
    text = state.active_prompt
    # The cache was built from the static prompt, so that text counts as already
    # encoded before the first chunk. A restart clears the record and the
    # rebuilt cache carries the static prompt again.
    if not state.encoded_prompt and state.blocks_generated == 0:
        state.encoded_prompt = state.static_prompt
    if text == state.encoded_prompt:
        return
    pipeline = state.pipeline
    if not hasattr(pipeline, "encode_text"):
        raise RuntimeError(
            "dynamic_prompt is enabled but the pipeline has no encode_text(); "
            "only the proxy-DINO 14B pipeline supports per-chunk prompts."
        )
    embeddings = None
    encodes = int(os.environ.get("RANK", "0")) == 0
    encode_started_at = time.perf_counter()
    if encodes:
        logger.info(
            "Game2V dynamic prompt: re-encoding {} characters for the next chunk",
            len(text),
        )
        embeddings = pipeline.encode_text([text])
    if is_distributed_initialized():
        import torch.distributed as dist

        if embeddings is None:
            embeddings = torch.empty(
                1,
                pipeline.diffusion_model.transformer.network.text_len,
                4096,
                device=state.config.device,
                dtype=torch.bfloat16,
            )
        dist.broadcast(embeddings, src=0)
    pipeline.replace_text_embeddings(state.cache, embeddings)
    state.encoded_prompt = text
    state.prompt_encode_s = time.perf_counter() - encode_started_at


def _rollout_cache_kwargs(state: "Game2VModelState") -> dict[str, int]:
    """Latent frames the first-frame condition clip must cover.

    The condition latent is encoded once and sliced per AR step, so it has to
    span every step the session will run. Only pipelines that precompute it
    (the proxy-DINO 14B one) accept the override; others keep their own cache
    signature.
    """
    pipeline = state.pipeline
    if (
        "rollout_latent_frames"
        not in inspect.signature(pipeline.initialize_cache).parameters
    ):
        return {}
    len_t = int(pipeline.diffusion_model.transformer.config.len_t)
    return {"rollout_latent_frames": state.config.total_blocks * len_t}


def _text_encoder_kwargs(state: "Game2VModelState") -> dict[str, bool]:
    """Keep the text encoder loaded on the rank that will re-encode prompts.

    ``initialize_cache`` frees the text encoder once the opening prompt is
    encoded. With ``dynamic_prompt`` the first changed prompt would then reload
    UMT5 from disk mid-rollout, which stalls that chunk for seconds. Rank zero is
    the only rank that encodes per-chunk prompts, so it keeps the encoder that
    startup already loaded; the other ranks still free theirs.
    """
    if not getattr(state.config, "dynamic_prompt", False):
        return {}
    if int(os.environ.get("RANK", "0")) != 0:
        return {}
    if (
        "release_oneshot_encoders"
        not in inspect.signature(state.pipeline.initialize_cache).parameters
    ):
        return {}
    return {"release_oneshot_encoders": False}


def _opening_prompt(state: Game2VModelState, fallback: str) -> str:
    """The text a rollout's cache is built from, before any chunk is simulated.

    That is the registry's scene description. ``fallback`` covers a registry
    entry that declares none, so the cache is never initialized from nothing.
    """
    static, _ = _compose_chunk_prompt(state)
    return static or fallback


def _pace_duration_s(state: Game2VModelState, step_index: int) -> float:
    """Stretch the in-flight chunk to the last generate, else a generate-like default."""
    del step_index
    if state.last_generate_wall_s is not None and state.last_generate_wall_s > 0:
        return float(state.last_generate_wall_s)
    return _DEFAULT_RENDER_PACE_S


def _dialog_pause_enabled(state: Game2VModelState) -> bool:
    """Pause for cards only in interactive sessions.

    A file render has nobody to dismiss a card, and some games end a round on
    their own, so holding the rollout there would never resume. The runtime
    selects ``ON_DEMAND`` for ``--mode mp4``; the interactive modes generate
    continuously. Every rank reads the same session description, so this cannot
    make ranks disagree.
    """
    return state.session_desc.presentation_mode is PresentationMode.CONTINUOUS


def _hold_for_dialog(state: Game2VModelState) -> bool:
    """Return whether the collected chunk ends with a blocking card on screen.

    Only rank zero renders the game, so only it can see the card. The answer is
    broadcast because a rank that rendered a successor its peers skipped would
    deadlock the next collective.
    """
    paused = False
    if _dialog_pause_enabled(state) and state.last_game_chunk is not None:
        frames = state.last_game_chunk.frames
        paused = bool(frames) and is_dialog_open(frames[-1].ui_state)
    if is_distributed_initialized():
        paused = broadcast_step_gate(paused)
    state.dialog_held = paused
    return paused


def _await_dialog_dismissed(
    state: Game2VModelState,
    shutdown_event: threading.Event,
) -> float:
    """Block until the client closes the card that paused this rollout.

    Each poll hands the page whatever input has arrived and redraws it without
    stepping the 60 Hz simulation, so the card can be dismissed while neither
    the simulation nor the model advances.

    Rank zero decides when to stop waiting and broadcasts that decision, for the
    same reason it decides the hold itself: it owns both the browser and the
    client whose departure sets ``shutdown_event``. The loop is left on that one
    broadcast answer rather than on any rank's local view, so every rank runs the
    same number of iterations and the same number of collectives. A rank that
    left an iteration early would leave its peers broadcasting into the
    collectives of the generate it had already started.

    Returns:
        Seconds spent waiting, for the caller to exclude from the step time the
        presentation cadence is derived from. Zero when nothing was held.
    """
    if not state.dialog_held:
        return 0.0
    logger.info("Game2V paused: a client card is open; generation is held.")
    held_since = time.perf_counter()
    while True:
        keep_waiting = False
        if state.game_capture is not None:
            probe = state.game_capture.probe(
                drain_events=lambda: _drain_live_game_events(state)
            )
            keep_waiting = is_dialog_open(probe.get("state"))
        # A requested game switch ends the hold: the switch is applied at the
        # top of the next step, which a held step would otherwise never reach
        # once the client that could dismiss the card has left.
        if getattr(state, "pending_game_slug", None) or getattr(
            state, "pending_home", False
        ):
            keep_waiting = False
        # A run that is shutting down stops waiting, so a closed tab cannot
        # strand the model thread behind a card nobody is left to dismiss.
        if shutdown_event.is_set():
            keep_waiting = False
        if is_distributed_initialized():
            keep_waiting = broadcast_step_gate(keep_waiting)
        if not keep_waiting:
            break
        shutdown_event.wait(_DIALOG_POLL_SECONDS)
    state.dialog_held = False
    held_s = time.perf_counter() - held_since
    logger.info(
        "Game2V resumed after {:.1f} s: generation continues past the card.",
        held_s,
    )
    return held_s


def _anchor_frame(state: Game2VModelState) -> torch.Tensor:
    """Load the i2v anchor frame the rollout (re)starts from.

    Cached on the state: a restart rebuilds the rollout cache from this same
    frame, and the proxy-DINO pipeline reuses its one-shot encoder outputs
    only when it sees the identical tensor again. Reloading and resizing the
    image on every restart would both repeat that work and defeat the reuse.
    """
    cached = state.anchor_frame
    if cached is not None:
        return cached
    model_height = state.config.model_video_height or state.session_desc.video_height
    model_width = state.config.model_video_width or state.session_desc.video_width
    import math
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    pixels = (
        torch.from_numpy(
            np.array(
                Image.open(state.config.conditioning.first_frame_path).convert("RGB")
            )
        )
        .permute(2, 0, 1)
        .float()
    )
    scale = max(model_height / pixels.shape[1], model_width / pixels.shape[2])
    pixels = F.interpolate(
        pixels[None],
        size=(math.ceil(pixels.shape[1] * scale), math.ceil(pixels.shape[2] * scale)),
        mode="bilinear",
        align_corners=False,
    )[0]
    top, left = (
        (pixels.shape[1] - model_height) // 2,
        (pixels.shape[2] - model_width) // 2,
    )
    frame = (
        (
            pixels[:, top : top + model_height, left : left + model_width].contiguous()
            / 127.5
            - 1
        )
        .unsqueeze(0)
        .to(device=state.config.device, dtype=state.config.first_frame_dtype)
    )
    state.anchor_frame = frame
    return frame


def _scene_id(chunk: GameChunkPacket | None) -> str | None:
    """Return the ``scene`` the chunk ends on, or ``None`` if it publishes none."""
    if chunk is None or not chunk.frames:
        return None
    ui_state = chunk.frames[-1].ui_state
    if not isinstance(ui_state, Mapping):
        return None
    scene = ui_state.get("scene")
    return scene if isinstance(scene, str) and scene else None


def _scene_cover_path(slug: str, scene: str):
    """``<games>/<slug>/shots/<scene>/cover.png`` if the game ships it, else ``None``."""
    from .game14b import game_cover_path

    if "/" in scene or "\\" in scene or scene.startswith("."):
        return None
    path = game_cover_path(slug).parent / "shots" / scene / "cover.png"
    return path if path.is_file() else None


def _round_id(chunk: GameChunkPacket | None) -> int | None:
    """Return the ``roundId`` the chunk ends on, or ``None`` if it publishes none.

    The last frame is the one that matters: a restart pressed partway through a
    chunk leaves the frames after it in the new round, so the chunk ends there.
    """
    if chunk is None or not chunk.frames:
        return None
    ui_state = chunk.frames[-1].ui_state
    if not isinstance(ui_state, Mapping):
        return None
    round_id = ui_state.get("roundId")
    if isinstance(round_id, bool) or not isinstance(round_id, int):
        return None
    return round_id


def _apply_pending_game_switch(state: Game2VModelState) -> None:
    """Honor a rank-zero panel request to change games, on every rank together.

    The HTTP handler only writes :attr:`Game2VModelState.pending_game_slug` on
    rank zero. The actual swap happens here, on the model thread, so NCCL
    collectives stay in lockstep: every rank hears the same slug, every rank
    rewinds, and the next generate uses one game.
    """
    requested = state.pending_game_slug
    restart = getattr(state, "pending_game_restart", False)
    if is_distributed_initialized():
        from flashdreams.runtime_v2.distributed_sync import dist

        payload = [(requested, restart)]
        dist.broadcast_object_list(payload, src=0)
        requested, restart = payload[0]
    state.pending_game_slug = None
    state.pending_game_restart = False
    if not requested or (requested == state.config.game_slug and not restart):
        return
    _switch_game(state, requested)


def _end_rollout(state: Game2VModelState) -> None:
    """Rewind everything the current rollout owns, leaving the game unchanged.

    Shared by the two ways a rollout ends: switching to another game, and
    returning to the home screen. The queued proxy-DINO encode belongs to that
    state too, so it goes with the rest -- a step that resumed while a stale
    encode was still queued for its AR index would fail outright.
    """
    _reset_browser_metrics(state)
    _discard_pending_game_chunk(state)
    control_stream = getattr(state, "control_stream", None)
    if control_stream is not None:
        control_stream.reset()
    if state.game_capture is not None:
        state.game_capture.close()
        state.game_capture = None
    state.pending_game_frames.clear()
    state.dialog_held = False
    state.cache = None
    state.anchor_frame = None
    state.prompt_events.clear()
    state.static_prompt = ""
    state.active_prompt = ""
    state.active_event_lines = ()
    state.encoded_prompt = ""
    state.ar_index = 0
    state.blocks_generated = 0
    state.frames_generated = 0
    state.game_round_id = None
    state.game_scene = None
    state.last_game_chunk = None
    state.last_generate_wall_s = None
    state.input_timeline.reset(start_s=0.0)
    state.keyboard_track.reset()
    state.pose_integrator.reset()
    state.steady_started_at = None
    state.steady_frames_generated = 0
    state.steady_held_s = 0.0
    state._recent_model_frame_rate_tracker.reset()
    state.comparison_pending_generated_frames = None
    if state.postprocess_stream is not None:
        state.postprocess_stream.reset()
    if state.event_buffer is not None:
        state.event_buffer.read(GAME_EVENT_READER_ID)


def _switch_game(state: Game2VModelState, slug: str) -> None:
    """Rebuild the rollout around ``slug`` without changing resolution or fps.

    Closes the capture worker, drops the i2v cache and the previous game's
    prompt/cover, and starts a new :class:`GameTimeline`. Same rewind as a
    round restart, plus a new first frame so the next generate is i2v from
    the new cover rather than the old one.
    """
    from .game14b import resolve_game14b_conditioning

    logger.info("Game2V switch: leaving {} for {}.", state.config.game_slug, slug)
    _end_rollout(state)
    model_width = state.config.model_video_width or state.session_desc.video_width
    model_height = state.config.model_video_height or state.session_desc.video_height
    conditioning = resolve_game14b_conditioning(
        {
            "game": slug,
            "pixel_width": model_width,
            "pixel_height": model_height,
        }
    )
    state.config = replace(state.config, game_slug=slug, conditioning=conditioning)
    state.game_timeline = GameTimeline(slug)


def _enter_start_gate(state: Game2VModelState) -> bool:
    """Decide, on every rank together, whether this step should wait for a game.

    Rank zero owns the decision because only it sees the HTTP requests, and the
    broadcast is what keeps the ranks in lockstep: a rank that entered the wait
    alone would start broadcasting into collectives its peers had already left
    for the generate. The local ``started`` flag is deliberately not the
    decision, only an input to rank zero's.
    """
    if not state.config.wait_for_start:
        # Identical on every rank (it comes from argv), so leaving early here
        # costs no collective and cannot desynchronise the ranks.
        return False
    enter = not state.started or getattr(state, "pending_home", False)
    if is_distributed_initialized():
        # Only the broadcast answer is acted on, never a rank's own view: a
        # rank that entered the wait on its own would broadcast into
        # collectives its peers had already passed.
        enter = broadcast_step_gate(enter)
    if enter:
        state.pending_home = False
        state.start_requested = False
        state.started = False
        # The rollout is over, so it is rewound here rather than left to the
        # next switch: picking the same game again is a no-op switch, and this
        # rollout's queued encode and cache must not outlive the wait.
        _end_rollout(state)
    return enter


def _await_start(state: Game2VModelState, shutdown_event: threading.Event) -> float:
    """Block the first step until the web panel picks a game.

    Rank zero decides and broadcasts each poll, as in
    :func:`_await_dialog_dismissed`, so every rank leaves on the same iteration.

    Returns:
        Seconds spent waiting.
    """
    logger.info("Game2V waiting: choose a game in the workbench to start rendering.")
    since = time.perf_counter()
    while True:
        keep_waiting = not state.start_requested and not shutdown_event.is_set()
        if is_distributed_initialized():
            keep_waiting = broadcast_step_gate(keep_waiting)
        if not keep_waiting:
            break
        shutdown_event.wait(_DIALOG_POLL_SECONDS)
    waited = time.perf_counter() - since
    logger.info(
        "Game2V starting {} after {:.1f} s.",
        state.pending_game_slug or state.config.game_slug,
        waited,
    )
    return waited


def request_game_switch(
    state: Game2VModelState, slug: str, *, restart: bool = False
) -> None:
    """Ask the next model step to switch to ``slug``. Rank zero only.

    Also releases a ``wait_for_start`` gate: picking a game is the start signal.
    ``restart`` asks for a full rewind even when ``slug`` is already current, so
    "play again" begins the same game from its first frame.
    """
    _reset_browser_metrics(state)
    state.start_requested = True
    state.pending_home = False
    state.pending_game_restart = bool(restart)
    if not slug or (slug == state.config.game_slug and not restart):
        state.pending_game_slug = None
        state.pending_game_restart = False
        return
    state.pending_game_slug = slug


def request_home(state: Game2VModelState) -> None:
    """Ask the next model step to stop and wait for another game. Rank zero only.

    Re-arms the ``wait_for_start`` gate through :func:`_enter_start_gate`, which
    broadcasts the decision so every rank stops on the same step.
    """
    _reset_browser_metrics(state)
    state.pending_home = True


def _reset_browser_metrics(state: Game2VModelState) -> None:
    lock = getattr(state, "browser_metrics_lock", None)
    if lock is None:  # Legacy test states constructed without dataclass defaults.
        return
    with lock:
        state.browser_metrics_epoch += 1
        state.browser_metrics = None


def browser_metrics_snapshot(
    state: Game2VModelState,
) -> dict[str, float | int | str] | None:
    """Copy the latest completed step without reading mutable model fields."""
    lock = getattr(state, "browser_metrics_lock", None)
    if lock is None:
        return None
    with lock:
        snapshot = state.browser_metrics
        return dict(snapshot) if snapshot is not None else None


def _publish_browser_metrics(
    state: Game2VModelState,
    metrics: Mapping[str, float | int],
    *,
    epoch: int,
) -> None:
    lock = getattr(state, "browser_metrics_lock", None)
    if lock is None:
        return
    snapshot: dict[str, float | int | str] = {
        "slug": state.config.game_slug,
        "chunk_fps": float(metrics["chunk_fps"]),
        "model_step_ms": float(metrics["model_step_wall_s"]) * 1000.0,
        "completed_blocks": state.blocks_generated,
        "sampled_at": time.monotonic(),
    }
    # Device-wide used/total memory, sampled on the model thread after a
    # completed chunk. This is memory consumption, not GPU utilization.
    device = state.config.device
    if device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            snapshot["gpu_memory_used_gib"] = (total_bytes - free_bytes) / 2**30
            snapshot["gpu_memory_total_gib"] = total_bytes / 2**30
        except (RuntimeError, OSError):
            pass
    with lock:
        if epoch == state.browser_metrics_epoch:
            state.browser_metrics = snapshot


def _reanchor_rollout(state: Game2VModelState) -> bool:
    """Return whether the collected chunk entered a scene with its own first frame.

    Errors accumulate block by block and the cover stops constraining anything
    once the camera has moved to a different shot. A game that reports a
    ``scene`` id and ships a matching first frame gets the rollout rebuilt from
    that frame, exactly like a round restart but without touching the game:
    the cache is dropped, the anchor frame is replaced, and generation resumes
    at autoregressive index zero. A scene without its own frame keeps the
    current anchor, so games that publish nothing are unaffected.

    Only rank zero renders, so it decides; the answer and the scene id are
    broadcast because a rank that rebuilt its cache alone would deadlock.
    """
    scene = None
    path = None
    if state.game_capture is not None:
        reported = _scene_id(state.last_game_chunk)
        if reported is not None and reported != state.game_scene:
            path = _scene_cover_path(state.config.game_slug, reported)
            if path is not None:
                scene = reported
    if is_distributed_initialized():
        from flashdreams.runtime_v2.distributed_sync import dist

        payload = [(scene, None if path is None else str(path))]
        dist.broadcast_object_list(payload, src=0)
        scene, path_text = payload[0]
        path = None if path_text is None else Path(path_text)
    if scene is None:
        return False

    logger.info(
        "Game2V re-anchor: scene {} -> {}; rebuilding the rollout from {}.",
        state.game_scene,
        scene,
        path,
    )
    state.game_scene = scene
    state.config = replace(
        state.config,
        conditioning=replace(state.config.conditioning, first_frame_path=path),
    )
    state.anchor_frame = None
    _rewind_rollout(state)
    return True


def _restart_rollout(state: Game2VModelState) -> bool:
    """Return whether the collected chunk belongs to a round the rollout has not seen.

    A game restart resets the game in place, so the chunk that contains it ends on
    a new ``roundId`` and its renders are the new round's opening frames. They
    cannot condition a rollout that has been generating the previous round, so the
    rollout goes back to the start of the i2v flow: the cache is dropped for the
    caller to rebuild, and the autoregressive index returns to zero.

    Only rank zero renders, so only it can see the change. The answer is broadcast
    for the same reason a card hold is: a rank that rewound a rollout its peers
    kept would deadlock on the next collective.

    The first chunk of a session records its round without rewinding. There is no
    rollout yet to rewind, and the cache built for it is already the fresh one.
    """
    restarted = False
    if state.game_capture is not None:
        round_id = _round_id(state.last_game_chunk)
        if round_id is not None:
            restarted = (
                state.game_round_id is not None and round_id != state.game_round_id
            )
            state.game_round_id = round_id
    if is_distributed_initialized():
        restarted = broadcast_step_gate(restarted)
    if not restarted:
        return False

    _reset_browser_metrics(state)
    logger.info(
        "Game2V restart: round changed to {}; clearing the rollout cache and "
        "regenerating from the first frame.",
        state.game_round_id,
    )
    if state.game_scene is not None:
        state.game_scene = None
        from .game14b import game_cover_path

        # A restart regenerates the round from its opening, which is the cover.
        state.config = replace(
            state.config,
            conditioning=replace(
                state.config.conditioning,
                first_frame_path=game_cover_path(state.config.game_slug),
            ),
        )
        state.anchor_frame = None
    _rewind_rollout(state)
    return True


def _begin_game_chunk(
    state: Game2VModelState, step_index: int, *, duration_s: float = 0.0
) -> Any:
    """Start rendering the chunk for ``step_index`` in the background.

    Returns a future on rank zero, which owns the browser, and the timeline
    packet itself elsewhere: other ranks need the tick math advanced identically
    but have nothing to render to. Live keys are drained by the worker between
    frames; they are not frozen onto the packet.
    """
    chunk = state.game_timeline.begin_chunk(
        int(state.pipeline.get_num_output_frames(step_index)),
        state.session_desc.frames_per_second_for_step,
    )
    if int(os.environ.get("RANK", "0")) != 0:
        # Ranks that do not render still register the step, so their condition
        # encoder stays in lockstep and its broadcast has a matching receiver.
        if state.control_stream is not None:
            state.control_stream.submit(step_index, chunk)
        return chunk
    if state.game_capture is None:
        from .game_renderer import GameCaptureWorker

        state.game_capture = GameCaptureWorker(
            state.config.game_slug,
            width=state.config.model_video_width or state.session_desc.video_width,
            height=state.config.model_video_height or state.session_desc.video_height,
        )
    pending = state.game_capture.submit(
        chunk,
        duration_s=duration_s,
        drain_events=lambda: _drain_live_game_events(state),
    )
    # Chained on the render future, not on the later collect: this chunk is
    # rendered while its predecessor denoises, so the condition encode runs
    # there too instead of in front of the step that consumes it.
    if state.control_stream is not None:
        state.control_stream.submit(step_index, pending)
    return pending


def _collect_game_chunk(state: Game2VModelState) -> GameChunkPacket:
    """Wait for the in-flight render and return its fully decoded chunk."""
    pending = state.pending_game_chunk
    state.pending_game_chunk = None
    if pending is None:
        raise RuntimeError("Game2V has no pending game chunk to generate from.")
    return pending.result() if isinstance(pending, Future) else pending


def _discard_pending_game_chunk(state: Game2VModelState) -> None:
    """Drain an in-flight render so its browser cannot outlive the step."""
    pending = state.pending_game_chunk
    state.pending_game_chunk = None
    if not isinstance(pending, Future):
        return
    if pending.cancel():
        return
    try:
        pending.result()
    except Exception:
        logger.opt(exception=True).debug("Discarded Game2V chunk render failed.")


def _publish_ui_status(
    state: Game2VModelState,
    metrics: Mapping[str, float | int],
) -> None:
    """Send immutable model status to the UI loop without sharing state."""
    ui_loop = state.ui_loop
    if ui_loop is None:
        return
    recent_model_fps = metrics.get("recent_model_fps")
    status = Cam2VUIStatus(
        completed_blocks=state.blocks_generated,
        frames_generated=state.frames_generated,
        chunk_fps=float(metrics["chunk_fps"]),
        recent_model_rate_snapshot=(
            None
            if recent_model_fps is None
            else state._recent_model_frame_rate_tracker.snapshot()
        ),
        model_step_wall_s=float(metrics["model_step_wall_s"]),
    )
    invoke_async(
        ui_loop,
        lambda ui_state, status=status: ui_state.update_status(status),
    )


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    """Keep numeric pipeline metrics accepted by the v2 result contract."""
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _video_frame_count(
    video: torch.Tensor,
    *,
    layout: VideoTensorLayout,
) -> int:
    """Return a v2 video tensor's temporal extent."""
    time_dim = {
        VideoTensorLayout.tchw: 0,
        VideoTensorLayout.btchw: 1,
        VideoTensorLayout.bcthw: 2,
        VideoTensorLayout.bvtchw: 2,
    }[layout]
    return int(video.shape[time_dim])


def _synchronize_output(output: torch.Tensor) -> None:
    """Wait for CUDA output so step wall time covers completed production."""
    if output.is_cuda:
        torch.cuda.current_stream(output.device).synchronize()


def _concatenate_video(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    layout: VideoTensorLayout,
) -> torch.Tensor:
    """Join postprocess output and its final tail in presentation order."""
    time_dim = {
        VideoTensorLayout.tchw: 0,
        VideoTensorLayout.btchw: 1,
        VideoTensorLayout.bcthw: 2,
        VideoTensorLayout.bvtchw: 2,
    }[layout]
    if first.shape[time_dim] == 0:
        return second
    return torch.cat((first, second), dim=time_dim)


def _resize_raw_for_presentation(
    frames: torch.Tensor,
    *,
    layout: VideoTensorLayout,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize raw video frames to the fixed presentation dimensions."""
    canonical = to_bvtchw(frames, layout=layout.value)
    if canonical.shape[-2:] == (height, width):
        return frames
    resized = F.interpolate(
        canonical.flatten(0, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return from_bvtchw(
        resized.reshape(*canonical.shape[:-2], height, width),
        layout=layout.value,
    )


__all__ = [
    "GAME_EVENT_READER_ID",
    "Game2VModelState",
    "Game2VModelLoop",
    "Game2VSession",
    "Game2VSessionConfig",
    "CameraControlInput",
    "request_game_switch",
]


def _rewind_rollout(state: Game2VModelState) -> None:
    """Drop the rollout cache and everything indexed by it; shared by restart and re-anchor."""
    # Nothing rendered ahead can survive: it was simulated for the old round, and
    # the frame count at index zero differs from the one it was begun with.
    _discard_pending_game_chunk(state)
    # Queued condition encodes belong to the old round and are indexed by the
    # AR steps the new rollout is about to reuse from zero.
    control_stream = getattr(state, "control_stream", None)
    if control_stream is not None:
        control_stream.reset()
    state.pending_game_frames.clear()
    # The successor of a chunk that ended on a card was never begun, so a restart
    # that closed that card must not be read as another pause.
    state.dialog_held = False
    state.cache = None
    # The chunk that straddles the restart is dropped below, and its events
    # belong to neither round, so the new rollout opens on the static prompt.
    state.prompt_events.clear()
    state.active_prompt = ""
    state.active_event_lines = ()
    # The rebuilt cache carries the static prompt again, so the next chunk
    # compares against it rather than against the previous round's text.
    state.encoded_prompt = ""
    state.ar_index = 0
    state.blocks_generated = 0
    state.frames_generated = 0
    state.input_timeline.reset(start_s=0.0)
    state.keyboard_track.reset()
    state.pose_integrator.reset()
    state.steady_started_at = None
    state.steady_frames_generated = 0
    state.steady_held_s = 0.0
    state._recent_model_frame_rate_tracker.reset()
    state.comparison_pending_generated_frames = None
    if state.postprocess_stream is not None:
        state.postprocess_stream.reset()

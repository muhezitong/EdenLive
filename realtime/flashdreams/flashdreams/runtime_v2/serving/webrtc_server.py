# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# Modified by Vast AI for EdenLive.
# SPDX-License-Identifier: Apache-2.0

"""Standalone WebRTC server used by the v2 client window."""

from __future__ import annotations

import asyncio
import json
import math
import mimetypes
import os
import re
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from loguru import logger

from flashdreams.runtime_v2.cuda_utils import resolve_cuda_device
from flashdreams.runtime_v2.serving.extensions import (
    dispatch_extension,
    serve_extension_manifest,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    ClientActionUserInputEvent,
    CloseUserInputEvent,
    FocusUserInputEvent,
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    QueryStringUserInputEvent,
    ResetUserInputEvent,
    TouchUserInputEvent,
    UserInputEvent,
    XRControllerUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_WEB_RESOURCES = files("flashdreams.runtime_v2.serving").joinpath("web")
_BROWSER_PAGE = _WEB_RESOURCES.joinpath("index.html").read_text(encoding="utf-8")
_BROWSER_SCRIPT = _WEB_RESOURCES.joinpath("app.js").read_text(encoding="utf-8")

_GAME_OVERLAY_SLUG = re.compile(r"^[a-z0-9-]+$")
_GAME_OVERLAY_FILES = {
    "overlay.js": "text/javascript",
    "overlay.css": "text/css",
}
"""Per-game overlay modules served from ``EDEN_GAMES_ROOT``; not part of the wheel."""

_INTERACTIVE_FRAME_QUEUE_SIZE = 2
"""Send-ready frames retained while the media sender is temporarily busy."""

_TRANSFER_STREAM_PRIORITY = -1
"""Portable high-priority CUDA stream request for interactive output copies."""

_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 2.0
"""Bound shutdown time spent waiting for aiortc's next sender request."""

_AIOICE_COMPLETED_RETRY_MESSAGE = "Exception in callback Transaction.__retry()"
"""Asyncio callback message produced by the completed STUN retry race."""

_CLIENT_HEARTBEAT_TIMEOUT_SECONDS = 3.0
"""Drop the client if the controls channel is silent this long.

The browser pings every 1s. Three missed intervals treats a closed tab as
gone without flapping on one delayed timer. ICE ``disconnected`` is not
enough on its own: TURN relay often stays there after the page is closed.
"""


_CLOSED_DATAGRAM_ATTRIBUTES = frozenset({"sendto", "call_exception_handler"})
"""Attributes a closed asyncio datagram transport fails on once torn down.

``_call_connection_lost`` sets the transport's ``_sock`` and ``_loop`` to
``None``. A later ``sendto`` fails on ``_sock.sendto``, and the fatal-error
path then fails again on ``_loop.call_exception_handler``.
"""


def _is_closed_transport_retry(exception: BaseException | None) -> bool:
    """Return whether ``exception`` is a STUN retry on a torn-down transport."""
    return (
        type(exception) is AttributeError
        and exception.obj is None
        and exception.name in _CLOSED_DATAGRAM_ATTRIBUTES
    )


def _handle_webrtc_loop_exception(
    loop: asyncio.AbstractEventLoop,
    context: dict[str, Any],
) -> None:
    """Suppress the two ``aioice`` STUN retry races after a transaction ends.

    Both are a ``Transaction.__retry`` timer outliving its transaction: it
    fires after the response already completed the future, or after ICE
    teardown closed the UDP socket. Nothing is lost either way; the retry had
    no live peer left to reach.
    """
    # TODO: Remove after https://github.com/aiortc/aioice/pull/100 is released.
    exception = context.get("exception")
    if (
        isinstance(context.get("handle"), asyncio.TimerHandle)
        and context.get("message") == _AIOICE_COMPLETED_RETRY_MESSAGE
        and (
            type(exception) is asyncio.InvalidStateError
            or _is_closed_transport_retry(exception)
        )
    ):
        return
    loop.default_exception_handler(context)


_TURN_HOST_ENV = "FLASHDREAMS_TURN_HOST"
_TURN_BROWSER_HOST_ENV = "FLASHDREAMS_TURN_BROWSER_HOST"
"""Optional TURN host handed to the browser when it differs from the server's.

A browser that reaches the relay through a tunnel (``kubectl port-forward`` or
``ssh -L``) dials ``127.0.0.1`` while the server allocates over the pod IP.
Unset, the browser gets :data:`_TURN_HOST_ENV` as before.
"""
_TURN_PORT_ENV = "FLASHDREAMS_TURN_PORT"
_TURN_USERNAME_ENV = "FLASHDREAMS_TURN_USERNAME"
_TURN_CREDENTIAL_ENV = "FLASHDREAMS_TURN_CREDENTIAL"
"""Environment variables configuring the optional TURN relay."""


def _turn_host() -> str | None:
    """Return the configured TURN host, or ``None`` to disable TURN."""
    return os.environ.get(_TURN_HOST_ENV) or None


def _turn_credential() -> str:
    """Return the configured TURN credential.

    No default: a shared fallback secret would be published with the source and
    would silently authenticate against anyone's relay.
    """
    credential = os.environ.get(_TURN_CREDENTIAL_ENV)
    if not credential:
        raise RuntimeError(
            f"{_TURN_CREDENTIAL_ENV} is unset but {_TURN_HOST_ENV} names a TURN "
            "relay; set the relay's credential, or unset the host to disable TURN."
        )
    return credential


def _server_ice_servers() -> list[RTCIceServer] | None:
    """Return the TURN relay the server allocates over the VPC, if configured."""
    host = _turn_host()
    if host is None:
        return None
    port = int(os.environ.get(_TURN_PORT_ENV, "3478"))
    return [
        RTCIceServer(
            f"turn:{host}:{port}",
            username=os.environ.get(_TURN_USERNAME_ENV, "flashdreams"),
            credential=_turn_credential(),
        )
    ]


def _browser_ice_servers() -> list[dict[str, str]]:
    """Return the TURN relay the browser reaches over TCP, for JS injection."""
    host = _turn_host()
    if host is None:
        return []
    host = os.environ.get(_TURN_BROWSER_HOST_ENV) or host
    port = int(os.environ.get(_TURN_PORT_ENV, "3478"))
    return [
        {
            "urls": f"turn:{host}:{port}?transport=tcp",
            "username": os.environ.get(_TURN_USERNAME_ENV, "flashdreams"),
            "credential": _turn_credential(),
        }
    ]


_WILDCARD_BIND_HOSTS = ("", "0.0.0.0", "::", "[::]")


def _display_host(host: str) -> str:
    """Return a host a browser can reach, resolving a wildcard bind address.

    ``--host 0.0.0.0`` (or ``::``) binds every interface, so the bind address
    itself is not a usable destination. Report the local interface that carries
    the default route instead, so printed connect URLs are meaningful. When the
    route cannot be discovered, fall back to the literal bind host.
    """
    if host not in _WILDCARD_BIND_HOSTS:
        return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Connecting a UDP socket only selects the route; no packet is sent.
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return host


class _PinnedRGBFrameBuffer:
    """One reusable host frame for the synchronous CUDA materializer."""

    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None
        self._frame: torch.Tensor | None = None
        self._retired_frames: list[torch.Tensor] = []

    def get(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Return pinned storage with the session's fixed output shape."""
        if self._shape is None:
            self._shape = shape
        elif self._shape != shape:
            raise ValueError(
                f"Pinned RGB frame shape changed from {self._shape} to {shape}."
            )
        if self._frame is None:
            self._frame = torch.empty(
                shape,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
        return self._frame

    def retire(self) -> None:
        """Keep storage alive when CUDA cannot prove a failed copy completed."""
        if self._frame is not None:
            self._retired_frames.append(self._frame)
            self._frame = None

    def close(self) -> None:
        """Release storage after the owning transfer streams are synchronized."""
        self._frame = None
        self._retired_frames.clear()


_QUARANTINED_CUDA_TRANSFERS: list[
    tuple[tuple[torch.cuda.Stream, ...], _PinnedRGBFrameBuffer]
] = []
"""Failed CUDA transfers retained so their pinned storage cannot be reused."""

_QUARANTINED_CUDA_TRANSFERS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _QueuedRGBFrame:
    """One owned video frame waiting for aiortc handoff."""

    frame: VideoFrame
    """Owned, send-ready video frame."""

    enqueued_at: float
    """Monotonic timestamp at which ``write`` admitted this frame."""


class _VideoTrack(MediaStreamTrack):
    """Video track whose frames are supplied by the server."""

    kind = "video"

    def __init__(self, frames_per_second: int) -> None:
        """Configure immediate delivery with a bounded unsent-frame mailbox.

        Args:
            frames_per_second: RTP timestamp resolution.
        """
        super().__init__()
        self._frames_per_second = frames_per_second
        self._time_base = Fraction(1, frames_per_second)
        self._sender_loop = asyncio.get_running_loop()
        self._frames: deque[_QueuedRGBFrame] = deque()
        self._frame_available = asyncio.Event()
        self._sender_drained = threading.Event()
        self._sender_drained.set()
        self._recv_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._wake_scheduled = False
        self._enqueued_count = 0
        self._handed_off_count = 0
        self._dropped_for_lag = 0
        self._discarded_on_session_change_count = 0
        self._discarded_on_close_count = 0
        self._first_enqueued_at: float | None = None
        self._next_pts = 0
        self._frame_in_flight = False
        self._closed = False

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return counters and queue age without blocking frame delivery."""
        with self._state_lock:
            oldest_queue_age_s = (
                0.0
                if not self._frames
                else max(0.0, time.monotonic() - self._frames[0].enqueued_at)
            )
            sender_metrics = {
                "webrtc_sender_queue_depth_count": len(self._frames),
                "webrtc_sender_queue_capacity_count": (_INTERACTIVE_FRAME_QUEUE_SIZE),
                "webrtc_sender_enqueued_count": self._enqueued_count,
                "webrtc_sender_handed_off_count": self._handed_off_count,
                "webrtc_sender_dropped_for_lag_count": self._dropped_for_lag,
                "webrtc_sender_discarded_on_session_change_count": (
                    self._discarded_on_session_change_count
                ),
                "webrtc_sender_discarded_on_close_count": (
                    self._discarded_on_close_count
                ),
                "webrtc_sender_oldest_queue_age_s": oldest_queue_age_s,
            }
        return sender_metrics

    def enqueue(self, frame: VideoFrame) -> bool:
        """Synchronously admit one real frame and wake the sender.

        Returns:
            Whether the frame was admitted. A closed track rejects it.
        """
        with self._state_lock:
            if self._closed:
                return False
            enqueued_at = time.monotonic()
            queued_frame = _QueuedRGBFrame(
                frame=frame,
                enqueued_at=enqueued_at,
            )
            if len(self._frames) >= _INTERACTIVE_FRAME_QUEUE_SIZE:
                self._frames.popleft()
                self._dropped_for_lag += 1
            self._frames.append(queued_frame)
            self._sender_drained.clear()
            self._enqueued_count += 1
            schedule_wake = not self._wake_scheduled
            self._wake_scheduled = True
        if schedule_wake:
            # The UI thread enqueues most frames, so the wake has to cross onto
            # the sender loop. A frame replayed to a replacement client is
            # enqueued from that loop already, where call_soon_threadsafe only
            # runs after the current callback returns.
            try:
                on_sender_loop = self._sender_loop is asyncio.get_running_loop()
            except RuntimeError:
                on_sender_loop = False
            if on_sender_loop:
                self._finish_enqueue()
            else:
                self._sender_loop.call_soon_threadsafe(self._finish_enqueue)
        return True

    async def recv(self) -> VideoFrame:
        """Serialize aiortc demand for the bounded frame queue."""
        async with self._recv_lock:
            with self._state_lock:
                if self._frame_in_flight:
                    self._frame_in_flight = False
                    if not self._frames:
                        self._sender_drained.set()
            return await self._recv_one()

    def wait_until_drained(self, timeout_s: float) -> bool:
        """Wait until aiortc requests another frame after the latest handoff."""
        return self._sender_drained.wait(timeout_s)

    async def start_session(self) -> None:
        """Discard queued frames that belong to the replaced session."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot start a session on a closed video track.")
            discarded_count = len(self._frames)
            self._frames.clear()
            self._first_enqueued_at = None
            self._discarded_on_session_change_count += discarded_count
            if not self._frame_in_flight:
                self._sender_drained.set()
        self._frame_available.clear()

    async def _recv_one(self) -> VideoFrame:
        """Return the next send-ready frame immediately when aiortc requests it."""
        if self._closed:
            raise MediaStreamError
        queued_frame = await self._next_queued_frame()
        if self._first_enqueued_at is None:
            self._first_enqueued_at = queued_frame.enqueued_at
        elapsed = queued_frame.enqueued_at - self._first_enqueued_at
        pts = max(self._next_pts, round(elapsed * self._frames_per_second))

        video_frame = queued_frame.frame
        video_frame.pts = pts
        video_frame.time_base = self._time_base
        self._next_pts = pts + 1
        with self._state_lock:
            self._frame_in_flight = True
            self._handed_off_count += 1
        return video_frame

    async def close(self) -> None:
        """Stop the track and release a pending receiver."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._wake_scheduled = False
            discarded_count = len(self._frames)
            self._frames.clear()
            self._frame_in_flight = False
            self._sender_drained.set()
            self._discarded_on_close_count += discarded_count
        self._frame_available.set()
        self.stop()

    def _finish_enqueue(self) -> None:
        """Wake a waiting receiver after a cross-thread enqueue."""
        with self._state_lock:
            self._wake_scheduled = False
        self._frame_available.set()

    async def _next_queued_frame(self) -> _QueuedRGBFrame:
        while True:
            with self._state_lock:
                if self._frames:
                    queued_frame = self._frames.popleft()
                    if not self._frames:
                        self._frame_available.clear()
                    return queued_frame
                if self._closed:
                    raise MediaStreamError
                self._frame_available.clear()
            await self._frame_available.wait()


class _AuxiliaryVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    async def recv(self):
        await self.owner._paired_ready.wait()
        if self.owner._closed:
            raise MediaStreamError
        frame = self.owner._paired_frame
        self.owner._paired_ready.clear()
        self.owner._paired_consumed.set()
        return frame


class _PairedVideoTrack(_VideoTrack):
    """Drop queued pairs together and hand both tracks the same timestamps."""

    def __init__(self, fps):
        super().__init__(fps)
        self._paired_ready = asyncio.Event()
        self._paired_consumed = asyncio.Event()
        self._paired_consumed.set()
        self._paired_frame = None
        self.auxiliary = _AuxiliaryVideoTrack(self)

    async def _recv_one(self):
        await self._paired_consumed.wait()
        if self._closed:
            raise MediaStreamError
        queued = await self._next_queued_frame()
        main, auxiliary = queued.frame
        if self._first_enqueued_at is None:
            self._first_enqueued_at = queued.enqueued_at
        pts = max(self._next_pts, round((queued.enqueued_at - self._first_enqueued_at) * self._frames_per_second))
        for frame in (main, auxiliary):
            frame.pts = pts
            frame.time_base = self._time_base
        self._next_pts = pts + 1
        self._paired_frame = auxiliary
        self._paired_consumed.clear()
        self._paired_ready.set()
        with self._state_lock:
            self._frame_in_flight = True
            self._handed_off_count += 1
        return main

    async def start_session(self):
        await super().start_session()
        self._paired_ready.clear()
        self._paired_frame = None
        self._paired_consumed.set()

    async def close(self):
        await super().close()
        self._paired_ready.set()
        self._paired_consumed.set()
        self.auxiliary.stop()


class WebRTCServer:
    """Own the HTTP, signaling, input buffering, and media transport."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        """
        Args:
            host: Interface on which the HTTP server listens.
            port: Listening port. Zero asks the operating system to choose one.
            startup_timeout_seconds: Maximum time to wait for server startup.

        Raises:
            RuntimeError: The server cannot start.
            TimeoutError: The server does not start before the timeout.
        """
        if not host:
            raise ValueError("host must not be empty.")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0.")

        self._host = host
        self._port = port
        self._startup_timeout_seconds = startup_timeout_seconds
        self._ice_servers = _server_ice_servers()
        self._input_callback: Callable[[UserInputEvent], None] | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._offer_lock: asyncio.Lock | None = None
        self._peer_connection: RTCPeerConnection | None = None
        self._control_channel: Any | None = None
        self._sent_cursor_options: tuple[bool, bool] | None = None
        self._session_metadata: dict[str, Any] = {"prompt": "Waiting for backend prompt…"}
        self._video_track: _VideoTrack | None = None
        self._final_video_track_metrics: dict[str, float | int] | None = None
        self._media_connected = threading.Event()
        self._session_desc: SessionDesc | None = None
        self._event_origin_ns: int | None = None
        self._transfer_streams: dict[int, torch.cuda.Stream] = {}
        self._materialization_buffer = _PinnedRGBFrameBuffer()
        self._materialization_count = 0
        self._closed = False
        self._client_connected = False
        self._last_control_at: float | None = None
        self._hide_cursor = False
        self._lock_cursor_to_window = False
        # Last frame handed to a track, replayed to a replacement client so one
        # that connects while the model is not producing still gets a picture.
        self._latest_main_frame: VideoFrame | None = None
        self._latest_auxiliary_frame: VideoFrame | None = None
        self._latest_client_messages: tuple[str, ...] = ()
        self._latest_replayed = False
        self._thread = threading.Thread(
            target=self._run_server,
            name="flashdreams-webrtc",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(startup_timeout_seconds):
            raise TimeoutError("WebRTC server did not start before the timeout.")
        if self._startup_error is not None:
            raise RuntimeError(
                "WebRTC server failed to start."
            ) from self._startup_error

    @property
    def host(self) -> str:
        """Return the interface on which the server is listening."""
        return self._host

    @property
    def port(self) -> int:
        """Return the bound server port."""
        return self._port

    @property
    def url(self) -> str:
        """Return the browser URL for this server."""
        return f"http://{_display_host(self._host)}:{self._port}/"

    def is_client_connected(self) -> bool:
        """Return whether a browser currently holds this server's client slot.

        Occupying the slot is not enough: a closed tab often leaves ICE in
        ``disconnected`` without closing the datachannel. The browser must also
        have sent a controls-channel message (heartbeat ping or input) within
        ``_CLIENT_HEARTBEAT_TIMEOUT_SECONDS``.

        Read from the model thread, so this uses only plain attributes the
        WebRTC event loop publishes rather than taking a lock.
        """
        if not self._client_connected:
            return False
        last_control_at = self._last_control_at
        if last_control_at is None:
            return False
        return (
            time.monotonic() - last_control_at
        ) < _CLIENT_HEARTBEAT_TIMEOUT_SECONDS

    def _mark_control_activity(self) -> None:
        """Record that the current browser is still sending on a datachannel."""
        self._last_control_at = time.monotonic()

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Return non-blocking sender diagnostics."""
        track = self._video_track
        if track is not None:
            sender_metrics = track.metrics_snapshot()
        elif self._final_video_track_metrics is not None:
            sender_metrics = self._final_video_track_metrics
        else:
            sender_metrics = {
                "webrtc_sender_queue_depth_count": 0,
                "webrtc_sender_queue_capacity_count": _INTERACTIVE_FRAME_QUEUE_SIZE,
                "webrtc_sender_enqueued_count": 0,
                "webrtc_sender_handed_off_count": 0,
                "webrtc_sender_dropped_for_lag_count": 0,
                "webrtc_sender_discarded_on_session_change_count": 0,
                "webrtc_sender_discarded_on_close_count": 0,
                "webrtc_sender_oldest_queue_age_s": 0.0,
            }
        return {
            **sender_metrics,
            "webrtc_sender_materialized_count": self._materialization_count,
        }

    def configure_cursor(
        self, *, hide_cursor: bool, lock_cursor_to_window: bool
    ) -> None:
        """Apply cursor options and notify an already-connected browser."""
        if not isinstance(hide_cursor, bool):
            raise TypeError("hide_cursor must be a bool.")
        if not isinstance(lock_cursor_to_window, bool):
            raise TypeError("lock_cursor_to_window must be a bool.")
        if (
            hide_cursor == self._hide_cursor
            and lock_cursor_to_window == self._lock_cursor_to_window
        ):
            return
        self._hide_cursor = hide_cursor
        self._lock_cursor_to_window = lock_cursor_to_window
        loop = self._loop
        if loop is not None and self._control_channel is not None:
            loop.call_soon_threadsafe(self._send_cursor_options)

    def _send_cursor_options(self) -> None:
        """Send current cursor options on the WebRTC event-loop thread."""
        channel = self._control_channel
        if channel is None or channel.readyState != "open":
            return
        cursor_options = (self._hide_cursor, self._lock_cursor_to_window)
        if cursor_options == self._sent_cursor_options:
            return
        channel.send(
            json.dumps(
                {
                    "type": "cursor_options",
                    "hide_cursor": cursor_options[0],
                    "lock_cursor_to_window": cursor_options[1],
                }
            )
        )

        self._sent_cursor_options = cursor_options
        channel.send(json.dumps({"type": "session_metadata", **self._session_metadata}))

    def send_client_messages(self, messages: tuple[Mapping[str, Any], ...]) -> None:
        """Forward application payloads to the client on the control channel.

        The payloads are sent verbatim, in order, as JSON text. They are not
        interpreted here: the runtime only guarantees that a client which
        receives them has already been offered the frame they accompany.

        Args:
            messages: JSON-serializable objects, each carrying its own ``type``.
        """
        if not messages:
            return
        encoded = tuple(json.dumps(message) for message in messages)
        # Retained even when there is nobody to send to: these describe the
        # state of the latest frame, so a client taking the slot over later is
        # replayed them alongside that frame.
        self._latest_client_messages = encoded
        loop = self._loop
        if loop is None or self._control_channel is None:
            return
        loop.call_soon_threadsafe(self._send_encoded_client_messages, encoded)

    def _send_encoded_client_messages(self, encoded: tuple[str, ...]) -> None:
        """Send pre-encoded client payloads on the WebRTC event-loop thread."""
        channel = self._control_channel
        if channel is None or channel.readyState != "open":
            return
        for message in encoded:
            channel.send(message)

    def _replay_latest_to_client(self) -> None:
        """Give the client that just took the slot the last frame produced.

        A takeover replaces the video track, and nothing refills its queue until
        the model loop publishes again. While a dialog pause holds generation
        that never happens, so the replacement client would wait on a first
        frame forever. The last frame is still a faithful picture of the game:
        the simulation is frozen at exactly that point, and the client messages
        that accompanied it carry the card that is on screen.

        The retained frames are the ones the previous client's encoder already
        consumed, so each copy handed out is a clone. Two copies go out because
        a codec holds its first input until a later RTP timestamp arrives; a
        single frame would never be emitted.

        Runs on the WebRTC event-loop thread, which is the only thread that
        touches the track queue and the control channel.
        """
        main = self._latest_main_frame
        if main is None or self._latest_replayed:
            return
        track = self._video_track
        if track is None:
            return
        auxiliary = self._latest_auxiliary_frame
        paired = isinstance(track, _PairedVideoTrack)
        if paired and auxiliary is None:
            return
        for _ in range(2):
            copy = _clone_video_frame(main)
            if paired:
                track.enqueue((copy, _clone_video_frame(auxiliary)))
            else:
                track.enqueue(copy)
        self._latest_replayed = True


    def _resend_latest_client_messages(self) -> None:
        """Resend the state that describes the last frame, once it can be heard.

        The frame is replayed as soon as media connects, but the control channel
        that carries the card state may not be open yet. Whichever of the two
        happens second sends it, and sending twice is harmless: the client drops
        a repeat of the same frame id.
        """
        if self._latest_client_messages:
            self._send_encoded_client_messages(self._latest_client_messages)

    def event_timestamp_us(self) -> np.uint64:
        """Return the current timestamp relative to this server's first session.

        Raises:
            RuntimeError: The server has not opened a session.
        """
        timestamp_us = self._timestamp_us()
        if timestamp_us is None:
            raise RuntimeError("Open the WebRTC server before reading its clock.")
        return timestamp_us

    def open(self, session_desc: SessionDesc) -> None:
        """Configure the server for one session's generated video.

        A replacement session may open an active server again when its stream
        format is unchanged. The peer connection stays up and frames still
        queued from the previous session are discarded.

        Args:
            session_desc: Resolved dimensions, frame rate, and tensor layout.

        Raises:
            RuntimeError: The server is closed or has no input callback.
            TimeoutError: The active video track cannot begin the replacement
                before the server timeout.
            ValueError: A replacement changes the negotiated stream format.
        """
        if self._closed:
            raise RuntimeError("Cannot open a closed WebRTC server.")
        if self._input_callback is None:
            raise RuntimeError("Register an input callback before opening WebRTC.")
        current_session_desc = self._session_desc
        if current_session_desc is not None and not _same_stream_format(
            current_session_desc,
            session_desc,
        ):
            raise ValueError(
                "A replacement WebRTC session must keep the same stream format: "
                "output layout, frame rates, width, and height."
            )

        track = self._video_track
        loop = self._loop
        if track is not None and loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._start_session_if_active(track), loop
            )
            try:
                future.result(timeout=self._startup_timeout_seconds)
            except BaseException:
                future.cancel()
                raise
        self._session_desc = session_desc
        # The retained frame belongs to the session that produced it. A
        # replacement session resets generation, so replaying it would hand the
        # client a picture of a game that no longer exists.
        self._latest_main_frame = None
        self._latest_auxiliary_frame = None
        self._latest_client_messages = ()
        self._latest_replayed = False
        if self._event_origin_ns is None:
            # Keep timestamps comparable across sessions so events buffered
            # during a handoff retain their real order.
            self._event_origin_ns = time.monotonic_ns()

    def register_input_callback(
        self, callback: Callable[[UserInputEvent], None]
    ) -> None:
        """Register the function called for each received browser event.

        Args:
            callback: Function that accepts one validated, timestamped event.

        Raises:
            RuntimeError: A callback has already been registered.
        """
        if self._input_callback is not None:
            raise RuntimeError("An input callback is already registered.")
        self._input_callback = callback

    def write(self, result: StepResult) -> None:
        """Materialize and admit one generated result to the sender mailbox.

        Args:
            result: Generated frames matching the description passed to
                :meth:`open`.

        Raises:
            RuntimeError: The server is not open or has been closed.
            ValueError: The result shape, layout, or frame count is invalid.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed WebRTC server.")
        session_desc = self._session_desc
        if session_desc is None:
            raise RuntimeError("Open the WebRTC server before writing.")
        frames = _validated_result_frames(result, session_desc)
        if result.frame_count != 1:
            raise ValueError(
                "WebRTC window writes must contain exactly one UI-composited frame."
            )
        track = self._video_track
        if track is None:
            return
        queued_frame = self._materialize_video_frame(result, frames[0])
        game_frame: VideoFrame | None = None
        if isinstance(track, _PairedVideoTrack):
            auxiliary = result.auxiliary_frames.get("game")
            if auxiliary is None:
                auxiliary = torch.zeros((3, 2, 2), dtype=torch.uint8)
            if auxiliary.is_cuda or auxiliary.dtype != torch.uint8 or auxiliary.ndim != 3 or auxiliary.shape[0] != 3:
                raise ValueError("Auxiliary video must be CPU uint8 RGB CHW.")
            game_frame = VideoFrame.from_ndarray(auxiliary.permute(1, 2, 0).contiguous().numpy(), format="rgb24")
            track.enqueue((queued_frame, game_frame))
        else:
            track.enqueue(queued_frame)
        # Retained for replay to a replacement client. A client that takes the
        # slot over while the model is not producing -- held by a dialog pause,
        # or simply between chunks -- would otherwise wait on a track whose
        # queue nothing is going to fill. A fresh frame has not been shown to
        # the current client, so the next one to connect is owed it.
        self._latest_main_frame = queued_frame
        self._latest_auxiliary_frame = game_frame
        self._latest_replayed = False
        # Sent after the frame is queued so a client can never render state that
        # describes a frame it has not been offered yet.
        self.send_client_messages(result.client_messages)

    def close(self) -> None:
        """Close the peer connection and stop the WebRTC server."""
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        track = self._video_track
        if track is not None and self._media_connected.is_set():
            try:
                track.wait_until_drained(_SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
            except BaseException as error:
                failures.append(error)
        loop = self._loop
        if loop is not None:
            future = None
            try:
                future = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(
                        self._shutdown(),
                        timeout=self._startup_timeout_seconds,
                    ),
                    loop,
                )
                future.result(timeout=self._startup_timeout_seconds + 1.0)
            except BaseException as error:
                if future is not None:
                    future.cancel()
                failures.append(error)
        transfer_streams = tuple(self._transfer_streams.values())
        transfers_drained = True
        for stream in transfer_streams:
            try:
                stream.synchronize()
            except BaseException as error:
                transfers_drained = False
                failures.append(error)
        self._transfer_streams.clear()
        if transfers_drained:
            self._materialization_buffer.close()
        else:
            with _QUARANTINED_CUDA_TRANSFERS_LOCK:
                _QUARANTINED_CUDA_TRANSFERS.append(
                    (transfer_streams, self._materialization_buffer)
                )
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except BaseException as error:
                failures.append(error)
        try:
            self._thread.join(timeout=self._startup_timeout_seconds)
        except BaseException as error:
            failures.append(error)
        if self._thread.is_alive():
            failures.append(
                TimeoutError("WebRTC server did not stop before the timeout.")
            )
        self._loop = None
        if failures:
            primary = failures[0]
            for secondary in failures[1:]:
                logger.opt(exception=secondary).warning(
                    "Additional WebRTC cleanup failure"
                )
            raise primary

    def _materialize_video_frame(
        self,
        result: StepResult,
        frame: torch.Tensor,
    ) -> VideoFrame:
        """Return one owned video frame before admitting it to WebRTC."""
        if not frame.is_cuda:
            materialized = _prepare_cpu_video_frame(frame)
            self._materialization_count += 1
            return materialized
        device = resolve_cuda_device(frame.device)
        transfer_stream = self._transfer_stream(device)
        with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
            result.read_output()
        materialized = _materialize_cuda_video_frame(
            frame,
            transfer_stream=transfer_stream,
            buffer=self._materialization_buffer,
        )
        self._materialization_count += 1
        return materialized

    def _transfer_stream(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream:
        """Return the sink-owned high-priority CUDA transfer stream."""
        device = resolve_cuda_device(device)
        assert device.index is not None
        stream = self._transfer_streams.get(device.index)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(
                    device=device,
                    priority=_TRANSFER_STREAM_PRIORITY,
                )
            self._transfer_streams[device.index] = stream
        return stream

    def _run_server(self) -> None:
        """Own the WebRTC asyncio loop for the lifetime of the server."""
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_handle_webrtc_loop_exception)
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_server())
        except BaseException as error:
            self._startup_error = error
            self._started.set()
            loop.close()
            return
        self._started.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _start_server(self) -> None:
        """Create and bind the standalone aiohttp application."""
        self._offer_lock = asyncio.Lock()
        app = web.Application()
        app.router.add_get("/", self._serve_browser)
        app.router.add_get("/app.js", self._serve_browser_script)
        app.router.add_get("/games/{slug}/hud/{filename}", self._serve_game_overlay)
        app.router.add_get("/vendor/{filename}", self._serve_game_vendor)
        app.router.add_get("/assets/{tail:.*}", self._serve_game_tree)
        app.router.add_get("/host/{filename}", self._serve_game_host)
        app.router.add_get("/game/{slug}/{tail:.*}", self._serve_registered_game)
        app.router.add_get("/healthz", self._health)
        app.router.add_post("/api/webrtc/offer", self._offer)
        app.router.add_get("/extensions.json", serve_extension_manifest)
        app.router.add_route("*", "/ext/{name}/{tail:.*}", dispatch_extension)
        runner = web.AppRunner(app)
        await runner.setup()
        address_family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        server_socket = socket.socket(address_family, socket.SOCK_STREAM)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._host, self._port))
            server_socket.setblocking(False)
            server_socket.listen(128)
            self._port = int(server_socket.getsockname()[1])
            site = web.SockSite(runner, server_socket)
            await site.start()
        except Exception:
            server_socket.close()
            await runner.cleanup()
            raise
        self._runner = runner

    async def _serve_browser(self, _: web.Request) -> web.Response:
        """Return the browser client with its ICE servers injected."""
        page = _BROWSER_PAGE.replace("</head>", "<script>window.__PAIRED_VIDEO__ = %s;</script></head>" % json.dumps(os.environ.get("FLASHDREAMS_PAIRED_VIDEO") == "1"))
        ice_script = (
            "<script>window.__ICE_SERVERS__ = %s;</script>" % json.dumps(_browser_ice_servers())
        )
        page = page.replace("</head>", ice_script + "</head>", 1)
        # The ICE servers are injected inline, so a cached copy pins whatever
        # relay was configured when it was stored — a stale one breaks every
        # later connection with an ICE gathering timeout.
        return web.Response(
            text=page,
            content_type="text/html",
            headers={"cache-control": "no-store"},
        )

    async def _serve_browser_script(self, _: web.Request) -> web.Response:
        """Return the browser client's JavaScript."""
        return web.Response(
            text=_BROWSER_SCRIPT,
            content_type="text/javascript",
            headers={"cache-control": "no-store"},
        )

    async def _serve_game_overlay(self, request: web.Request) -> web.Response:
        """Return one per-game overlay module from the games checkout.

        Only ``overlay.js`` and ``overlay.css`` under ``<slug>/hud/`` are served.
        The slug is a registered-looking identifier, and the resolved path must
        stay inside ``EDEN_GAMES_ROOT``.
        """
        slug = request.match_info["slug"]
        filename = request.match_info["filename"]
        content_type = _GAME_OVERLAY_FILES.get(filename)
        if content_type is None or _GAME_OVERLAY_SLUG.fullmatch(slug) is None:
            raise web.HTTPNotFound()
        root = _games_root()
        path = (root / slug / "hud" / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound()
        return web.Response(body=path.read_bytes(), content_type=content_type)

    def _game_file_response(self, path: Path, *, root: Path) -> web.Response:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise web.HTTPNotFound()
        content_type = (
            "text/javascript"
            if resolved.suffix in {".js", ".mjs"}
            else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
        )
        if resolved.suffix == ".html":
            content_type = "text/html"
        if resolved.suffix == ".css":
            content_type = "text/css"
        return web.Response(body=resolved.read_bytes(), content_type=content_type)

    async def _serve_game_vendor(self, request: web.Request) -> web.Response:
        """Serve the shared Three.js build the games import as ``/vendor/``."""
        filename = request.match_info["filename"]
        root = _games_root()
        if filename == "three.module.js":
            path = Path(os.environ.get("EDEN_THREE_MODULE", root / "node_modules/three/build/three.module.js"))
        elif filename in ("three-demo.module.js", "three.core.js"):
            name = "three.module.js" if filename == "three-demo.module.js" else "three.core.js"
            path = root / "node_modules/three-demo/build" / name
        else:
            raise web.HTTPNotFound()
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.Response(body=path.read_bytes(), content_type="text/javascript")

    async def _serve_game_host(self, request: web.Request) -> web.Response:
        """Serve ``games/host/*.js`` for live browser preview."""
        filename = request.match_info["filename"]
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.js", filename):
            raise web.HTTPNotFound()
        root = _games_root()
        return self._game_file_response(root / "host" / filename, root=root)

    async def _serve_game_tree(self, request: web.Request) -> web.Response:
        """Serve ``games/assets/...`` for live browser preview."""
        rel = request.match_info["tail"]
        root = _games_root()
        return self._game_file_response(root / "assets" / rel, root=root)

    async def _serve_registered_game(self, request: web.Request) -> web.Response:
        """Serve ``games/<slug>/...`` under ``/game/<slug>/`` for live preview."""
        slug = request.match_info["slug"]
        if _GAME_OVERLAY_SLUG.fullmatch(slug) is None:
            raise web.HTTPNotFound()
        root = _games_root()
        registry = root / "registry.json"
        if not registry.is_file():
            raise web.HTTPNotFound()
        slugs = {
            entry.get("slug")
            for entry in json.loads(registry.read_text()).get("games", [])
        }
        if slug not in slugs and slug not in {"host", "runtime"}:
            raise web.HTTPNotFound()
        return self._game_file_response(root / slug / request.match_info["tail"], root=root)

    async def _health(self, _: web.Request) -> web.Response:
        """Report whether the server has an open session and client."""
        return web.json_response(
            {
                "open": self._session_desc is not None,
                "client_connected": self.is_client_connected(),
                "hide_cursor": self._hide_cursor,
                "lock_cursor_to_window": self._lock_cursor_to_window,
            }
        )

    async def _offer(self, request: web.Request) -> web.Response:
        """Negotiate one browser peer connection."""
        if self._closed:
            raise web.HTTPServiceUnavailable(reason="WebRTC server is closed.")
        if self._session_desc is None:
            raise web.HTTPConflict(reason="WebRTC server is not open.")

        try:
            payload = await request.json()
        except (json.JSONDecodeError, web.HTTPException) as error:
            raise web.HTTPBadRequest(reason="Expected a JSON WebRTC offer.") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="WebRTC offer must be an object.")
        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            raise web.HTTPBadRequest(
                reason="WebRTC offer requires string sdp and type."
            )

        offer_lock = self._offer_lock
        if offer_lock is None:
            raise web.HTTPServiceUnavailable(reason="WebRTC server is not ready.")
        async with offer_lock:
            if self._closed:
                raise web.HTTPServiceUnavailable(reason="WebRTC server is closed.")
            session_desc = self._session_desc
            if session_desc is None:
                raise web.HTTPConflict(reason="WebRTC server is not open.")

            peer_connection = RTCPeerConnection(
                configuration=RTCConfiguration(iceServers=self._ice_servers)
            )
            # Who is negotiating, and over which candidate types. A client that
            # cannot reach the pod over UDP is invisible otherwise: its offer
            # arrives over TCP and then nothing happens.
            peer_name = request.remote or "unknown"
            logger.info(
                "WebRTC offer from {} ({} candidate lines, turn={})",
                peer_name,
                sdp.count("a=candidate"),
                bool(self._ice_servers),
            )
            video_track = (_PairedVideoTrack if payload.get("paired_video") is True else _VideoTrack)(session_desc.frames_per_second_for_ui)
            peer_connection.addTrack(video_track)
            if isinstance(video_track, _PairedVideoTrack):
                peer_connection.addTrack(video_track.auxiliary)
            has_reliable_control = False
            control_channel: Any | None = None

            @peer_connection.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                nonlocal control_channel, has_reliable_control
                is_reliable_control = channel.label == "controls"
                if is_reliable_control:
                    has_reliable_control = True
                    control_channel = channel
                    if self._peer_connection is peer_connection:
                        self._client_connected = True
                        self._control_channel = channel
                        self._mark_control_activity()

                        self._sent_cursor_options = None

                    @channel.on("open")
                    def on_open() -> None:
                        if self._peer_connection is peer_connection:
                            self._send_cursor_options()
                            # Media may already be connected by the time the
                            # control channel opens, in which case the card
                            # state that describes the replayed frame still
                            # needs sending.
                            self._replay_latest_to_client()
                            self._resend_latest_client_messages()

                    if (
                        self._peer_connection is peer_connection
                        and channel.readyState == "open"
                    ):
                        self._send_cursor_options()
                        self._replay_latest_to_client()
                        self._resend_latest_client_messages()

                @channel.on("message")
                def on_message(message: Any) -> None:
                    if self._peer_connection is not peer_connection:
                        return
                    self._mark_control_activity()
                    try:
                        self._buffer_browser_message(message)
                    except ValueError as error:
                        channel.send(
                            json.dumps({"type": "error", "message": str(error)})
                        )

                @channel.on("close")
                async def on_close() -> None:
                    if is_reliable_control:
                        await self._release_peer_connection(peer_connection)

            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if self._peer_connection is not peer_connection:
                    return
                # A client whose media never opens leaves no other trace, so the
                # transition is logged: it is the only server-side evidence of
                # whether a browser's candidates ever reached us.
                logger.info(
                    "WebRTC peer {} -> {}", peer_name, peer_connection.connectionState
                )
                if peer_connection.connectionState == "connected":
                    self._media_connected.set()
                    # The media path is open now, which is the earliest a frame
                    # written to the track can actually reach the client. A
                    # client that took the slot over while nothing was being
                    # produced gets the last frame here, before it has to wait
                    # for the next one.
                    self._replay_latest_to_client()
                    self._resend_latest_client_messages()
                elif peer_connection.connectionState == "disconnected":
                    self._media_connected.clear()
                elif peer_connection.connectionState in {"failed", "closed"}:
                    await self._release_peer_connection(peer_connection)

            try:
                await peer_connection.setRemoteDescription(
                    RTCSessionDescription(sdp=sdp, type=offer_type)
                )
                await peer_connection.setLocalDescription(
                    await peer_connection.createAnswer()
                )
            except ValueError as error:
                await self._release_peer_connection(
                    peer_connection,
                    detached_video_track=video_track,
                )
                raise web.HTTPBadRequest(reason="Invalid WebRTC offer.") from error
            except BaseException:
                await self._release_peer_connection(
                    peer_connection,
                    detached_video_track=video_track,
                )
                raise

            local_description = peer_connection.localDescription
            if local_description is None:
                await self._release_peer_connection(
                    peer_connection,
                    detached_video_track=video_track,
                )
                raise web.HTTPInternalServerError(
                    reason="WebRTC peer did not create an answer."
                )

            existing_peer = self._peer_connection
            if existing_peer is not None:
                # This server has one client slot. A valid new offer is a page
                # refresh or replacement browser and takes ownership of it.
                try:
                    await self._release_peer_connection(existing_peer)
                except BaseException:
                    await self._release_peer_connection(
                        peer_connection,
                        detached_video_track=video_track,
                    )
                    raise

            self._media_connected.clear()
            self._final_video_track_metrics = None
            self._peer_connection = peer_connection
            self._video_track = video_track
            self._client_connected = has_reliable_control
            if has_reliable_control:
                self._mark_control_activity()
            if peer_connection.connectionState == "connected":
                self._media_connected.set()
            self._control_channel = control_channel
            self._sent_cursor_options = None
            if control_channel is not None and control_channel.readyState == "open":
                self._send_cursor_options()
            query_string = request.rel_url.raw_query_string
            if query_string:
                timestamp_us = self._timestamp_us()
                if timestamp_us is not None:
                    self._append_event(
                        QueryStringUserInputEvent(
                            timestamp=timestamp_us,
                            query_string=query_string,
                        )
                    )
            return web.json_response(
                {"sdp": local_description.sdp, "type": local_description.type}
            )

    def _buffer_browser_message(
        self,
        raw_message: object,
    ) -> None:
        """Validate and append one data-channel message."""
        if not isinstance(raw_message, str):
            raise ValueError("Browser event must be a JSON string.")
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError as error:
            raise ValueError("Browser event must contain valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Browser event must be a JSON object.")

        event_type = payload.get("type")
        if event_type == "ping":
            return
        timestamp_us = self._timestamp_us()
        if timestamp_us is None:
            return

        if event_type == "keyboard":
            key = payload.get("key")
            pressed = payload.get("pressed")
            if not isinstance(key, str) or not key:
                raise ValueError("Keyboard event requires a non-empty key.")
            if not isinstance(pressed, bool):
                raise ValueError("Keyboard event requires a boolean pressed value.")
            event = KeyboardUserInputEvent(
                timestamp=timestamp_us,
                key=key,
                state=(
                    KeyboardInputState.PRESSED
                    if pressed
                    else KeyboardInputState.RELEASED
                ),
            )
        elif event_type == "mouse":
            action = payload.get("action")
            if action not in {"move", "button", "wheel"}:
                raise ValueError(
                    "Mouse event action must be 'move', 'button', or 'wheel'."
                )
            x = _finite_number(payload.get("x"), label="Mouse x")
            y = _finite_number(payload.get("y"), label="Mouse y")
            button = payload.get("button", 0)
            pressed = payload.get("pressed", False)
            wheel_x = _finite_number(payload.get("wheel_x", 0.0), label="wheel_x")
            wheel_y = _finite_number(payload.get("wheel_y", 0.0), label="wheel_y")
            if isinstance(button, bool) or not isinstance(button, int) or button < 0:
                raise ValueError("Mouse button must be a non-negative integer.")
            if not isinstance(pressed, bool):
                raise ValueError("Mouse pressed must be a boolean.")
            event = MouseUserInputEvent(
                timestamp=timestamp_us,
                action=action,
                x=x,
                y=y,
                button=button,
                pressed=pressed,
                wheel_x=wheel_x,
                wheel_y=wheel_y,
            )
        elif event_type == "focus":
            focused = payload.get("focused")
            if not isinstance(focused, bool):
                raise ValueError("Focus event requires a boolean focused value.")
            event = FocusUserInputEvent(
                timestamp=timestamp_us,
                focused=focused,
            )
        elif event_type == "touch":
            action = payload.get("action")
            if action not in {"start", "move", "end", "cancel"}:
                raise ValueError(
                    "Touch event action must be 'start', 'move', 'end', or 'cancel'."
                )
            primary = payload.get("primary", False)
            if not isinstance(primary, bool):
                raise ValueError("Touch primary must be a boolean.")
            event = TouchUserInputEvent(
                timestamp=timestamp_us,
                action=action,
                touch_id=_nonnegative_int(payload.get("touch_id"), label="touch_id"),
                x=_normalized_coordinate(payload.get("x"), label="Touch x"),
                y=_normalized_coordinate(payload.get("y"), label="Touch y"),
                pressure=_unit_number(
                    payload.get("pressure", 0.0), label="Touch pressure"
                ),
                primary=primary,
            )
        elif event_type == "gamepad":
            buttons = _number_tuple(payload.get("buttons", ()), label="buttons")
            pressed = _bool_tuple(payload.get("pressed", ()), label="pressed")
            if len(buttons) != len(pressed):
                raise ValueError("Gamepad buttons and pressed must have equal length.")
            event = GamepadUserInputEvent(
                timestamp=timestamp_us,
                action=_controller_action(payload),
                index=_nonnegative_int(payload.get("index", 0), label="index"),
                controller_id=_string(
                    payload.get("controller_id", payload.get("id", "")),
                    label="controller_id",
                ),
                mapping=_string(payload.get("mapping", ""), label="mapping"),
                axes=_number_tuple(payload.get("axes", ()), label="axes"),
                buttons=buttons,
                pressed=pressed,
            )
        elif event_type == "game_wheel":
            event = GameWheelUserInputEvent(
                timestamp=timestamp_us,
                action=_controller_action(payload),
                index=_nonnegative_int(payload.get("index", 0), label="index"),
                controller_id=_string(
                    payload.get("controller_id", payload.get("id", "")),
                    label="controller_id",
                ),
                steering=_bounded_number(
                    payload.get("steering", 0.0),
                    label="steering",
                    low=-1.0,
                    high=1.0,
                ),
                throttle=_unit_number(payload.get("throttle", 0.0), label="throttle"),
                brake=_unit_number(payload.get("brake", 0.0), label="brake"),
                clutch=_unit_number(payload.get("clutch", 0.0), label="clutch"),
                buttons=_bool_tuple(payload.get("buttons", ()), label="buttons"),
            )
        elif event_type == "xr_controller":
            handedness = payload.get("handedness", "none")
            if handedness not in {"left", "right", "none"}:
                raise ValueError("XR handedness must be 'left', 'right', or 'none'.")
            buttons = _number_tuple(payload.get("buttons", ()), label="buttons")
            pressed = _bool_tuple(payload.get("pressed", ()), label="pressed")
            if len(buttons) != len(pressed):
                raise ValueError("XR buttons and pressed must have equal length.")
            event = XRControllerUserInputEvent(
                timestamp=timestamp_us,
                action=_controller_action(payload),
                handedness=handedness,
                controller_id=_string(
                    payload.get("controller_id", payload.get("id", "")),
                    label="controller_id",
                ),
                axes=_number_tuple(payload.get("axes", ()), label="axes"),
                buttons=buttons,
                pressed=pressed,
                position=cast(
                    tuple[float, float, float] | None,
                    _fixed_number_tuple(
                        payload.get("position"), label="position", length=3
                    ),
                ),
                orientation=cast(
                    tuple[float, float, float, float] | None,
                    _fixed_number_tuple(
                        payload.get("orientation"), label="orientation", length=4
                    ),
                ),
            )
        elif event_type == "client_action":
            action = payload.get("action")
            if not isinstance(action, str) or not action:
                raise ValueError("Client action requires a non-empty action name.")
            data = payload.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Client action data must be a JSON object.")
            event = ClientActionUserInputEvent(
                timestamp=timestamp_us,
                action=action,
                channel=_string(payload.get("channel", ""), label="channel"),
                data=dict(data),
            )
        elif event_type == "reset":
            event = ResetUserInputEvent(timestamp=timestamp_us)
        elif event_type == "close":
            event = CloseUserInputEvent(timestamp=timestamp_us)
        else:
            raise ValueError("Unsupported browser event type.")
        self._append_event(event)

    def _append_event(self, event: UserInputEvent) -> None:
        """Buffer one validated browser event."""
        callback = self._input_callback
        if callback is None:
            raise RuntimeError("WebRTC input callback is not registered.")
        callback(event)

    async def _release_peer_connection(
        self,
        peer_connection: RTCPeerConnection,
        *,
        detached_video_track: _VideoTrack | None = None,
    ) -> None:
        """Release one disconnected peer while leaving the server open."""
        is_active = self._peer_connection is peer_connection
        if not is_active and detached_video_track is None:
            return
        if is_active:
            self._peer_connection = None
            self._media_connected.clear()
            self._client_connected = False
            self._last_control_at = None
            track = self._video_track
            self._control_channel = None
            self._video_track = None
            self._sent_cursor_options = None
        else:
            track = detached_video_track
        failures: list[BaseException] = []
        if track is not None:
            try:
                await track.close()
            except BaseException as error:
                failures.append(error)
            if is_active:
                try:
                    self._final_video_track_metrics = track.metrics_snapshot()
                except BaseException as error:
                    failures.append(error)
        if peer_connection.connectionState != "closed":
            try:
                await peer_connection.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            primary = failures[0]
            for secondary in failures[1:]:
                logger.opt(exception=secondary).warning(
                    "Additional WebRTC peer cleanup failure"
                )
            raise primary

    async def _start_session_if_active(self, track: _VideoTrack) -> None:
        """Reset ``track`` only if it still belongs to the active peer."""
        if self._video_track is track:
            await track.start_session()

    def _timestamp_us(self) -> np.uint64 | None:
        """Return the current server-relative event timestamp."""
        event_origin_ns = self._event_origin_ns
        if event_origin_ns is None:
            return None
        return np.uint64((time.monotonic_ns() - event_origin_ns) // 1_000)

    async def _shutdown(self) -> None:
        """Release async server resources on their owning loop."""
        failures: list[BaseException] = []
        peer_connection = self._peer_connection
        if peer_connection is not None:
            try:
                await self._release_peer_connection(peer_connection)
            except BaseException as error:
                failures.append(error)
        self._media_connected.clear()
        self._client_connected = False
        self._last_control_at = None
        track = self._video_track
        self._video_track = None
        if track is not None:
            try:
                await track.close()
            except BaseException as error:
                failures.append(error)
            try:
                self._final_video_track_metrics = track.metrics_snapshot()
            except BaseException as error:
                failures.append(error)
        runner = self._runner
        self._runner = None
        if runner is not None:
            try:
                await runner.cleanup()
            except BaseException as error:
                failures.append(error)
        if failures:
            primary = failures[0]
            for secondary in failures[1:]:
                logger.opt(exception=secondary).warning(
                    "Additional WebRTC cleanup failure"
                )
            raise primary


def _games_root() -> Path:
    """Return the checkout directory that holds per-game overlay modules."""
    configured = os.environ.get("EDEN_GAMES_ROOT")
    if configured:
        return Path(configured).resolve()
    return (Path(__file__).resolve().parents[4] / "games").resolve()


def _same_stream_format(first: SessionDesc, second: SessionDesc) -> bool:
    """Return whether two sessions can share one negotiated video track."""
    return (
        first.output_layout == second.output_layout
        and first.frames_per_second_for_ui == second.frames_per_second_for_ui
        and first.frames_per_second_for_step == second.frames_per_second_for_step
        and first.video_width == second.video_width
        and first.video_height == second.video_height
    )


def _finite_number(value: object, *, label: str) -> float:
    """Return a finite browser-input number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _normalized_coordinate(value: object, *, label: str) -> float:
    """Return a normalized browser pointer coordinate."""
    result = _finite_number(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _bounded_number(value: object, *, label: str, low: float, high: float) -> float:
    """Return a finite browser-input number within an inclusive range."""
    result = _finite_number(value, label=label)
    if result < low or result > high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return result


def _unit_number(value: object, *, label: str) -> float:
    """Return a browser-input number in ``[0, 1]``."""
    return _bounded_number(value, label=label, low=0.0, high=1.0)


def _nonnegative_int(value: object, *, label: str) -> int:
    """Return a non-negative browser-input integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _string(value: object, *, label: str) -> str:
    """Return a browser-input string."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _number_tuple(value: object, *, label: str) -> tuple[float, ...]:
    """Return a tuple of finite browser-input numbers."""
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be an array.")
    return tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _bool_tuple(value: object, *, label: str) -> tuple[bool, ...]:
    """Return a tuple of browser-input booleans."""
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{label} must be a boolean array.")
    return cast(tuple[bool, ...], tuple(value))


def _fixed_number_tuple(
    value: object, *, label: str, length: int
) -> tuple[float, ...] | None:
    """Return an optional fixed-length tuple of finite numbers."""
    if value is None:
        return None
    result = _number_tuple(value, label=label)
    if len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} values.")
    return result


def _controller_action(
    payload: dict[str, object],
) -> Literal["connected", "disconnected", "state"]:
    """Return a validated controller lifecycle action."""
    action = payload.get("action", "state")
    if action not in {"connected", "disconnected", "state"}:
        raise ValueError(
            "Controller action must be 'connected', 'disconnected', or 'state'."
        )
    return cast(Literal["connected", "disconnected", "state"], action)


def _validated_result_frames(
    result: StepResult, session_desc: SessionDesc
) -> torch.Tensor:
    """Return validated time-major frames without materializing them on the host."""
    # This path may inspect metadata and create views only. The transfer-stream
    # read orders the result before any CUDA operation consumes those views.
    output = result.read_output(sync_with_event=False).detach()
    if result.output_layout == VideoTensorLayout.tchw:
        frames = output
    elif result.output_layout == VideoTensorLayout.btchw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("btchw WebRTC output requires a batch size of one.")
        frames = output[0]
    elif result.output_layout == VideoTensorLayout.bcthw:
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("bcthw WebRTC output requires a batch size of one.")
        frames = output[0].permute(1, 0, 2, 3)
    elif result.output_layout == VideoTensorLayout.bvtchw:
        if output.ndim != 6 or output.shape[:2] != (1, 1):
            raise ValueError(
                "bvtchw WebRTC output requires one batch and one video view."
            )
        frames = output[0, 0]
    else:
        raise ValueError(f"Unsupported WebRTC output layout: {result.output_layout}.")

    if frames.ndim != 4:
        raise ValueError("WebRTC output must resolve to a tchw tensor.")
    if frames.shape[0] != result.frame_count:
        raise ValueError("StepResult.frame_count does not match its output tensor.")
    if frames.shape[1] not in (1, 3):
        raise ValueError("WebRTC output must have one or three color channels.")
    if frames.shape[2:] != (session_desc.video_height, session_desc.video_width):
        raise ValueError("WebRTC output dimensions do not match SessionDesc.")
    if result.output_layout != session_desc.output_layout:
        raise ValueError("StepResult.output_layout does not match SessionDesc.")

    return frames


def _rgb_uint8_thwc(frames: torch.Tensor) -> torch.Tensor:
    """Convert validated frames to contiguous ``[T, H, W, C]`` uint8 storage."""

    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    if frames.is_floating_point():
        frames = ((frames.to(torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous()


def _clone_video_frame(frame: VideoFrame) -> VideoFrame:
    """Return an independent copy of one owned video frame.

    The retained frame is the one a previous client's encoder already consumed,
    so it cannot be handed to another encoder. The copy carries no timestamp;
    the track that admits it assigns one.
    """
    clone = VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
    clone.pts = None
    return clone


def _prepare_cpu_video_frame(frame: torch.Tensor) -> VideoFrame:
    """Materialize one CPU tensor as independently owned RGB pixels."""
    rgb_frame = _rgb_uint8_thwc(frame.unsqueeze(0))[0]
    return VideoFrame.from_ndarray(np.asarray(rgb_frame.numpy()), format="rgb24")


def _materialize_cuda_video_frame(
    source: torch.Tensor,
    *,
    transfer_stream: torch.cuda.Stream,
    buffer: _PinnedRGBFrameBuffer,
) -> VideoFrame:
    """Synchronously convert and copy one CUDA frame into an owned AV frame."""
    if not source.is_cuda:
        raise ValueError("CUDA RGB materialization requires a CUDA tensor.")
    device = resolve_cuda_device(source.device)
    if resolve_cuda_device(transfer_stream.device) != device:
        raise ValueError("CUDA RGB source and transfer stream must match.")
    _, height, width = source.shape
    host_frame = buffer.get((height, width, 3))
    copy_enqueued = False
    try:
        with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
            rgb = _rgb_uint8_thwc(source.unsqueeze(0))
            host_frame.copy_(rgb[0], non_blocking=True)
            copy_enqueued = True
            ready_event = torch.cuda.Event()
            ready_event.record(transfer_stream)
        ready_event.synchronize()
        return VideoFrame.from_ndarray(
            np.asarray(host_frame.numpy()),
            format="rgb24",
        )
    except BaseException:
        if copy_enqueued:
            try:
                transfer_stream.synchronize()
            except BaseException:
                logger.exception(
                    "Failed to drain a WebRTC transfer after materialization "
                    "failed; retiring its pinned frame."
                )
                buffer.retire()
        raise

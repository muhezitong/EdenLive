"""Scene-driven re-anchoring (muhezitong/three-body-game#19).

A game that reports ``scene`` in its state and ships
``<slug>/shots/<scene>/cover.png`` gets its rollout rebuilt from that frame;
these tests pin the pure helpers that decide it.
"""

import json

import pytest

from eden.apps.game2v.game_timeline import GameChunkPacket, GameFramePacket
from eden.apps.game2v.session import _reanchor_rollout, _scene_cover_path, _scene_id

pytestmark = pytest.mark.ci_cpu


def _chunk(*states):
    frames = tuple(
        GameFramePacket(
            game_frame_id=i, logical_time_s=i / 24, simulation_tick=i,
            input_sequence=0, game_slug="demo", ui_state=state,
        )
        for i, state in enumerate(states)
    )
    return GameChunkPacket(
        chunk_id=0, start_time_s=0.0, end_time_s=1.0, input_sequence_start=0,
        input_sequence_end=0, simulation_tick_start=0, simulation_tick_end=len(states),
        frames=frames,
    )


def test_scene_id_reads_the_last_frame():
    assert _scene_id(_chunk({"scene": "open"}, {"scene": "overhead"})) == "overhead"


def test_scene_id_ignores_missing_or_invalid_values():
    assert _scene_id(None) is None
    assert _scene_id(_chunk({"scene": "open"}, {})) is None
    assert _scene_id(_chunk({"scene": ""})) is None
    assert _scene_id(_chunk({"scene": 3})) is None
    assert _scene_id(_chunk(None)) is None


@pytest.fixture
def games(tmp_path, monkeypatch):
    game = tmp_path / "demo"
    (game / "shots" / "overhead").mkdir(parents=True)
    (game / "cover.png").write_bytes(b"png")
    (game / "shots" / "overhead" / "cover.png").write_bytes(b"png")
    (tmp_path / "registry.json").write_text(
        json.dumps({"games": [{"slug": "demo", "cover": "cover.png"}]})
    )
    monkeypatch.setenv("EDEN_GAMES_ROOT", str(tmp_path))
    return game


def test_scene_cover_path_finds_a_shipped_shot(games):
    assert _scene_cover_path("demo", "overhead") == games / "shots" / "overhead" / "cover.png"


def test_scene_cover_path_is_none_without_a_frame_or_with_a_path_escape(games):
    assert _scene_cover_path("demo", "plaque") is None
    assert _scene_cover_path("demo", "../demo") is None
    assert _scene_cover_path("demo", ".hidden") is None



def _chunk_with_scene(scene):
    return _chunk({"scene": scene})


def test_reanchor_fires_once_per_scene_change(monkeypatch, tmp_path):
    game = tmp_path / "demo"
    (game / "shots" / "overhead").mkdir(parents=True)
    (game / "cover.png").write_bytes(b"png")
    (game / "shots" / "overhead" / "cover.png").write_bytes(b"png")
    (tmp_path / "registry.json").write_text(
        json.dumps({"games": [{"slug": "demo", "cover": "cover.png"}]})
    )
    monkeypatch.setenv("EDEN_GAMES_ROOT", str(tmp_path))
    import eden.apps.game2v.session as mod

    class FakeState:
        game_capture = object()
        game_scene = None
        config = None

    st = FakeState()
    st.config = mod.replace(
        None if st.config is None else st.config,
        **{},
    ) if st.config is not None else None
    # 构造一个最小的 config：只需 game_slug 和 conditioning
    class C:
        game_slug = "demo"
        conditioning = type("cond", (), {})()
    st.config = C()
    st.config.conditioning.first_frame_path = None
    st.last_game_chunk = _chunk_with_scene("open")

    calls = []
    monkeypatch.setattr(mod, "_rewind_rollout", lambda s: calls.append(s.game_scene))
    monkeypatch.setattr(mod, "is_distributed_initialized", lambda: False)

    assert _reanchor_rollout(st) is True
    assert calls == ["overhead"]
    assert st.config.conditioning.first_frame_path == game / "shots" / "overhead" / "cover.png"

    st.last_game_chunk = _chunk_with_scene("overhead")
    assert _reanchor_rollout(st) is False
    assert calls == ["overhead"]

    (game / "shots" / "overhead" / "cover.png").unlink()
    st.last_game_chunk = _chunk_with_scene("plaque")
    assert _reanchor_rollout(st) is False
    assert calls == ["overhead"]

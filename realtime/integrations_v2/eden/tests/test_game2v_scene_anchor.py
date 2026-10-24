"""Scene-driven re-anchoring (muhezitong/three-body-game#19).

A game that reports ``scene`` in its state and ships
``<slug>/shots/<scene>/cover.png`` gets its rollout rebuilt from that frame;
these tests pin the pure helpers that decide it.
"""

import json

import pytest

from eden.apps.game2v.game_timeline import GameChunkPacket, GameFramePacket
from eden.apps.game2v.session import _scene_cover_path, _scene_id

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

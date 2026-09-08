import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/filter_testing/filter_bench.py"
spec = importlib.util.spec_from_file_location("filter_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_parameter_bounds_and_step():
    schema = SimpleNamespace(name="strength", min=0.1, max=1.0, step=0.1)
    assert bench.validate_value(schema, 0.3) == 0.3
    for value in (float("nan"), float("inf"), 0, 1.1, 0.35):
        with pytest.raises(ValueError):
            bench.validate_value(schema, value)


def test_metrics_exclude_invalid_pixels_from_temporal_change():
    depths = np.array([[[0.1, 0, 0.3]], [[0.102, 0.2, np.nan]]])
    result = bench.metrics(depths, 0.04, 0.25)
    assert result["valid_fraction"] == 0.5
    assert result["median_frame_change_mm"] == pytest.approx(2)


def test_all_invalid_metrics_are_json_safe():
    result = bench.metrics(np.zeros((2, 3, 3)), 0.04, 0.25)
    assert result["valid_fraction"] == 0
    assert result["p95_frame_change_mm"] is None


def test_disparity_output_is_not_misinterpreted_as_metric_depth():
    frame = SimpleNamespace(as_depth_frame=lambda: SimpleNamespace(get_format=lambda: "DISPARITY"))
    sdk = SimpleNamespace(OBFormat=SimpleNamespace(Y16="Y16"))
    with pytest.raises(RuntimeError, match="not Y16"):
        bench.depth_array(frame, sdk)


def test_crashed_worker_does_not_stop_other_trials(tmp_path, monkeypatch):
    bag = tmp_path / "raw.bag"
    bag.touch()
    trials = tmp_path / "trials.json"
    bench.write_json(trials, [{"name": "edge", "filters": {"EdgeNoiseRemovalFilter": {}}}])
    calls = []

    def crash(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(bench.subprocess, "run", crash)
    out = tmp_path / "results"
    args = SimpleNamespace(bag=str(bag), output=str(out), trials=str(trials), seconds=5,
                           min_depth=0.04, max_depth=0.25, timeout=10)
    assert bench.compare(args) == 1
    assert len(calls) == 2
    assert len(bench.json.loads((out / "summary.json").read_text())) == 2


def test_different_playback_frames_invalidate_comparison(tmp_path, monkeypatch):
    bag = tmp_path / "raw.bag"
    bag.touch()
    trials = tmp_path / "trials.json"
    bench.write_json(trials, [{"name": "edge", "filters": {}}])

    def success(command, **kwargs):
        target = Path(command[command.index("--output") + 1])
        ids = [[1, 1000]] if target.name == "baseline" else [[2, 2000]]
        np.savez(target / "depths.npz", frame_ids=ids)
        bench.write_json(target / "metrics.json", {"frames": 1})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench.subprocess, "run", success)
    out = tmp_path / "results"
    args = SimpleNamespace(bag=str(bag), output=str(out), trials=str(trials), seconds=5,
                           min_depth=0.04, max_depth=0.25, timeout=10)
    assert bench.compare(args) == 1
    result = bench.json.loads((out / "summary.json").read_text())
    assert result[0]["same_input_frames"] is True
    assert result[1]["same_input_frames"] is False
    assert result[1]["status"].startswith("invalid comparison")

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("lingbot_gemini", Path(__file__).resolve().parents[1] /
                                             "scripts/filter_testing/lingbot_gemini.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_intrinsics_are_normalized_without_modifying_original():
    k = np.array([[600, 0, 640], [0, 610, 400], [0, 0, 1]], dtype=float)
    norm = module.normalized_intrinsics(k, (800, 1280))
    np.testing.assert_allclose(norm, [[600 / 1280, 0, 0.5], [0, 610 / 800, 0.5], [0, 0, 1]])
    assert k[0, 0] == 600


def test_gate_preserves_fractional_millimetres_and_rejects_model_infinity():
    depth, valid = module.gate(np.array([[0.10015, np.inf, np.nan, 0, 0.5]]), 0.095, 0.25)
    np.testing.assert_allclose(depth, [[0.10015, 0, 0, 0, 0]])
    assert valid.tolist() == [[True, False, False, False, False]]


def test_invalid_intrinsics_rejected():
    with pytest.raises(ValueError):
        module.normalized_intrinsics(np.zeros((3, 3)), (800, 1280))


def test_rectification_preserves_identity_grid_and_rejects_distorted_depth(monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    models = NS(BROWN_CONRADY=3, BROWN_CONRADY_K6=4)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", NS(OBCameraDistortionModel=models))
    distortion = NS(k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0, model=0)
    profile = NS(get_distortion=lambda: distortion,
                 get_intrinsic=lambda: NS(fx=10, fy=10, cx=2, cy=2))
    rgb = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
    actual, k = module.rectify_rgb(rgb, profile, profile)
    np.testing.assert_array_equal(actual, rgb)
    assert k[0, 0] == 10
    distortion.k1 = 0.1
    with pytest.raises(ValueError, match="zero-distortion"):
        module.rectify_rgb(rgb, profile, profile)

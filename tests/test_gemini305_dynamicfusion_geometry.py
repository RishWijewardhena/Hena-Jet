from __future__ import annotations

import unittest

import numpy as np

from scripts.gemini305_dynamicfusion.geometry import (
    backproject_depth,
    estimate_rigid_transform,
    invert_transform,
    transform_points,
)


class BackprojectionTests(unittest.TestCase):
    def test_backprojects_valid_depth_pixels_in_metres(self) -> None:
        depth = np.array([[0, 1000], [2000, 0]], dtype=np.uint16)

        points, pixels = backproject_depth(
            depth,
            intrinsics=(100.0, 100.0, 0.0, 0.0),
            depth_scale_m=0.001,
            min_depth_m=0.5,
            max_depth_m=2.0,
        )

        np.testing.assert_allclose(points, [[0.01, 0.0, 1.0], [0.0, 0.02, 2.0]])
        np.testing.assert_array_equal(pixels, [[1, 0], [0, 1]])


class RigidTransformTests(unittest.TestCase):
    def test_recovers_source_to_target_transform(self) -> None:
        source = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]
        )
        angle = np.deg2rad(20.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        target = source @ rotation.T + np.array([0.02, -0.03, 0.04])

        transform = estimate_rigid_transform(source, target)

        np.testing.assert_allclose(transform_points(source, transform), target, atol=1e-10)
        np.testing.assert_allclose(
            transform_points(target, invert_transform(transform)), source, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()

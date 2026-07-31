from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
import subprocess

import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_DIR = Path(
    "/media/rishmika/Shared Data/Idea8/Hena Jet/captures/zed_m_serial_scan_30cm"
)

OUTPUT_DIR = INPUT_DIR / "reconstruction"
TRANSFORMED_DIR = OUTPUT_DIR / "01_transformed"
MATRIX_DIR = OUTPUT_DIR / "matrices"

MERGED_CLOUD_PATH = OUTPUT_DIR / "mascot_merged_cleaned.ply"
MESH_PATH = OUTPUT_DIR / "mascot_mesh_poisson.ply"
LOG_PATH = OUTPUT_DIR / "cloudcompare_pipeline.log"

# First/reference scan: angle_005p_00.ply
REFERENCE_ANGLE_DEG = 5.0

# ZED IMAGE coordinates: X right, Y down, Z forward.
# The physical orbit axis is vertical and is therefore parallel to camera Y.
# ORBIT_RADIUS_M = 0.192
ORBIT_RADIUS_M = 0.175


# Pivot location in the reference camera frame. The default assumes that the
# reference camera points directly at the pivot.
# PIVOT_IN_REFERENCE = np.array(
#     [0.025, 0.0, ORBIT_RADIUS_M],
#     dtype=float,
# )

PIVOT_IN_REFERENCE = np.array(
    [0.025, 0.0,ORBIT_RADIUS_M],
    dtype=float,
)
# Reverse this if the known-angle transformations move the scans apart.
ANGLE_SIGN = 1.0

# Optional crop after the known transform and before ICP, expressed in the
# reference camera frame as (Xmin, Ymin, Zmin, Xmax, Ymax, Zmax), in metres.
# Measure these bounds around only the mascot. Keeping the table/background can
# make ICP align those surfaces instead of the mascot. Leave as None to disable.
CROP_BOUNDS_M: tuple[float, float, float, float, float, float] | None = None
# Example only:
# CROP_BOUNDS_M = (-0.10, -0.12, 0.10, 0.10, 0.12, 0.30)
CROP_BOUNDS_M = (
    -0.060, -0.100, 0.095,
     0.110,  0.100, 0.255,
)

# Per-scan cleanup before ICP.
PRE_ICP_SOR_NEIGHBORS = 20
PRE_ICP_SOR_SIGMA = 2.0

# Progressive ICP settings.
# Adjacent 5-degree views should have high overlap. Reduce this only if large
# parts of adjacent scans do not overlap after cropping.
ICP_OVERLAP_PERCENT = 70
ICP_ITERATIONS = 60
ICP_RANDOM_SAMPLING_LIMIT = 35_000

# Final merged-cloud cleanup. All distances are metres.
FINAL_SOR_NEIGHBORS = 30
FINAL_SOR_SIGMA = 1.5
REMOVE_DUPLICATES_DISTANCE_M = 0.00010
SPATIAL_SUBSAMPLE_M = 0.00050
NORMAL_RADIUS_M = 0.0020
NORMAL_MST_NEIGHBORS = 12

# Set to False if you only want the registered point cloud.
CREATE_POISSON_MESH = True
POISSON_DEPTH = 9
POISSON_DENSITY_TRIM_QUANTILE = 0.02
POISSON_SCALE = 1.1

FLATPAK_APP_ID = "org.cloudcompare.CloudCompare"


def run_command(command: list[str]) -> None:
    """Run a command and raise a readable error if it fails."""
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def flatpak_app_is_installed(flatpak: str, app_id: str) -> bool:
    result = subprocess.run(
        [flatpak, "info", app_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def cloudcompare_command(
    *,
    which=shutil.which,
    flatpak_installed=flatpak_app_is_installed,
) -> list[str]:
    """Return the command prefix for native or Flatpak CloudCompare."""
    for executable_name in ("CloudCompare", "cloudcompare"):
        executable = which(executable_name)
        if executable is not None:
            return [executable]

    flatpak = which("flatpak")
    if flatpak is not None and flatpak_installed(flatpak, FLATPAK_APP_ID):
        return [flatpak, "run", FLATPAK_APP_ID]

    raise RuntimeError(
        "CloudCompare was not found as a native executable or as Flatpak "
        f"{FLATPAK_APP_ID}."
    )


def read_angle(filename: str) -> float:
    """
    Read angles from names such as:
        angle_005p_00.ply -> 5.00
        angle_0345p00.ply -> 345.00
    """
    match = re.search(r"angle_(\d+)p_?(\d+)", filename)
    if match is None:
        raise ValueError(f"Cannot extract an angle from: {filename}")

    return float(f"{match.group(1)}.{match.group(2)}")


def find_ply_files(input_dir: Path) -> list[Path]:
    """Find input captures and order them by their recorded angle."""
    files = list(input_dir.glob("angle_*p*.ply"))
    return sorted(files, key=lambda path: read_angle(path.name))


def normalize_angle_degrees(angle_degrees: float) -> float:
    """Represent an angle in the interval [-180, 180)."""
    return (angle_degrees + 180.0) % 360.0 - 180.0


def rotation_about_y(
    angle_degrees: float,
    pivot: np.ndarray,
) -> np.ndarray:
    """Build a 4x4 transform that rotates around the fixed vertical pivot."""
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)

    rotation = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=float,
    )

    # p_reference = R @ p_camera + (pivot - R @ pivot)
    translation = pivot - rotation @ pivot

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def cloudcompare_file_argument(path: Path) -> str:
    """
    CloudCompare parses the value following SAVE_CLOUDS FILE itself.
    Literal quotes preserve output paths containing spaces.
    """
    return f'"{path}"'


def cloudcompare_crop_argument(
    bounds: tuple[float, float, float, float, float, float],
) -> str:
    """Convert XYZ minimum/maximum bounds to CloudCompare's crop syntax."""
    return ":".join(f"{value:.10g}" for value in bounds)


def initial_transform_and_clean(
    cloudcompare: list[str],
    ply_files: list[Path],
) -> list[Path]:
    """Apply the known motor-angle pose and per-scan SOR cleanup."""
    transformed_paths: list[Path] = []

    print("\nStage 1/4: known-angle transforms and per-scan cleanup")

    for ply_path in ply_files:
        captured_angle = read_angle(ply_path.name)
        raw_relative_angle = ANGLE_SIGN * (
            captured_angle - REFERENCE_ANGLE_DEG
        )
        relative_angle = normalize_angle_degrees(raw_relative_angle)

        transform = rotation_about_y(
            relative_angle,
            PIVOT_IN_REFERENCE,
        )

        matrix_path = MATRIX_DIR / f"{ply_path.stem}_matrix.txt"
        output_path = (
            TRANSFORMED_DIR / f"{ply_path.stem}_transformed.ply"
        )

        np.savetxt(matrix_path, transform, fmt="%.10f")

        command = [
            *cloudcompare,
            "-VERBOSITY",
            "2",
            "-SILENT",
            "-AUTO_SAVE",
            "OFF",
            "-O",
            str(ply_path),
            "-APPLY_TRANS",
            str(matrix_path),
        ]

        if CROP_BOUNDS_M is not None:
            command.extend(
                [
                    "-CROP",
                    cloudcompare_crop_argument(CROP_BOUNDS_M),
                ]
            )

        command.extend(
            [
                "-SOR",
                str(PRE_ICP_SOR_NEIGHBORS),
                str(PRE_ICP_SOR_SIGMA),
                "-C_EXPORT_FMT",
                "PLY",
                "-SAVE_CLOUDS",
                "FILE",
                cloudcompare_file_argument(output_path),
            ]
        )

        print(
            f"\n{ply_path.name}: captured={captured_angle:.2f} deg, "
            f"relative={relative_angle:.2f} deg"
        )
        run_command(command)
        transformed_paths.append(output_path)

    return transformed_paths


def reference_first(
    transformed_paths: list[Path],
) -> list[Path]:
    """Order scans forward around the orbit, starting at the reference."""
    reference_candidates = [
        path
        for path in transformed_paths
        if math.isclose(
            read_angle(path.name),
            REFERENCE_ANGLE_DEG,
            abs_tol=1.0e-6,
        )
    ]

    if not reference_candidates:
        raise RuntimeError(
            "The configured reference scan was not found. "
            f"REFERENCE_ANGLE_DEG={REFERENCE_ANGLE_DEG}"
        )

    return sorted(
        transformed_paths,
        key=lambda path: (
            read_angle(path.name) - REFERENCE_ANGLE_DEG
        ) % 360.0,
    )


def progressive_icp_merge(
    cloudcompare: list[str],
    transformed_paths: list[Path],
) -> None:
    """
    ICP-align each incoming scan to the accumulated reconstruction.

    At every step:
      first cloud  = accumulated model/reference
      second cloud = incoming data that ICP is allowed to move
    """
    ordered_paths = reference_first(transformed_paths)

    print("\nStage 2/4: progressive ICP and merge")

    command = [
        *cloudcompare,
        "-VERBOSITY",
        "2",
        "-SILENT",
        "-AUTO_SAVE",
        "OFF",
        "-LOG_FILE",
        str(LOG_PATH),
        "-C_EXPORT_FMT",
        "PLY",
        "-O",
        str(ordered_paths[0]),
    ]

    for incoming_path in ordered_paths[1:]:
        command.extend(
            [
                "-O",
                str(incoming_path),
                "-ICP",
                "-REFERENCE_IS_FIRST",
                "-OVERLAP",
                str(ICP_OVERLAP_PERCENT),
                "-ITER",
                str(ICP_ITERATIONS),
                "-RANDOM_SAMPLING_LIMIT",
                str(ICP_RANDOM_SAMPLING_LIMIT),
                "-FARTHEST_REMOVAL",
                "-MERGE_CLOUDS",
            ]
        )

    # Clean, reduce duplicate density, calculate consistently oriented normals,
    # and save one final registered cloud.
    command.extend(
        [
            "-SOR",
            str(FINAL_SOR_NEIGHBORS),
            str(FINAL_SOR_SIGMA),
            "-RDP",
            str(REMOVE_DUPLICATES_DISTANCE_M),
            "-SS",
            "SPATIAL",
            str(SPATIAL_SUBSAMPLE_M),
            "-OCTREE_NORMALS",
            str(NORMAL_RADIUS_M),
            "-ORIENT",
            "PLUS_BARYCENTER",
            "-ORIENT_NORMS_MST",
            str(NORMAL_MST_NEIGHBORS),
            "-SAVE_CLOUDS",
            "FILE",
            cloudcompare_file_argument(MERGED_CLOUD_PATH),
        ]
    )

    run_command(command)


def create_poisson_mesh() -> None:
    """Create and crop a Poisson mesh using Open3D."""
    print("\nStage 3/4: Poisson surface reconstruction")

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required only for the mesh stage. Install it with:\n"
            "python3 -m pip install open3d\n"
            "Then run this script again, or set CREATE_POISSON_MESH=False "
            "to generate only the registered point cloud."
        ) from error

    cloud = o3d.io.read_point_cloud(str(MERGED_CLOUD_PATH))
    if cloud.is_empty():
        raise RuntimeError(
            f"Open3D could not read points from {MERGED_CLOUD_PATH}"
        )

    # Re-estimate normals here so the meshing stage does not depend on whether
    # the installed PLY importer retains CloudCompare normal properties.
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_M,
            max_nn=50,
        )
    )
    cloud.orient_normals_consistent_tangent_plane(
        NORMAL_MST_NEIGHBORS
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=POISSON_DEPTH,
            scale=POISSON_SCALE,
            linear_fit=False,
        )
    )

    densities_array = np.asarray(densities)
    if (
        densities_array.size > 0
        and POISSON_DENSITY_TRIM_QUANTILE > 0.0
    ):
        threshold = np.quantile(
            densities_array,
            POISSON_DENSITY_TRIM_QUANTILE,
        )
        mesh.remove_vertices_by_mask(densities_array < threshold)

    # Remove Poisson geometry extending beyond the measured cloud bounds.
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(
        str(MESH_PATH),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    ):
        raise RuntimeError(f"Failed to save mesh to {MESH_PATH}")


def print_summary() -> None:
    print("\nStage 4/4: finished")
    print(f"Registered cloud: {MERGED_CLOUD_PATH}")
    print(f"CloudCompare log: {LOG_PATH}")
    if CREATE_POISSON_MESH:
        print(f"Poisson mesh: {MESH_PATH}")


def validate_settings() -> None:
    if ORBIT_RADIUS_M <= 0.0:
        raise ValueError("ORBIT_RADIUS_M must be greater than zero.")
    if PIVOT_IN_REFERENCE.shape != (3,):
        raise ValueError("PIVOT_IN_REFERENCE must contain X, Y, Z.")
    if CROP_BOUNDS_M is not None:
        if len(CROP_BOUNDS_M) != 6:
            raise ValueError("CROP_BOUNDS_M must contain six values.")
        x_min, y_min, z_min, x_max, y_max, z_max = CROP_BOUNDS_M
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError(
                "Every CROP_BOUNDS_M minimum must be below its maximum."
            )
    if not 10 <= ICP_OVERLAP_PERCENT <= 100:
        raise ValueError(
            "ICP_OVERLAP_PERCENT must be between 10 and 100."
        )
    if not 0.0 <= POISSON_DENSITY_TRIM_QUANTILE < 1.0:
        raise ValueError(
            "POISSON_DENSITY_TRIM_QUANTILE must be in [0, 1)."
        )


def main() -> None:
    validate_settings()
    cloudcompare = cloudcompare_command()
    ply_files = find_ply_files(INPUT_DIR)

    if not ply_files:
        raise RuntimeError(f"No matching PLY files found in:\n{INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using CloudCompare command: {' '.join(cloudcompare)}")
    print(f"Found {len(ply_files)} point clouds.")
    print(f"Orbit radius: {ORBIT_RADIUS_M:.4f} m")
    print(f"Pivot in reference frame: {PIVOT_IN_REFERENCE}")

    transformed_paths = initial_transform_and_clean(
        cloudcompare,
        ply_files,
    )
    progressive_icp_merge(cloudcompare, transformed_paths)

    if CREATE_POISSON_MESH:
        create_poisson_mesh()

    print_summary()


if __name__ == "__main__":
    main()
from pathlib import Path
import math
import re
import shutil
import subprocess
import numpy as np

INPUT_DIR = Path(
    "/media/rishmika/Shared Data/Idea8/Hena Jet/captures/zed_m_serial_scan"
)

OUTPUT_DIR = INPUT_DIR / "transformed"
MATRIX_DIR = OUTPUT_DIR / "matrices"

# First/reference scan is angle_005p_00.ply
REFERENCE_ANGLE = 5.0

# ROTATION_AXIS = "Y"# Assumes the mascot rotation centre is 19.2 cm along camera Z.
# The camera follows a circular orbit around a fixed pivot.
# The orbit axis is the Y-axis, and the orbit radius is editable.
ORBIT_RADIUS_M = 0.192

# This assumes the pivot is directly in front of the first camera.
PIVOT_IN_REFERENCE = np.array([
    0.0,
    0.0,
    ORBIT_RADIUS_M,
], dtype=float)

# Change to -1 if the clouds move in the opposite direction.
ANGLE_SIGN = 1


FLATPAK_APP_ID = "org.cloudcompare.CloudCompare"


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
        "CloudCompare was not found as a native executable or an installed "
        f"Flatpak ({FLATPAK_APP_ID})."
    )


def find_ply_files(input_dir: Path) -> list[Path]:
    """Find captures such as angle_0005p00.ply and angle_005p_00.ply."""
    return sorted(input_dir.glob("angle_*p*.ply"))


def build_transform_command(
    cloudcompare: list[str],
    ply_path: Path,
    matrix_path: Path,
    output_path: Path,
) -> list[str]:
    # SAVE_CLOUDS parses FILE as its own space-separated list. Literal quotes
    # are therefore required inside the argv value when a path has spaces.
    quoted_output_path = f'"{output_path}"'
    return [
        *cloudcompare,
        "-SILENT",
        "-AUTO_SAVE", "OFF",
        "-O", str(ply_path),
        "-APPLY_TRANS", str(matrix_path),
        "-C_EXPORT_FMT", "PLY",
        "-SAVE_CLOUDS", "FILE", quoted_output_path,
    ]


def read_angle(filename: str) -> float:
    """
    Reads names such as:
        angle_005p_00.ply -> 5.00
        angle_010p_00.ply -> 10.00

    Also supports names without the underscore after p.
    """
    match = re.search(r"angle_(\d+)p_?(\d+)", filename)

    if match is None:
        raise ValueError(f"Cannot extract angle from: {filename}")

    whole_part = match.group(1)
    decimal_part = match.group(2)

    return float(f"{whole_part}.{decimal_part}")


def rotation_about_y(
    angle_degrees: float,
    center: np.ndarray
) -> np.ndarray:
    angle_radians = math.radians(angle_degrees)

    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)

    rotation = np.array([
        [cos_angle,  0.0, sin_angle],
        [0.0,        1.0, 0.0],
        [-sin_angle, 0.0, cos_angle],
    ])

    # Rotate around the mascot centre rather than coordinate origin.
    translation = center - rotation @ center

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def main() -> None:
    cloudcompare = cloudcompare_command()
    ply_files = find_ply_files(INPUT_DIR)

    if not ply_files:
        raise RuntimeError(f"No matching PLY files found in:\n{INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using CloudCompare command: {' '.join(cloudcompare)}")
    print(f"Found {len(ply_files)} point clouds.")

    for ply_path in ply_files:
        captured_angle = read_angle(ply_path.name)

        relative_angle = ANGLE_SIGN * (
            captured_angle - REFERENCE_ANGLE
        )

        transform = rotation_about_y(
            relative_angle,
            PIVOT_IN_REFERENCE
        )

        matrix_path = (
            MATRIX_DIR / f"{ply_path.stem}_matrix.txt"
        )

        output_path = (
            OUTPUT_DIR / f"{ply_path.stem}_transformed.ply"
        )

        np.savetxt(
            matrix_path,
            transform,
            fmt="%.10f"
        )

        command = build_transform_command(
            cloudcompare,
            ply_path,
            matrix_path,
            output_path,
        )

        print(
            f"{ply_path.name}: "
            f"captured={captured_angle:.2f}°, "
            f"relative={relative_angle:.2f}°"
        )

        subprocess.run(command, check=True)

    print(f"\nFinished. Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

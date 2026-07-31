# ADR-001: Use Plane-Constrained Global Markerless Registration

## Status

Accepted

## Date

2026-07-31

## Context

The scanner must merge 72 mechanically indexed ZED-M captures without placing
ArUco markers in the scene. The earlier fixed-axis, progressive ICP pipeline
showed layered platform surfaces, doubled mascot geometry, cumulative drift,
and long runtimes on an ever-growing accumulated cloud.

Motor angles constrain yaw well but do not measure camera-mount pitch, roll, or
vertical variation. Object-only ICP can refine residual error, but the mascot
is sufficiently smooth and symmetric that point-to-plane ICP can slide around
its surface and report a good RMSE for an incorrect yaw correction.

## Decision

Use the motor angles as strong circular-orbit priors, estimate pitch/roll/height
from the visible platform in every capture, and add only guarded object ICP
edges to an Open3D pose graph. Optimize absolute scan poses globally, project
reliable platform planes back onto the reference platform, then apply those
poses to original-resolution clouds with CloudCompare.

Rejected ICP refinements retain their motor/plane fallback. Output is blocked
unless plane coverage, sequential usability, platform residual, and optimized
loop closure pass explicit quality gates.

## Alternatives Considered

### Progressive accumulated-cloud ICP

- Simple and previously implemented.
- One incorrect merge contaminates every later scan.
- Runtime and correspondence ambiguity grow with the accumulated cloud.
- Rejected because it cannot distribute drift globally or provide a clean loop
  closure constraint.

### Object ICP without platform constraints

- Does not require a visible platform.
- Cannot reliably observe pitch, roll, and height on this smooth mascot.
- Rejected because good fitness/RMSE did not prevent visible platform layers.

### ArUco markers

- Provides a strong external pose reference and previously avoided the
  platform gap.
- Markers cannot currently be placed in the capture setup.
- Deferred rather than technically rejected; it remains preferable when the
  physical setup permits it.

### Relax ICP correction guards

- Would accept more edges.
- Dataset evidence showed ICP undoing 2–4 degrees of known 5-degree motor steps
  for only a very small RMSE improvement.
- Rejected because acceptance rate alone is not evidence of accuracy.

## Consequences

- Open3D is required for all registration runs, not only meshing.
- CloudCompare remains required for full-resolution transforms and cleanup.
- Registration uses more pairwise work but avoids progressive drift.
- The platform remains in final outputs but is excluded from object ICP.
- Diagnostics distinguish accepted ICP from a usable guarded prior fallback.
- A final projection guarantees platform consistency for reliable plane fits.
- Markerless yaw accuracy remains limited by pivot calibration, encoder
  accuracy, and object symmetry.

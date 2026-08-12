# Hena Jet: MANO-Based Human Hand Reconstruction Plan

## 1. Project objective

The Hena Jet system needs an accurate 3D model of a person's hand so that a 2D mehendi design can be projected onto the curved skin surface and converted into a safe robot-nozzle trajectory.

The proposed reconstruction system combines:

- A general articulated hand model called **MANO**.
- Wide-view RGB images that show the complete hand and its finger pose.
- Close-range RGB-D observations that measure the real skin surface.
- Known camera-orbit information from the mechanical scanning system.
- Multi-frame optimization and non-rigid fusion.
- Live hand tracking and safety monitoring during the drawing stage.

The main difficulty is that the close-range RGB-D camera cannot see the entire hand in one frame. The hand may also move slightly between frames. Ordinary rigid point-cloud registration, including ICP, is not sufficient when fingers bend or the hand changes shape.

MANO provides a complete, anatomically structured hand mesh even when only part of the hand is visible. The RGB-D observations then personalize and refine the visible areas of that mesh.

## 2. Central design decision

Use **MANO**, not standard SMPL, as the parametric model.

| Model | Intended purpose | Suitability for Hena Jet |
|---|---|---|
| SMPL | Full human body with simplified hands | Not suitable for detailed hand reconstruction |
| SMPL-X | Full body, face and articulated hands | Useful only if the complete person must be modelled |
| MANO | Detailed articulated left or right hand | Recommended |

MANO separates a hand into:

- **Shape parameters**, \(\beta\): identity-dependent properties such as palm width, hand thickness and finger proportions.
- **Pose parameters**, \(\theta\): joint rotations and finger articulation.
- **Global orientation**, \(R\): rotation of the complete hand.
- **Global translation**, \(T\): position of the complete hand.

For frame \(t\), the hand mesh is represented as:

\[
V_t = \operatorname{MANO}(\beta,\theta_t)
\]

and is placed in the camera coordinate system using:

\[
V_t^{camera} = R_tV_t + T_t
\]

The same person's shape \(\beta\) should normally be shared across all frames, while \(\theta_t\), \(R_t\), and \(T_t\) may change with time.

## 3. What MANO solves and what it does not solve

### MANO can help with

- Providing a complete hand topology when each close frame shows only one region.
- Identifying the relationship between the palm, fingers and wrist.
- Representing finger bending and wrist articulation.
- Mapping observations from different hand poses into a common canonical pose.
- Keeping the reconstructed mesh anatomically plausible.
- Establishing consistent mesh vertices and triangles across all frames.

### MANO cannot provide by itself

- The exact surface of the scanned person's skin.
- Accurate veins, wrinkles, nails and other fine details.
- Correct metric depth from a single RGB image.
- Reliable recovery from arbitrary fast motion, blur or severe occlusion.
- Safe compensation for unrestricted movement while the robot is drawing.

The MANO mesh is a statistical prior. Unseen regions are plausible estimates, not measured surfaces. The actual RGB-D depth must remain the main source of metric surface geometry for the nozzle trajectory.

## 4. Recommended system architecture

The most reliable design uses two complementary camera views.

### 4.1 Wide-view camera

A fixed wide RGB camera should:

- See the complete hand, wrist and all fingers.
- Estimate global hand movement.
- Estimate the complete MANO pose.
- Maintain joint identity when the close camera sees only a small region.
- Track movement during scanning and drawing.

This camera can be a separate RGB camera, a phone used during early experiments, or another calibrated camera installed above the hand.

### 4.2 Moving close-range RGB-D camera

The moving RGB-D camera should:

- Capture accurate local depth and RGB texture.
- Move around the hand using the mechanical orbit.
- Record overlapping close views of the dorsal hand surface.
- Supply the geometry needed for nozzle-height planning.

The selected camera must be tested at the actual working distance. If the target distance is approximately 7 cm, its valid minimum depth range, depth noise, field of view, RGB-depth alignment and exposure behaviour must be verified experimentally.

### 4.3 Mechanical pose source

The scanning mechanism should record:

- Motor step count.
- Commanded angle.
- Calibrated actual angle, if available.
- Orbit radius.
- Camera-to-pivot extrinsic transform.
- Timestamp for every capture.

Mechanical pose is valuable as an initial estimate, but it should not be assumed to be exact. Backlash, flex, step error and calibration error can introduce residual pose errors.

### 4.4 Processing system

The reconstruction software combines:

```text
Wide RGB hand tracking
        +
Close RGB-D measurements
        +
Known camera motion
        +
MANO hand prior
        ↓
Joint multi-frame fitting
        ↓
Canonicalized depth observations
        ↓
Personalized high-resolution hand mesh
```

## 5. Coordinate frames

The following coordinate frames should be defined explicitly:

- **Wide-camera frame**, \(C_w\)
- **Close RGB-D camera frame**, \(C_d(t)\)
- **Robot base frame**, \(B\)
- **Canonical MANO frame**, \(M\)
- **Current hand frame**, \(H(t)\)
- **Scan pivot or fixture frame**, \(P\)

Required transformations include:

- \(T_{C_w \rightarrow B}\): wide camera to robot base.
- \(T_{C_d(t) \rightarrow B}\): close camera to robot base at frame \(t\).
- \(T_{H(t) \rightarrow B}\): current hand pose in the robot base.
- \(T_{M \rightarrow H(t)}\): articulated MANO deformation and global hand transform.

The final nozzle trajectory must be expressed in the robot base frame. Therefore, the complete calibration chain must be validated before drawing.

## 6. How RGB-D data adjusts the MANO mesh

The system begins with a general MANO mesh and repeatedly compares its prediction with the measurements. An optimizer changes the hand parameters until the disagreement becomes small.

### 6.1 RGB information

The RGB image supplies:

- 2D hand landmarks.
- Finger and joint identity.
- Hand silhouette.
- Left-hand or right-hand classification.
- Occlusion and visibility information.
- An initial MANO pose estimate.

The predicted MANO joints are projected into the image using the camera intrinsics:

\[
u=f_x\frac{X}{Z}+c_x
\]

\[
v=f_y\frac{Y}{Z}+c_y
\]

The projected joints are compared with the detected 2D joints:

\[
E_{keypoints}
=
\sum_j c_j
\left\|
\pi(J_j^{MANO})-j_j^{RGB}
\right\|^2
\]

where \(c_j\) is the confidence of landmark \(j\).

### 6.2 Depth information

For a valid hand pixel \((u,v)\) with depth \(Z\), the corresponding 3D point is:

\[
X=(u-c_x)\frac{Z}{f_x}
\]

\[
Y=(v-c_y)\frac{Z}{f_y}
\]

\[
Z=D(u,v)
\]

The masked depth pixels therefore become a measured hand point cloud.

The system compares this cloud with the visible MANO surface. Recommended comparison methods, from simpler to stronger, are:

1. Point-to-vertex distance for early debugging.
2. Point-to-triangle distance for correct surface fitting.
3. Point-to-plane distance after stable surface normals are available.
4. Differentiable rendered-depth comparison for pixel-aligned optimization.

A robust depth loss can be written as:

\[
E_{depth}
=
\sum_{p\in P_t}
\rho\left(
d(p,S(V_t))
\right)
\]

where:

- \(P_t\) is the valid observed depth cloud.
- \(S(V_t)\) is the current MANO mesh surface.
- \(d\) is point-to-surface distance.
- \(\rho\) is a robust loss that reduces the influence of depth outliers.

### 6.3 Silhouette information

Render the current MANO mesh into a binary mask and compare it with the measured hand mask:

\[
E_{silhouette}
=
1-\operatorname{IoU}
\left(
M_{rendered},M_{observed}
\right)
\]

This prevents the mesh from matching depth locally while having an incorrect outline.

### 6.4 Priors and regularization

Observation losses alone can produce anatomically impossible or unstable solutions. Add:

- A pose prior to discourage impossible joint angles.
- A shape prior to prevent unrealistic hand proportions.
- A temporal loss to prevent sudden changes between frames.
- A collision loss to prevent fingers from passing through one another.
- A surface-smoothness loss for any additional displacement field.

An example total loss is:

\[
E =
w_dE_{depth}
+w_kE_{keypoints}
+w_sE_{silhouette}
+w_pE_{pose}
+w_\beta E_{shape}
+w_tE_{temporal}
+w_cE_{collision}
\]

The weights should be tuned using measured validation data. Depth should receive high importance because accurate surface height is required for the nozzle.

## 7. Multi-frame fitting for partial close views

The entire hand does not need to appear in every close-range frame. Instead, all frames are fitted jointly:

\[
\underset{
\beta,\{\theta_t,R_t,T_t\}
}{
\operatorname{minimize}
}
\sum_t E_t
\]

The parameters are organized as follows:

| Parameter | Shared or frame-specific? | Purpose |
|---|---|---|
| \(\beta\) | Shared | Identity-dependent hand shape |
| \(\theta_t\) | Frame-specific | Finger and wrist articulation |
| \(R_t\) | Frame-specific | Overall hand orientation |
| \(T_t\) | Frame-specific | Overall hand position |
| Camera pose | Frame-specific | Close-camera location at capture time |
| Surface offsets | Shared in canonical space | Person-specific surface refinement |

An example sequence may observe:

```text
Frame 1: palm and thumb
Frame 2: palm and index finger
Frame 3: middle and ring fingers
Frame 4: wrist and little finger
                ↓
One shared personalized hand model
```

Each frame constrains only the visible, valid surface. MANO supplies a consistent structure that connects those partial measurements.

Neighbouring close views should have at least approximately 30–50% overlap. The wrist, palm landmarks and multiple finger joints should reappear regularly to avoid ambiguous correspondences.

## 8. Canonicalization and non-rigid fusion

Rigid ICP assumes that all points belong to one rigid object. A moving hand violates this assumption when fingers bend or soft tissue deforms.

Instead, each observation should be mapped into a common canonical MANO pose.

For each valid measured point:

1. Determine its corresponding triangle or surface coordinate on the fitted MANO mesh at time \(t\).
2. Use the MANO skinning transformation to undo the frame's finger and wrist pose.
3. Transform the point into the canonical hand coordinate system.
4. Fuse compatible observations from all frames.
5. Reject observations with poor confidence, large residuals or inconsistent normals.

Conceptually:

```text
Measured point in frame t
        ↓
Correspondence on posed MANO mesh
        ↓
Inverse skinning / inverse deformation
        ↓
Canonical MANO surface
        ↓
Multi-frame fusion
```

This process allows depth captured under slightly different poses to contribute to one consistent personalized model.

## 9. Personalized surface refinement

Optimizing only the standard MANO shape parameters will still produce a smooth statistical hand. To reproduce the actual skin surface, store additional offsets in canonical space.

For a measured canonical point \(p_i\) corresponding to a MANO surface location \(v_i\):

\[
\Delta v_i = p_i-v_i^{MANO}
\]

The personalized surface is:

\[
V_{personalized}=V_{MANO}+\Delta V
\]

The offsets should be:

- Fused from multiple observations.
- Weighted by depth confidence and viewing angle.
- Smoothed locally without removing real curvature.
- Limited so noise cannot create unsafe spikes.
- Marked as measured or model-inferred.

The final mesh should carry a per-vertex confidence value:

- **High confidence**: observed from several good RGB-D views.
- **Medium confidence**: observed from one or two acceptable views.
- **Low confidence**: inferred mainly from MANO.

The robot should draw only on regions with sufficient measured-surface confidence.

## 10. Development strategy

The complete system should not be implemented in one step. Each stage must pass a clear validation checkpoint before movement and robot drawing are introduced.

## Stage 0: Hardware and calibration verification

### Tasks

1. Select and test the close-range RGB-D camera at the real operating distance.
2. Confirm aligned RGB and depth resolutions.
3. Measure the depth scale.
4. Record camera intrinsics.
5. Calibrate RGB-to-depth extrinsics if the SDK does not provide registered depth.
6. Calibrate the close camera relative to the scanner mechanism.
7. Calibrate the wide camera relative to the robot base or fixture.
8. Measure capture latency and timestamp accuracy.
9. Test depth noise on a flat target and a hand-like curved target.

### Outputs

- Intrinsics JSON files.
- Camera-to-robot and camera-to-pivot transforms.
- Depth-error report in millimetres.
- Valid working-distance range.
- Timestamp and synchronization report.

### Pass condition

The depth and coordinate transformations must be accurate enough for the intended nozzle clearance. The acceptable limit must be defined from the mechanical and dispensing tests, not guessed.

## Stage 1: Load and visualize a neutral MANO hand

### Tasks

1. Register on the official MANO website.
2. Review the licence for commercial use by Idea8.
3. Download the left and right MANO model files.
4. Install the `smplx` Python package.
5. Generate a neutral MANO mesh.
6. Export it as PLY or OBJ.
7. Visualize joints, vertex coordinates and triangle faces.
8. Test changing several finger-pose and shape parameters.

### Suggested environment

```bash
python3 -m venv mano_env
source mano_env/bin/activate

pip install torch torchvision numpy scipy opencv-python \
    trimesh open3d smplx
```

Additional libraries may later include:

- PyTorch3D or another differentiable renderer.
- MediaPipe for initial 2D landmarks.
- HaMeR for initial MANO estimates.
- Segment Anything or a dedicated hand-segmentation model.

### Pass condition

The program can generate, articulate, display and export both left- and right-hand MANO meshes with correct units and orientation.

## Stage 2: Wide RGB initialization

### Tasks

1. Capture one RGB image that contains the complete hand.
2. Detect the hand bounding box and 2D landmarks.
3. Determine whether it is a left or right hand.
4. Obtain an initial MANO pose using HaMeR or another MANO-based estimator.
5. Project the predicted mesh back onto the RGB image.
6. Visually verify every finger and the wrist.

### Pass condition

The projected MANO mesh follows the correct hand, finger order and approximate pose. It does not yet need millimetre-level accuracy.

## Stage 3: Single RGB-D frame fitting

This is the first essential feasibility experiment.

### Required dataset

```text
dataset/
└── subject_001/
    └── sequence_001/
        └── frame_000000/
            ├── rgb.png
            ├── depth.png
            ├── hand_mask.png
            ├── camera_intrinsics.json
            └── metadata.json
```

### Example camera-intrinsics file

```json
{
  "width": 1280,
  "height": 720,
  "fx": 0.0,
  "fy": 0.0,
  "cx": 0.0,
  "cy": 0.0,
  "distortion_model": "replace_with_actual_model",
  "distortion_coefficients": []
}
```

The zeros are placeholders and must be replaced with measured or SDK-provided values.

### Example metadata file

```json
{
  "frame_id": 0,
  "timestamp_ns": 0,
  "handedness": "right",
  "depth_unit_m": 0.001,
  "rgb_depth_aligned": true,
  "motor_angle_deg": 0.0,
  "orbit_radius_m": 0.192,
  "camera_model": "replace_with_actual_camera",
  "subject_id": "subject_001"
}
```

### Optimization order

Do not optimize every parameter at once initially.

1. Load the HaMeR or landmark-based initialization.
2. Optimize global translation.
3. Optimize global orientation.
4. Optimize finger pose.
5. Optimize the shared MANO shape.
6. Add the silhouette term.
7. Add the depth surface term.
8. Add priors and robust outlier handling.

This staged order reduces the risk that one parameter incorrectly compensates for another.

### Outputs

- `fitted_hand.ply`
- Mesh overlay on RGB.
- Rendered MANO depth image.
- Measured-versus-rendered depth residual image.
- Landmark error in pixels.
- Surface error in millimetres.
- Optimization-loss history.

### Pass condition

The visible mesh surface follows the valid hand depth, finger identity is correct, and errors are stable across repeated runs.

## Stage 4: Rigid multi-view partial scanning

The hand must remain completely still during this stage.

### Tasks

1. Capture 3–5 close RGB-D views.
2. Maintain 30–50% overlap between neighbouring views.
3. Record mechanical camera pose and timestamps.
4. Fit one shared \(\beta\) across all frames.
5. Keep one consistent hand pose unless small correction is necessary.
6. Refine each camera transform.
7. Transform all measured points into one hand coordinate frame.
8. Compare angle-based registration with optional local refinement.

ICP may be used here only for small residual rigid corrections after the known mechanical transformation. It should not be the primary solution and should not be applied across genuinely different finger poses.

### Pass condition

All partial clouds land on the correct palm or finger without:

- Double surfaces.
- Swapped finger correspondences.
- Large seams between views.
- Alignment dominated by the table or background.

## Stage 5: Dense personalized surface

### Tasks

1. Establish point-to-MANO-surface correspondences.
2. Store canonical surface offsets.
3. Fuse repeated observations using confidence weights.
4. Smooth the offset field.
5. Fill unobserved regions with the MANO prior.
6. Mark measured and inferred regions.
7. Export the watertight or robot-usable mesh.

### Suggested weighting factors

Each depth sample's weight may depend on:

- Camera depth confidence.
- Distance from the camera.
- Angle between the viewing ray and surface normal.
- Landmark or MANO fitting confidence.
- Motion blur.
- Segmentation certainty.
- Agreement with neighbouring views.
- Temporal stability.

### Pass condition

The dorsal hand area intended for mehendi is mostly supported by real depth observations and satisfies the geometric error requirement.

## Stage 6: Controlled hand-movement experiment

Only start this stage after rigid multi-view fitting works.

### Tasks

1. Keep the wide camera fixed and continuously visible.
2. Track the full MANO pose for every frame.
3. Synchronize wide RGB and close RGB-D observations.
4. Estimate \(\theta_t\), \(R_t\), and \(T_t\) for every timestamp.
5. Canonicalize each close depth observation.
6. Fuse the canonical observations.
7. Compare the result with a scan captured while the hand remained still.

### Initial movement limits

Begin with:

- Small global translation.
- Small wrist rotation.
- No finger bending.

Then test:

- One slowly bending finger.
- Multiple small joint movements.
- Brief occlusion.

Do not begin with arbitrary or fast motion.

### Pass condition

The canonicalized surface remains single and consistent, and movement does not create duplicated fingers or thickened skin surfaces.

## Stage 7: Design projection and path generation

After the personalized hand mesh is validated:

1. Select the dorsal drawing region.
2. Parameterize or locally flatten that surface.
3. Map the 2D mehendi design onto the 3D mesh.
4. Sample the projected curves into ordered waypoints.
5. Calculate a surface normal at every waypoint.
6. Offset the nozzle by the required clearance.
7. Smooth the position and orientation trajectory.
8. Check joint limits, collisions, reachability and singularities.
9. Simulate the complete path.
10. Execute first with the dispenser disabled.

The tool pose can be represented as:

\[
p_{nozzle}=p_{skin}+d\,n
\]

where:

- \(p_{skin}\) is the surface point.
- \(n\) is the outward surface normal.
- \(d\) is the required nozzle offset.

## Stage 8: Live drawing-time tracking

The scan represents the hand at one state. If the hand moves afterward, the original robot trajectory is no longer aligned.

During drawing:

- Track global hand position and orientation continuously.
- Transform the trajectory for small rigid movement.
- Stop for meaningful finger bending or non-rigid deformation.
- Stop when tracking confidence drops.
- Retract the nozzle when surface-position uncertainty exceeds the allowed limit.
- Stop when the hand leaves the calibrated workspace.

Recommended logic:

| Detected condition | Response |
|---|---|
| Small rigid hand movement | Update the complete path transform |
| Slow movement within tested tolerance | Pause, update pose and continue only after validation |
| Finger bending or local deformation | Stop and recompute affected trajectory |
| Low hand-tracking confidence | Stop and retract |
| Depth disagreement above threshold | Stop and rescan |
| Camera or robot synchronization lost | Stop and retract |
| Hand enters an unsafe region | Emergency stop |

MANO helps estimate movement, but it does not remove the need for safety limits.

## 11. Proposed software layout

```text
hena_jet_hand_reconstruction/
├── configs/
│   ├── cameras.yaml
│   ├── calibration.yaml
│   ├── fitting.yaml
│   └── safety.yaml
├── data/
│   └── subject_001/
├── models/
│   └── mano/
├── src/
│   ├── capture/
│   │   ├── wide_camera.py
│   │   ├── rgbd_camera.py
│   │   └── synchronizer.py
│   ├── calibration/
│   │   ├── intrinsics.py
│   │   ├── extrinsics.py
│   │   └── hand_eye.py
│   ├── preprocessing/
│   │   ├── hand_segmentation.py
│   │   ├── depth_filtering.py
│   │   └── depth_to_cloud.py
│   ├── initialization/
│   │   ├── hand_landmarks.py
│   │   └── mano_initializer.py
│   ├── fitting/
│   │   ├── losses.py
│   │   ├── single_frame.py
│   │   └── multi_frame.py
│   ├── fusion/
│   │   ├── canonicalize.py
│   │   ├── surface_offsets.py
│   │   └── confidence_fusion.py
│   ├── validation/
│   │   ├── metrics.py
│   │   └── visual_reports.py
│   └── robot/
│       ├── design_projection.py
│       ├── trajectory.py
│       └── safety_monitor.py
├── tests/
├── outputs/
├── requirements.txt
└── README.md
```

## 12. Core fitting pseudocode

```python
# Shared identity-dependent hand shape
beta = initialize_shape()

# Per-frame variables
theta = initialize_pose_for_each_frame()
global_rotation = initialize_rotation_for_each_frame()
global_translation = initialize_translation_for_each_frame()

for iteration in range(number_of_iterations):
    total_loss = 0.0

    for frame_index, frame in enumerate(frames):
        vertices, joints = mano(
            beta=beta,
            hand_pose=theta[frame_index],
        )

        vertices_camera = apply_global_transform(
            vertices,
            global_rotation[frame_index],
            global_translation[frame_index],
        )

        joints_camera = apply_global_transform(
            joints,
            global_rotation[frame_index],
            global_translation[frame_index],
        )

        total_loss += (
            depth_weight
            * depth_surface_loss(
                vertices_camera,
                frame.depth_points,
                frame.depth_confidence,
            )
            + keypoint_weight
            * keypoint_reprojection_loss(
                joints_camera,
                frame.rgb_keypoints,
                frame.keypoint_confidence,
                frame.camera_intrinsics,
            )
            + silhouette_weight
            * silhouette_loss(
                vertices_camera,
                frame.hand_mask,
                frame.camera_intrinsics,
            )
        )

    total_loss += pose_prior_weight * pose_prior(theta)
    total_loss += shape_prior_weight * shape_prior(beta)
    total_loss += temporal_weight * temporal_loss(theta)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

This is architectural pseudocode. The production version must handle mesh faces, visibility, occlusion, robust losses, valid-depth masks and differentiable rendering.

## 13. Depth preprocessing

Depth should be filtered carefully before fitting.

Recommended steps:

1. Align depth with RGB using calibrated geometry.
2. Apply the binary hand mask.
3. Remove invalid and zero-depth pixels.
4. Reject measurements outside the physical hand range.
5. Use the camera's confidence map where available.
6. Apply a light edge-aware depth filter.
7. Reject flying pixels at depth discontinuities.
8. Remove isolated points with gentle Statistical Outlier Removal.
9. Preserve fingertips and other thin structures.

Statistical Outlier Removal should begin with conservative settings such as:

```python
clean_cloud, kept_indices = cloud.remove_statistical_outlier(
    nb_neighbors=30,
    std_ratio=2.0,
)
```

Aggressive filtering can remove thin fingers, nail edges or high-curvature areas.

## 14. Validation plan

Visual inspection alone is not sufficient. The system should report numerical errors.

### 14.1 Camera validation

- Depth error on a calibrated flat plane.
- Depth repeatability.
- RGB-depth registration error.
- Reprojection error.
- Extrinsic calibration error.
- Timestamp difference between cameras.

### 14.2 MANO fitting validation

- Mean and median 2D landmark error in pixels.
- Mean and median point-to-surface error in millimetres.
- 90th and 95th percentile surface errors.
- Silhouette IoU.
- Percentage of valid depth pixels explained by the mesh.
- Per-finger error.
- Per-region confidence map.

### 14.3 Multi-view validation

- Surface thickness after fusion.
- Distance between overlapping observations.
- Number and location of duplicated surfaces.
- Cross-view landmark consistency.
- Difference between rigid and canonicalized reconstruction.

### 14.4 Robot-use validation

- Nozzle-to-surface distance error.
- Normal-direction error.
- Trajectory repeatability.
- Robot tracking error.
- End-to-end calibration error.
- Maximum error during small hand movement.

### 14.5 Ground truth

Where possible, compare against:

- A calibration object with known geometry.
- A high-quality commercial hand scan.
- A plaster or rigid hand model.
- Manual measurements of hand width and finger length.

A rigid hand model is especially useful for separating reconstruction error from real biological movement.

## 15. Data-quality requirements

Every capture should include:

- RGB image.
- Aligned depth image.
- Raw depth where possible.
- Hand mask.
- Depth confidence map, if available.
- Camera intrinsics.
- Camera extrinsics.
- Mechanical angle and step count.
- Wide-camera image.
- 2D landmarks and confidence.
- Timestamp from a common clock.
- Left/right hand label.
- Subject and sequence identifiers.
- Exposure and depth-mode settings.

Avoid changing camera resolution, autofocus, depth preset or crop without recording the new calibration.

## 16. Major risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Only a tiny hand section is visible | Ambiguous finger identity and pose | Add a fixed wide camera and preserve overlap |
| Close camera is below its valid depth range | Missing or biased depth | Test the actual camera at the required distance before model work |
| RGB-depth misalignment | Mesh is pulled toward incorrect points | Calibrate and validate registration |
| Hand moves during rigid experiments | Double surfaces and failed ICP | Immobilize the hand for early stages |
| Finger bending during fusion | Non-rigid distortion | Estimate MANO pose and canonicalize each frame |
| MANO smooths real surface details | Incorrect local nozzle height | Add depth-derived canonical surface offsets |
| Unseen regions look plausible but are unmeasured | Unsafe drawing over uncertain geometry | Store confidence and restrict drawing to measured areas |
| Mechanical orbit pose is inaccurate | Misregistered partial scans | Calibrate orbit and refine residual pose |
| Segmentation includes table/background | Incorrect fitting | Use hand mask, depth range and visibility checks |
| Fast motion causes blur or invalid depth | Tracking failure | Limit movement, monitor confidence and pause |
| Commercial licence restrictions | Product-use limitation | Review MANO and dependent-model licences before integration |
| Optimizer finds a wrong local minimum | Incorrect yet visually plausible mesh | Use strong initialization and staged optimization |

## 17. Safety requirements

The robot will operate close to human skin, so safety must be part of the architecture.

Required safeguards include:

- Physical emergency-stop button.
- Software emergency stop.
- Mechanical limit switches.
- Maximum speed and acceleration limits near the hand.
- Maximum permitted nozzle force or compliant end-effector design.
- Safe retract direction and retract distance.
- Hand-presence and hand-movement monitoring.
- Camera-confidence monitoring.
- Robot-watchdog monitoring.
- Automatic stop on missing timestamps or stale sensor data.
- Dry-run mode with the dispenser disabled.
- Simulation and unreachable-pose detection.
- Conservative exclusion zones around fingers and wrist during early trials.

The first tests should use a rigid artificial hand, not a human hand.

## 18. Recommended implementation milestones

| Milestone | Deliverable | Success criterion |
|---|---|---|
| M1 | Camera validation report | Close camera provides repeatable depth at the real distance |
| M2 | Neutral MANO viewer | Left/right MANO meshes load, articulate and export correctly |
| M3 | Wide RGB initialization | Complete hand pose is correctly identified |
| M4 | Single-frame RGB-D fitting | MANO follows visible depth with measured residuals |
| M5 | Rigid partial multi-view fit | Partial scans align without double surfaces |
| M6 | Personalized surface offsets | Dorsal surface is refined using real depth |
| M7 | Controlled-motion canonicalization | Small movement does not thicken or duplicate the hand |
| M8 | 2D-design projection | Design maps continuously onto the validated mesh |
| M9 | Robot dry run | Nozzle path is safe and follows a rigid hand model |
| M10 | Live tracking and stop logic | Tested movement and confidence failures stop safely |

## 19. Immediate starting task

The first practical dataset should contain:

```text
wide_rgb.png
close_rgb.png
close_depth.png
close_hand_mask.png
wide_camera_intrinsics.json
close_camera_intrinsics.json
wide_to_close_extrinsics.json
metadata.json
```

For the first attempt:

- Use one clearly visible right or left hand.
- Keep the hand completely still.
- Make sure the wide image shows the complete hand.
- Make sure the close image includes a recognizable palm/finger region.
- Record the true depth unit.
- Save the original unmodified images.

Then build the first program:

```text
Load RGB-D
    ↓
Generate masked depth cloud
    ↓
Load MANO
    ↓
Initialize pose from wide RGB
    ↓
Place mesh in close-camera coordinates
    ↓
Fit global transform, pose and shape
    ↓
Save fitted_hand.ply
    ↓
Report surface error in millimetres
```

Do not begin with multi-frame movement compensation. The single-frame fit must be correct and measurable first.

## 20. Final recommended workflow

### Scanning phase

```text
Calibrated wide RGB + close RGB-D capture
        ↓
Wide-view MANO initialization
        ↓
Per-frame hand pose and global motion
        ↓
RGB keypoint + silhouette + depth fitting
        ↓
Canonicalize partial depth observations
        ↓
Confidence-weighted non-rigid fusion
        ↓
Personalized MANO surface
        ↓
Measured-region confidence check
```

### Drawing phase

```text
Validated personalized hand mesh
        ↓
Project 2D mehendi design
        ↓
Generate offset nozzle trajectory
        ↓
Check reachability and collisions
        ↓
Dry run
        ↓
Live hand tracking
        ↓
Draw only while pose and confidence remain valid
```

## 21. Recommended references

- [Official MANO model](https://mano.is.tue.mpg.de/)
- [SMPL-X and MANO Python implementation](https://github.com/vchoutas/smplx)
- [HaMeR project](https://geopavlakos.github.io/hamer/)
- [HaMeR GitHub repository](https://github.com/geopavlakos/hamer)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [DynamicFusion paper](https://rse-lab.cs.washington.edu/papers/dynamic-fusion-cvpr-2015.pdf)
- [ShaRPy RGB-D MANO fitting paper](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Wirth_ShaRPy_Shape_Reconstruction_and_Hand_Pose_Estimation_from_RGB-D_with_ICCVW_2023_paper.pdf)

## 22. Final recommendation

Use MANO as the anatomical structure and motion model, not as the final source of surface accuracy. Use the wide camera to keep the complete hand observable, the close RGB-D camera to measure real local geometry, and canonical non-rigid fusion to combine partial observations.

For the first prototype:

1. Keep the hand still.
2. Prove single-frame RGB-D fitting.
3. Prove rigid multi-view partial reconstruction.
4. Add personalized surface offsets.
5. Introduce only small, controlled hand movement.
6. Add live tracking before any drawing test.
7. Test on a rigid artificial hand before operating near a real hand.

This staged approach provides measurable checkpoints and reduces the risk of building a complex moving-hand system before the basic depth fitting, calibration and safety requirements are proven.

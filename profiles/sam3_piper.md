# Robot Embodiment Declaration — SAM3 Piper Grasping System

> Profile: sam3_piper | Driver: SAM3PiperDriver

## Identity

- **Name**: SAM3 + Agilex Piper Grasping System
- **Type**: 6-DOF desktop robotic arm with zero-shot language-driven grasping
- **Hardware**: Agilex Piper arm (CAN bus) + Intel RealSense D405 depth camera
- **Vision Model**: Meta SAM3 (Segment Anything Model 3) — zero-shot, text-prompted

## Sensors

- **RGB-D**: Intel RealSense D405 (depth + color, 640×480, 30fps)
- **Segmentation**: SAM3 HTTP server (POST /predict, text prompt → binary mask)
- **Point Cloud**: RealSense depth → 3D centroid via hand-eye calibration

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `grasp` | `target_name: str` | Detect target in current view and grasp it. Uses SAM3 segmentation + IK. |
| `explore_and_grasp` | `target_name: str` | Rotate arm joint0 in steps to search for target, then grasp when found. |
| `explore_and_place` | `target_name: str` | While holding an object, rotate to find target container and release above it. |
| `explore_right_and_place` | `target_name: str` | Same as explore_and_place but rotates rightward (negative joint0 direction). |
| `check_target` | `target_name: str` | Check if target is currently visible in camera view. Returns found/not_found. |
| `go_home` | — | Move arm to standby observation pose (X=0.25, Y=0, Z=0.32, Pitch=135°). |
| `calibrate` | — | Run full AX=XB hand-eye calibration. Requires calibration board in view. |
| `connect` | — | Establish CAN bus connection and enable arm. |
| `disconnect` | — | Disable arm and release CAN bus. |

## Physical Constraints

- **Workspace radius**: max 0.46 m from base (enforced by driver)
- **Reachable height**: Z ∈ [0.05, 0.55] m above base
- **Gripper**: binary open/close (no force feedback)
- **Payload**: ≤ 1.5 kg
- **Collision policy**: IK solver rejects unreachable poses; driver returns `success: false`
- **Calibration requirement**: Hand-eye matrix must exist at `calibration_result.npz` before grasping

## Connection

- **Transport**: local (CAN bus via `can0` interface)
- **Remote mode**: HTTP bridge at `http://<host>:18791` (when `mode=remote`)
- **CAN activation**: run `can_activate.sh` before starting driver
- **SAM3 server**: must be running at `http://localhost:8000` (or configured `sam3_url`)

## Runtime Protocol

- **Connection channel**: `robots.sam3_piper_001.connection_state`
- **Pose channel**: `robots.sam3_piper_001.robot_pose` (joint angles, not Cartesian)
- **Arm state channel**: `robots.sam3_piper_001.arm_state`
- **Health owner**: `hal_watchdog.py` calls `health_check()` each poll cycle

## Operational Notes

- Call `go_home` before any grasp sequence to ensure consistent starting pose
- `explore_and_grasp` covers ~360° in 12 steps of −30° each (joint0)
- SAM3 inference latency: 0.5–2 s per call; actions are blocking
- Hand-eye calibration is persistent across restarts (loaded from `calibration_result.npz`)

"""
generate_mock_dannce.py
=======================
Generate mock DANNCE 3D keypoint data with realistic spatiotemporal characteristics.

The synthetic rat exhibits:
  - Anatomically plausible skeleton (23 joints, mm scale)
  - Behavioural state transitions: rest → walk → groom → rear → run (via Markov chain)
  - Smooth locomotion trajectories (Ornstein-Uhlenbeck COM drift)
  - Per-joint micro-motion (breathing, fidget) layered on top
  - Temporal smoothness enforced by low-pass filtering
  - Optional multi-session / multi-animal batch generation

Output: .npy file with shape (T, 23, 3)

Usage
-----
    python generate_mock_dannce.py                          # defaults: 5000 frames, 1 session
    python generate_mock_dannce.py --frames 20000 --fps 50  # longer session
    python generate_mock_dannce.py --n_sessions 5 --outdir ./mock_data  # batch
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  1. Skeleton definition — rest-pose offsets (mm) relative to spine_mid
#     Based on adult Sprague-Dawley rat (~250g), approximate proportions.
# ═══════════════════════════════════════════════════════════════════════════

JOINT_NAMES: list[str] = [
    "nose", "head_midline", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "spine_mid",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "spine_posterior", "hip_midline",
    "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "tail_base", "tail_mid", "tail_tip",
    "spine_anterior",
]
N_KP = 23
assert len(JOINT_NAMES) == N_KP

# Rest-pose template: (x_forward, y_lateral, z_height) in mm
# x+: nose direction, y+: left, z+: up
# spine_mid is origin (index 6)
REST_POSE = np.array([
    # head
    [ 65.0,   0.0,   5.0],   # nose
    [ 50.0,   0.0,   8.0],   # head_midline
    [ 45.0,   9.0,  12.0],   # left_ear
    [ 45.0,  -9.0,  12.0],   # right_ear
    # forelimbs
    [ 20.0,  14.0,  -5.0],   # left_shoulder
    [ 20.0, -14.0,  -5.0],   # right_shoulder
    [  0.0,   0.0,   0.0],   # spine_mid (origin)
    [ 18.0,  20.0, -18.0],   # left_elbow
    [ 18.0, -20.0, -18.0],   # right_elbow
    [ 22.0,  22.0, -30.0],   # left_wrist
    [ 22.0, -22.0, -30.0],   # right_wrist
    # torso / hip
    [-25.0,   0.0,  -2.0],   # spine_posterior
    [-35.0,   0.0,  -3.0],   # hip_midline
    [-30.0,  16.0,  -6.0],   # left_hip
    [-30.0, -16.0,  -6.0],   # right_hip
    [-38.0,  18.0, -20.0],   # left_knee
    [-38.0, -18.0, -20.0],   # right_knee
    [-45.0,  16.0, -30.0],   # left_ankle
    [-45.0, -16.0, -30.0],   # right_ankle
    # tail
    [-45.0,   0.0,  -2.0],   # tail_base
    [-75.0,   0.0,   0.0],   # tail_mid
    [-105.0,  0.0,   2.0],   # tail_tip
    # spine_anterior (between shoulders and head)
    [ 35.0,   0.0,   3.0],   # spine_anterior
], dtype=np.float64)

assert REST_POSE.shape == (N_KP, 3)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Behavioural states — Markov chain
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviourState:
    name: str
    com_speed_range: tuple[float, float]       # mm/frame at 50fps
    limb_amplitude: float                       # cyclic limb motion scale (mm)
    limb_freq: float                            # limb cycle frequency (Hz)
    head_bob_amp: float                         # head vertical oscillation (mm)
    body_pitch: float                           # forward tilt (radians)
    rear_height: float                          # extra z lift for rearing (mm)
    groom_amp: float                            # forelimb groom oscillation (mm)
    tail_wag_amp: float                         # lateral tail motion (mm)
    breathing_amp: float                        # torso micro-expansion (mm)


BEHAVIOURS: dict[str, BehaviourState] = {
    "rest":  BehaviourState("rest",  (0.0, 0.3),   0.5,  0.5,  0.2, 0.0,   0.0,  0.0,  0.5,  0.8),
    "walk":  BehaviourState("walk",  (0.8, 2.5),   3.5,  2.0,  1.0, 0.05,  0.0,  0.0,  2.0,  1.0),
    "run":   BehaviourState("run",   (3.0, 6.0),   6.0,  4.5,  2.0, 0.10,  0.0,  0.0,  3.5,  1.2),
    "groom": BehaviourState("groom", (0.0, 0.2),   0.3,  0.3,  0.3, -0.05, 0.0,  5.0,  0.5,  0.9),
    "rear":  BehaviourState("rear",  (0.0, 0.5),   1.0,  0.8,  0.5, -0.30, 35.0, 0.0,  1.5,  1.0),
}

STATE_NAMES = list(BEHAVIOURS.keys())

# Transition matrix (row = from, col = to) — rows sum to 1
# Designed for realistic dwell times: rest & walk dominate, rear & groom are brief
TRANSITION_MATRIX = np.array([
    #  rest   walk   run   groom  rear
    [  0.92,  0.04,  0.005, 0.025, 0.01 ],   # rest  → mostly stay
    [  0.03,  0.90,  0.04,  0.01,  0.02 ],   # walk  → may speed up or pause
    [  0.02,  0.06,  0.90,  0.005, 0.015],   # run   → decelerate back
    [  0.10,  0.03,  0.00,  0.85,  0.02 ],   # groom → back to rest
    [  0.08,  0.04,  0.01,  0.01,  0.86 ],   # rear  → drop down
], dtype=np.float64)

assert TRANSITION_MATRIX.shape == (5, 5)
assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  3. State sequence generator (with minimum dwell time)
# ═══════════════════════════════════════════════════════════════════════════

def generate_state_sequence(
    T: int,
    fps: float = 50.0,
    min_dwell_frames: int = 25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a Markov-chain behaviour label sequence of length T.
    Enforces a minimum dwell time to avoid unrealistic rapid switching.

    Returns
    -------
    labels : np.ndarray of int, shape (T,)
    """
    rng = rng or np.random.default_rng()
    labels = np.zeros(T, dtype=np.int32)
    current = 0  # start at rest
    t = 0
    while t < T:
        dwell = max(min_dwell_frames, int(rng.exponential(scale=fps * 2)))
        dwell = min(dwell, T - t)
        labels[t : t + dwell] = current
        t += dwell
        if t < T:
            current = rng.choice(len(STATE_NAMES), p=TRANSITION_MATRIX[current])
    return labels


# ═══════════════════════════════════════════════════════════════════════════
#  4. COM trajectory — Ornstein-Uhlenbeck with behavioural speed modulation
# ═══════════════════════════════════════════════════════════════════════════

def generate_com_trajectory(
    T: int,
    labels: np.ndarray,
    fps: float = 50.0,
    arena_radius: float = 300.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a 2D centre-of-mass trajectory on a circular arena floor.

    Uses an Ornstein-Uhlenbeck process with:
      - speed scaled by current behaviour state
      - soft wall repulsion at arena boundary
      - heading persistence (smooth turning)

    Returns
    -------
    com_xy : np.ndarray, shape (T, 2)
    """
    rng = rng or np.random.default_rng()
    dt = 1.0 / fps

    pos = np.zeros((T, 2), dtype=np.float64)
    heading = rng.uniform(0, 2 * np.pi)
    heading_rate = 0.0

    theta_ou = 2.0    # mean-reversion strength for heading
    sigma_h = 3.0     # heading noise intensity

    for t in range(1, T):
        bstate = BEHAVIOURS[STATE_NAMES[labels[t]]]
        speed = rng.uniform(*bstate.com_speed_range)

        # Soft wall: steer away from boundary
        dist_from_centre = np.linalg.norm(pos[t - 1])
        if dist_from_centre > arena_radius * 0.7:
            angle_to_centre = np.arctan2(-pos[t - 1, 1], -pos[t - 1, 0])
            wall_correction = 0.3 * np.sin(angle_to_centre - heading)
        else:
            wall_correction = 0.0

        # Update heading (OU on angular velocity)
        heading_rate += (-theta_ou * heading_rate + sigma_h * rng.standard_normal() + wall_correction) * dt
        heading += heading_rate * dt

        dx = speed * np.cos(heading)
        dy = speed * np.sin(heading)
        pos[t] = pos[t - 1] + np.array([dx, dy])

        # Hard clamp inside arena
        r = np.linalg.norm(pos[t])
        if r > arena_radius:
            pos[t] *= arena_radius / r

    return pos


# ═══════════════════════════════════════════════════════════════════════════
#  5. Joint-level motion synthesis
# ═══════════════════════════════════════════════════════════════════════════

def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# Joint groups for targeted motion
IDX_FORELIMB_L = [7, 9]     # left_elbow, left_wrist
IDX_FORELIMB_R = [8, 10]    # right_elbow, right_wrist
IDX_HINDLIMB_L = [15, 17]   # left_knee, left_ankle
IDX_HINDLIMB_R = [16, 18]   # right_knee, right_ankle
IDX_HEAD = [0, 1, 2, 3]     # nose, head_midline, ears
IDX_TAIL = [19, 20, 21]     # tail_base, tail_mid, tail_tip
IDX_TORSO = [4, 5, 6, 11, 12, 22]  # shoulders, spine_mid, spine_post, hip_mid, spine_ant


def synthesize_keypoints(
    T: int,
    labels: np.ndarray,
    com_xy: np.ndarray,
    fps: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build full (T, 23, 3) keypoint array by composing:
      1. Rest pose skeleton
      2. Body orientation (heading from COM trajectory)
      3. Body pitch (behaviour-dependent)
      4. Rearing height offset
      5. Cyclic limb motion (gait)
      6. Grooming forelimb oscillation
      7. Head bobbing
      8. Tail wagging
      9. Breathing micro-motion on torso
     10. Per-joint Gaussian jitter (sensor noise)
    """
    rng = rng or np.random.default_rng()
    dt = 1.0 / fps
    keypoints = np.zeros((T, N_KP, 3), dtype=np.float64)

    # Heading from COM velocity (smoothed)
    vel_xy = np.gradient(com_xy, axis=0)
    raw_heading = np.arctan2(vel_xy[:, 1], vel_xy[:, 0])
    heading = np.unwrap(raw_heading)
    heading = gaussian_filter1d(heading, sigma=fps * 0.3)

    # Smooth behaviour blend weights for transitions
    behaviour_weights = np.zeros((T, len(STATE_NAMES)), dtype=np.float64)
    for i in range(len(STATE_NAMES)):
        behaviour_weights[:, i] = (labels == i).astype(np.float64)
    for i in range(len(STATE_NAMES)):
        behaviour_weights[:, i] = gaussian_filter1d(behaviour_weights[:, i], sigma=fps * 0.1)
    row_sums = behaviour_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    behaviour_weights /= row_sums

    # Phase accumulators
    limb_phase = 0.0
    groom_phase = 0.0

    for t in range(T):
        w = behaviour_weights[t]
        limb_amp = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].limb_amplitude for i in range(len(STATE_NAMES)))
        limb_freq = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].limb_freq for i in range(len(STATE_NAMES)))
        head_bob = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].head_bob_amp for i in range(len(STATE_NAMES)))
        pitch = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].body_pitch for i in range(len(STATE_NAMES)))
        rear_z = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].rear_height for i in range(len(STATE_NAMES)))
        groom_a = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].groom_amp for i in range(len(STATE_NAMES)))
        tail_wag = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].tail_wag_amp for i in range(len(STATE_NAMES)))
        breath_a = sum(w[i] * BEHAVIOURS[STATE_NAMES[i]].breathing_amp for i in range(len(STATE_NAMES)))

        limb_phase += limb_freq * dt * 2 * np.pi
        groom_phase += 3.0 * dt * 2 * np.pi

        # 1. Rest pose
        pose = REST_POSE.copy()

        # 2. Body pitch
        pose = pose @ _rotation_y(pitch).T

        # 3. Rearing lift
        pose[:, 2] += rear_z

        # 4. Cyclic limb gait (alternating)
        fl = limb_amp * np.sin(limb_phase)
        fr = limb_amp * np.sin(limb_phase + np.pi)
        hl = limb_amp * np.sin(limb_phase + np.pi)
        hr = limb_amp * np.sin(limb_phase)

        for idx in IDX_FORELIMB_L:
            pose[idx, 0] += fl * 0.6
            pose[idx, 2] += max(fl, 0) * 0.4
        for idx in IDX_FORELIMB_R:
            pose[idx, 0] += fr * 0.6
            pose[idx, 2] += max(fr, 0) * 0.4
        for idx in IDX_HINDLIMB_L:
            pose[idx, 0] += hl * 0.8
            pose[idx, 2] += max(hl, 0) * 0.5
        for idx in IDX_HINDLIMB_R:
            pose[idx, 0] += hr * 0.8
            pose[idx, 2] += max(hr, 0) * 0.5

        # 5. Grooming
        if groom_a > 0.1:
            groom_dx = groom_a * np.sin(groom_phase)
            groom_dz = groom_a * np.cos(groom_phase) * 0.5
            for idx in IDX_FORELIMB_L + IDX_FORELIMB_R:
                pose[idx, 0] += groom_dx
                pose[idx, 2] += groom_dz + groom_a * 0.8

        # 6. Head bobbing
        pose[IDX_HEAD, 2] += head_bob * np.sin(limb_phase * 2)

        # 7. Tail wagging
        for k, idx in enumerate(IDX_TAIL):
            tail_gain = (k + 1) / len(IDX_TAIL)
            pose[idx, 1] += tail_wag * tail_gain * np.sin(limb_phase * 1.5 + k * 0.5)

        # 8. Breathing
        breath_phase = t * dt * 2 * np.pi * 2.5
        for idx in IDX_TORSO:
            side = 1.0 if REST_POSE[idx, 1] >= 0 else -1.0
            pose[idx, 1] += side * breath_a * np.sin(breath_phase) * 0.3

        # 9. Rotate to heading & translate to COM
        pose = pose @ _rotation_z(heading[t]).T
        ground_offset = 32.0 + rear_z * 0.1
        pose[:, 0] += com_xy[t, 0]
        pose[:, 1] += com_xy[t, 1]
        pose[:, 2] += ground_offset

        # 10. Sensor noise
        pose += rng.normal(scale=0.3, size=pose.shape)

        keypoints[t] = pose

    # Global temporal smoothing
    for j in range(N_KP):
        for d in range(3):
            keypoints[:, j, d] = gaussian_filter1d(keypoints[:, j, d], sigma=1.5)

    return keypoints


# ═══════════════════════════════════════════════════════════════════════════
#  6. Validation & statistics
# ═══════════════════════════════════════════════════════════════════════════

def validate_and_report(keypoints: np.ndarray, labels: np.ndarray, fps: float) -> dict:
    T = keypoints.shape[0]
    dt = 1.0 / fps

    com = keypoints.mean(axis=1)
    com_vel = np.linalg.norm(np.diff(com, axis=0), axis=1) / dt

    joint_vel = np.linalg.norm(np.diff(keypoints, axis=0), axis=2) / dt

    nose_tail = np.linalg.norm(keypoints[:, 0] - keypoints[:, 21], axis=1)

    z_min_per_frame = keypoints[:, :, 2].min(axis=1)

    unique, counts = np.unique(labels, return_counts=True)
    behaviour_pct = {STATE_NAMES[u]: f"{c / T * 100:.1f}%" for u, c in zip(unique, counts)}

    # Bone length consistency (nose→head_midline, ~15mm)
    bone_nose_head = np.linalg.norm(keypoints[:, 0] - keypoints[:, 1], axis=1)
    # Left shoulder → left elbow (~22mm)
    bone_sho_elb = np.linalg.norm(keypoints[:, 4] - keypoints[:, 7], axis=1)

    stats = {
        "frames": T,
        "duration_s": T / fps,
        "com_speed_mm_s": (com_vel.mean(), com_vel.std(), com_vel.max()),
        "joint_speed_mm_s": (joint_vel.mean(), joint_vel.std(), joint_vel.max()),
        "nose_tail_dist_mm": (nose_tail.mean(), nose_tail.std()),
        "z_floor_mm": (z_min_per_frame.mean(), z_min_per_frame.min()),
        "bone_nose_head_mm": (bone_nose_head.mean(), bone_nose_head.std()),
        "bone_sho_elb_mm": (bone_sho_elb.mean(), bone_sho_elb.std()),
        "behaviour_distribution": behaviour_pct,
    }

    print("\n" + "=" * 64)
    print("  Mock DANNCE Data — Validation Report")
    print("=" * 64)
    print(f"  Frames              : {T}  ({T / fps:.1f} s @ {fps} fps)")
    print(f"  COM speed  (mm/s)   : mean={com_vel.mean():.1f}  std={com_vel.std():.1f}  max={com_vel.max():.1f}")
    print(f"  Joint speed(mm/s)   : mean={joint_vel.mean():.1f}  std={joint_vel.std():.1f}  max={joint_vel.max():.1f}")
    print(f"  Nose→tail  (mm)     : mean={nose_tail.mean():.1f}  std={nose_tail.std():.1f}")
    print(f"  Bone nose-head (mm) : mean={bone_nose_head.mean():.1f}  std={bone_nose_head.std():.1f}")
    print(f"  Bone sho-elbow (mm) : mean={bone_sho_elb.mean():.1f}  std={bone_sho_elb.std():.1f}")
    print(f"  Floor z    (mm)     : mean={z_min_per_frame.mean():.2f}  min={z_min_per_frame.min():.2f}")
    print(f"  Behaviours          : {behaviour_pct}")

    assert com_vel.mean() < 500, f"COM speed too high: {com_vel.mean():.1f} mm/s"
    assert nose_tail.mean() > 80, f"Skeleton too compressed: {nose_tail.mean():.1f} mm"
    assert nose_tail.mean() < 250, f"Skeleton too stretched: {nose_tail.mean():.1f} mm"
    assert z_min_per_frame.min() > -10, f"Rat below ground: {z_min_per_frame.min():.1f} mm"
    assert bone_nose_head.std() < 5.0, f"Bone length unstable: std={bone_nose_head.std():.2f}"

    print("  ✅ All sanity checks passed.")
    print("=" * 64 + "\n")
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  7. Main
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_session(
    T: int = 5000,
    fps: float = 50.0,
    arena_radius: float = 300.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logger.info("Generating behaviour sequence (%d frames)...", T)
    labels = generate_state_sequence(T, fps=fps, rng=rng)
    logger.info("Generating COM trajectory (arena=%.0fmm)...", arena_radius)
    com_xy = generate_com_trajectory(T, labels, fps=fps, arena_radius=arena_radius, rng=rng)
    logger.info("Synthesizing joint keypoints...")
    keypoints = synthesize_keypoints(T, labels, com_xy, fps=fps, rng=rng)
    return keypoints, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock DANNCE data")
    parser.add_argument("--frames", type=int, default=5000, help="Frames per session")
    parser.add_argument("--fps", type=float, default=50.0, help="Frame rate")
    parser.add_argument("--arena_radius", type=float, default=300.0, help="Arena radius mm")
    parser.add_argument("--n_sessions", type=int, default=1, help="Number of sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./mock_data", help="Output directory")
    parser.add_argument("--save_labels", action="store_true", help="Also save behaviour labels")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_sessions):
        session_seed = args.seed + i
        logger.info("━━━ Session %d/%d (seed=%d) ━━━", i + 1, args.n_sessions, session_seed)

        keypoints, labels = generate_mock_session(
            T=args.frames, fps=args.fps,
            arena_radius=args.arena_radius, seed=session_seed,
        )
        validate_and_report(keypoints, labels, fps=args.fps)

        kp_path = outdir / f"mock_dannce_session{i:03d}.npy"
        np.save(kp_path, keypoints.astype(np.float32))
        logger.info("Saved keypoints → %s  shape=%s", kp_path, keypoints.shape)

        if args.save_labels:
            lbl_path = outdir / f"mock_labels_session{i:03d}.npy"
            np.save(lbl_path, labels)
            logger.info("Saved labels → %s", lbl_path)

    # Integration test with dannce_loader
    logger.info("━━━ Integration test with dannce_loader ━━━")
    try:
        from data.dannce_loader import load_dannce, STATE_DIM
        first_file = outdir / "mock_dannce_session000.npy"
        seq = load_dannce(first_file, fps=args.fps)
        s = seq.states
        assert s.shape == (args.frames, STATE_DIM)
        logger.info(
            "dannce_loader: states %s | pos [%.1f,%.1f] | speed [%.1f,%.1f] | "
            "height [%.1f,%.1f] | com_vel [%.1f,%.1f]",
            s.shape,
            seq.positions.min(), seq.positions.max(),
            seq.speeds.min(), seq.speeds.max(),
            seq.heights.min(), seq.heights.max(),
            seq.com_velocity.min(), seq.com_velocity.max(),
        )
        print("✅ Integration with dannce_loader passed.\n")
    except ImportError:
        logger.warning("dannce_loader not in path — skip integration test.")
    except Exception as e:
        logger.error("Integration test failed: %s", e)


if __name__ == "__main__":
    main()

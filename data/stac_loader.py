"""
stac_loader.py
==============
STAC (Skeleton-based Tracker with Automatic Calibration) loader.

Converts DANNCE 3D keypoints → MuJoCo-compatible joint angles via
analytical inverse kinematics on a simplified rodent skeleton.

The rodent skeleton model has 38 DoF (degrees of freedom):
    - Root (free joint):  6 DoF  (3 translation + 3 rotation)
    - Spine:              6 DoF  (3 segments × 2 DoF each: pitch + yaw)
    - Head:               3 DoF  (pitch + yaw + roll)
    - Left forelimb:      5 DoF  (shoulder 3-DoF + elbow 1-DoF + wrist 1-DoF)
    - Right forelimb:     5 DoF
    - Left hindlimb:      5 DoF  (hip 3-DoF + knee 1-DoF + ankle 1-DoF)
    - Right hindlimb:     5 DoF
    - Tail:               3 DoF  (3 segments × 1 DoF each: pitch)
    Total: 6 + 6 + 3 + 5×4 + 3 = 38

Output joint angle vector: q(t) ∈ ℝ³⁸

Pipeline snapshot (.pt) contains:
    states      : (T, 116)  — from dannce_loader
    joint_angles: (T, 38)   — from stac_loader
    meta        : dict      — fps, source, dimensions, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
N_JOINTS_DOF = 38       # Total degrees of freedom
N_KEYPOINTS = 23

# Joint angle vector layout
JOINT_NAMES_DOF: list[str] = [
    # Root (6)
    "root_tx", "root_ty", "root_tz",
    "root_rx", "root_ry", "root_rz",
    # Spine (6)
    "spine_ant_pitch", "spine_ant_yaw",
    "spine_mid_pitch", "spine_mid_yaw",
    "spine_post_pitch", "spine_post_yaw",
    # Head (3)
    "head_pitch", "head_yaw", "head_roll",
    # Left forelimb (5)
    "l_shoulder_pitch", "l_shoulder_yaw", "l_shoulder_roll",
    "l_elbow_pitch",
    "l_wrist_pitch",
    # Right forelimb (5)
    "r_shoulder_pitch", "r_shoulder_yaw", "r_shoulder_roll",
    "r_elbow_pitch",
    "r_wrist_pitch",
    # Left hindlimb (5)
    "l_hip_pitch", "l_hip_yaw", "l_hip_roll",
    "l_knee_pitch",
    "l_ankle_pitch",
    # Right hindlimb (5)
    "r_hip_pitch", "r_hip_yaw", "r_hip_roll",
    "r_knee_pitch",
    "r_ankle_pitch",
    # Tail (3)
    "tail_base_pitch", "tail_mid_pitch", "tail_tip_pitch",
]
assert len(JOINT_NAMES_DOF) == N_JOINTS_DOF

# Joint angle limits (radians) — anatomically reasonable for rodent
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # Root translation: ±arena
    "root_tx": (-400.0, 400.0), "root_ty": (-400.0, 400.0), "root_tz": (0.0, 100.0),
    # Root rotation
    "root_rx": (-0.5, 0.5), "root_ry": (-0.5, 0.5), "root_rz": (-np.pi, np.pi),
    # Spine
    "spine_ant_pitch": (-0.4, 0.6), "spine_ant_yaw": (-0.3, 0.3),
    "spine_mid_pitch": (-0.3, 0.5), "spine_mid_yaw": (-0.4, 0.4),
    "spine_post_pitch": (-0.3, 0.4), "spine_post_yaw": (-0.3, 0.3),
    # Head
    "head_pitch": (-0.8, 0.8), "head_yaw": (-0.6, 0.6), "head_roll": (-0.3, 0.3),
    # Forelimbs
    "l_shoulder_pitch": (-1.2, 1.2), "l_shoulder_yaw": (-0.8, 0.8),
    "l_shoulder_roll": (-0.5, 0.5), "l_elbow_pitch": (-1.5, 0.1),
    "l_wrist_pitch": (-0.8, 0.8),
    "r_shoulder_pitch": (-1.2, 1.2), "r_shoulder_yaw": (-0.8, 0.8),
    "r_shoulder_roll": (-0.5, 0.5), "r_elbow_pitch": (-1.5, 0.1),
    "r_wrist_pitch": (-0.8, 0.8),
    # Hindlimbs
    "l_hip_pitch": (-1.0, 1.5), "l_hip_yaw": (-0.6, 0.6),
    "l_hip_roll": (-0.4, 0.4), "l_knee_pitch": (-0.1, 1.8),
    "l_ankle_pitch": (-1.0, 1.0),
    "r_hip_pitch": (-1.0, 1.5), "r_hip_yaw": (-0.6, 0.6),
    "r_hip_roll": (-0.4, 0.4), "r_knee_pitch": (-0.1, 1.8),
    "r_ankle_pitch": (-1.0, 1.0),
    # Tail
    "tail_base_pitch": (-0.8, 0.8), "tail_mid_pitch": (-0.6, 0.6),
    "tail_tip_pitch": (-0.5, 0.5),
}

# Keypoint indices for IK reference
KP_SPINE_MID = 6
KP_SPINE_ANT = 22
KP_SPINE_POST = 11
KP_HIP_MID = 12
KP_HEAD_MID = 1
KP_NOSE = 0
KP_L_SHOULDER = 4
KP_R_SHOULDER = 5
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 13
KP_R_HIP = 14
KP_L_KNEE = 15
KP_R_KNEE = 16
KP_L_ANKLE = 17
KP_R_ANKLE = 18
KP_TAIL_BASE = 19
KP_TAIL_MID = 20
KP_TAIL_TIP = 21


# ── Skeleton bone lengths (mm, from REST_POSE) ────────────────────────────

# Reference bone lengths for IK (approximate, from rest pose)
BONE_LENGTHS: dict[str, float] = {
    "spine_ant_to_mid": 35.0,
    "spine_mid_to_post": 25.0,
    "spine_post_to_hip": 10.0,
    "head_mid_to_nose": 15.0,
    "shoulder_to_elbow": 13.0,
    "elbow_to_wrist": 13.0,
    "hip_to_knee": 13.0,
    "knee_to_ankle": 13.0,
    "tail_base_to_mid": 30.0,
    "tail_mid_to_tip": 30.0,
}


# ── Helper functions ───────────────────────────────────────────────────────

def _safe_arctan2(y: float, x: float) -> float:
    return np.arctan2(y, x)


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 3D vectors in radians."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _extract_limb_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    body_forward: np.ndarray,
    body_up: np.ndarray,
    side: str = "left",
) -> tuple[float, float, float, float, float]:
    """
    Extract 5-DoF limb joint angles from 3 keypoints.

    Returns (shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_pitch, wrist_pitch).
    """
    body_lateral = np.cross(body_forward, body_up)
    if side == "right":
        body_lateral = -body_lateral
    body_lateral /= np.linalg.norm(body_lateral) + 1e-12

    # Upper arm vector
    upper = elbow - shoulder
    upper_norm = upper / (np.linalg.norm(upper) + 1e-12)

    # Shoulder angles
    shoulder_pitch = _safe_arctan2(
        -np.dot(upper_norm, body_up),
        np.dot(upper_norm, body_forward),
    )
    shoulder_yaw = _safe_arctan2(
        np.dot(upper_norm, body_lateral),
        np.sqrt(np.dot(upper_norm, body_forward)**2 + np.dot(upper_norm, body_up)**2 + 1e-12),
    )
    shoulder_roll = 0.0  # Simplified: no roll measurement from 3 points

    # Elbow: angle between upper arm and forearm
    forearm = wrist - elbow
    elbow_angle = _angle_between_vectors(upper, forearm)
    elbow_pitch = -(np.pi - elbow_angle)  # Flexion is negative

    # Wrist: simplified pitch from forearm direction
    forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-12)
    wrist_pitch = _safe_arctan2(-np.dot(forearm_norm, body_up), np.dot(forearm_norm, body_forward)) * 0.3

    return shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_pitch, wrist_pitch


# ── Core IK solver ─────────────────────────────────────────────────────────

def keypoints_to_joint_angles(kp: np.ndarray) -> np.ndarray:
    """
    Convert a single frame of keypoints (23, 3) to joint angles (38,).

    Uses analytical inverse kinematics on the simplified skeleton.

    Parameters
    ----------
    kp : np.ndarray, shape (23, 3)
        One frame of keypoint positions in mm.

    Returns
    -------
    q : np.ndarray, shape (38,)
        Joint angle vector.
    """
    q = np.zeros(N_JOINTS_DOF, dtype=np.float64)

    # ── Root (6 DoF) ──
    # Translation: spine_mid position
    com = kp.mean(axis=0)
    q[0] = com[0]  # tx
    q[1] = com[1]  # ty
    q[2] = com[2]  # tz

    # Body axes from spine
    spine_fwd = kp[KP_SPINE_ANT] - kp[KP_SPINE_POST]
    spine_fwd_norm = spine_fwd / (np.linalg.norm(spine_fwd) + 1e-12)

    # Root rotation: heading (rz), pitch (ry), roll (rx)
    q[5] = _safe_arctan2(spine_fwd_norm[1], spine_fwd_norm[0])  # rz (heading)
    q[4] = _safe_arctan2(-spine_fwd_norm[2],
                         np.sqrt(spine_fwd_norm[0]**2 + spine_fwd_norm[1]**2 + 1e-12))  # ry (pitch)

    # Roll from shoulder tilt
    shoulder_vec = kp[KP_L_SHOULDER] - kp[KP_R_SHOULDER]
    body_up = np.cross(spine_fwd_norm, shoulder_vec)
    body_up /= np.linalg.norm(body_up) + 1e-12
    q[3] = _safe_arctan2(shoulder_vec[2], np.linalg.norm(shoulder_vec[:2]) + 1e-12) * 0.5  # rx

    # ── Spine (6 DoF) ──
    # Anterior segment
    seg_ant = kp[KP_SPINE_ANT] - kp[KP_SPINE_MID]
    seg_ant_n = seg_ant / (np.linalg.norm(seg_ant) + 1e-12)
    q[6] = _safe_arctan2(-seg_ant_n[2], np.linalg.norm(seg_ant_n[:2]) + 1e-12)  # pitch
    q[7] = _safe_arctan2(seg_ant_n[1], seg_ant_n[0]) - q[5]  # yaw relative to heading

    # Mid segment
    seg_mid = kp[KP_SPINE_MID] - kp[KP_SPINE_POST]
    seg_mid_n = seg_mid / (np.linalg.norm(seg_mid) + 1e-12)
    q[8] = _safe_arctan2(-seg_mid_n[2], np.linalg.norm(seg_mid_n[:2]) + 1e-12)
    q[9] = _safe_arctan2(seg_mid_n[1], seg_mid_n[0]) - q[5]

    # Posterior segment
    seg_post = kp[KP_SPINE_POST] - kp[KP_HIP_MID]
    seg_post_n = seg_post / (np.linalg.norm(seg_post) + 1e-12)
    q[10] = _safe_arctan2(-seg_post_n[2], np.linalg.norm(seg_post_n[:2]) + 1e-12)
    q[11] = _safe_arctan2(seg_post_n[1], seg_post_n[0]) - q[5]

    # ── Head (3 DoF) ──
    head_vec = kp[KP_NOSE] - kp[KP_HEAD_MID]
    head_n = head_vec / (np.linalg.norm(head_vec) + 1e-12)
    q[12] = _safe_arctan2(-head_n[2], np.linalg.norm(head_n[:2]) + 1e-12)  # pitch
    q[13] = _safe_arctan2(head_n[1], head_n[0]) - q[5]  # yaw
    q[14] = 0.0  # roll (hard to measure from 2 midline points)

    # ── Forelimbs (5 DoF each) ──
    fl = _extract_limb_angles(
        kp[KP_L_SHOULDER], kp[KP_L_ELBOW], kp[KP_L_WRIST],
        spine_fwd_norm, body_up, "left",
    )
    q[15:20] = fl

    fr = _extract_limb_angles(
        kp[KP_R_SHOULDER], kp[KP_R_ELBOW], kp[KP_R_WRIST],
        spine_fwd_norm, body_up, "right",
    )
    q[20:25] = fr

    # ── Hindlimbs (5 DoF each) ──
    hl = _extract_limb_angles(
        kp[KP_L_HIP], kp[KP_L_KNEE], kp[KP_L_ANKLE],
        -spine_fwd_norm, body_up, "left",  # Hind faces backward
    )
    q[25:30] = hl

    hr = _extract_limb_angles(
        kp[KP_R_HIP], kp[KP_R_KNEE], kp[KP_R_ANKLE],
        -spine_fwd_norm, body_up, "right",
    )
    q[30:35] = hr

    # ── Tail (3 DoF) ──
    tail_seg1 = kp[KP_TAIL_MID] - kp[KP_TAIL_BASE]
    tail_seg2 = kp[KP_TAIL_TIP] - kp[KP_TAIL_MID]
    q[35] = _safe_arctan2(-tail_seg1[2], np.linalg.norm(tail_seg1[:2]) + 1e-12)
    q[36] = _safe_arctan2(-tail_seg2[2], np.linalg.norm(tail_seg2[:2]) + 1e-12) - q[35]
    q[37] = 0.0  # Tip simplified

    # ── Clamp to limits ──
    for i, name in enumerate(JOINT_NAMES_DOF):
        lo, hi = JOINT_LIMITS[name]
        q[i] = _clamp(q[i], lo, hi)

    return q


def batch_keypoints_to_joint_angles(
    keypoints: np.ndarray,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Convert (T, 23, 3) keypoints to (T, 38) joint angles.

    Parameters
    ----------
    keypoints : np.ndarray, shape (T, 23, 3)
    smooth_sigma : float
        Gaussian smoothing sigma (frames). Set 0 to disable.

    Returns
    -------
    joint_angles : np.ndarray, shape (T, 38)
    """
    T = keypoints.shape[0]
    angles = np.zeros((T, N_JOINTS_DOF), dtype=np.float64)

    for t in range(T):
        angles[t] = keypoints_to_joint_angles(keypoints[t])

    # Temporal smoothing
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        for j in range(N_JOINTS_DOF):
            angles[:, j] = gaussian_filter1d(angles[:, j], sigma=smooth_sigma)
        # Re-clamp after smoothing
        for j, name in enumerate(JOINT_NAMES_DOF):
            lo, hi = JOINT_LIMITS[name]
            angles[:, j] = np.clip(angles[:, j], lo, hi)

    logger.info("STAC IK: %d frames → (%d, %d) joint angles", T, T, N_JOINTS_DOF)
    return angles


# ── STACResult container ───────────────────────────────────────────────────

@dataclass
class STACResult:
    """STAC registration result for one session."""
    joint_angles: np.ndarray     # (T, 38)
    keypoints: np.ndarray        # (T, 23, 3) — original
    states: np.ndarray           # (T, 116) — from dannce_loader
    fps: float = 50.0
    source_path: Optional[str] = None

    # ── Derived properties ──

    @property
    def T(self) -> int:
        return self.joint_angles.shape[0]

    @property
    def joint_velocities(self) -> np.ndarray:
        """Joint angular velocities (T, 38), rad/s. First frame = 0."""
        dq = np.diff(self.joint_angles, axis=0) * self.fps
        return np.vstack([np.zeros((1, N_JOINTS_DOF)), dq])

    def get_joint_by_name(self, name: str) -> np.ndarray:
        """Get time series of a single joint angle by name."""
        idx = JOINT_NAMES_DOF.index(name)
        return self.joint_angles[:, idx]

    # ── Export ──

    def save_snapshot(self, path: Union[str, Path]) -> Path:
        """
        Save pipeline snapshot as .pt (torch) or .npz (numpy fallback).

        If torch is available → .pt with torch.Tensor values.
        Otherwise → .npz with numpy arrays (rename .npz → .pt for consistency).

        Contents:
            states       : (T, 116) float32
            joint_angles : (T, 38)  float32
            meta         : dict
        """
        path = Path(path)
        meta = {
            "T": self.T,
            "fps": self.fps,
            "state_dim": self.states.shape[1],
            "joint_dof": self.joint_angles.shape[1],
            "joint_names": JOINT_NAMES_DOF,
            "source": self.source_path or "unknown",
        }

        try:
            import torch
            snapshot = {
                "states": torch.from_numpy(self.states).float(),
                "joint_angles": torch.from_numpy(self.joint_angles).float(),
                "meta": meta,
            }
            torch.save(snapshot, path)
        except ImportError:
            import pickle
            snapshot = {
                "states": self.states.astype(np.float32),
                "joint_angles": self.joint_angles.astype(np.float32),
                "meta": meta,
            }
            with open(path, "wb") as f:
                pickle.dump(snapshot, f, protocol=4)
            logger.info("(torch not available, saved as pickle)")

        logger.info("Saved snapshot → %s  (%d frames)", path, self.T)
        return path

    @staticmethod
    def load_snapshot(path: Union[str, Path]) -> dict:
        """
        Load a .pt snapshot (torch or pickle fallback).

        Returns
        -------
        dict with keys: states, joint_angles, meta
        """
        path = Path(path)
        try:
            import torch
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                snapshot = pickle.load(f)
        logger.info(
            "Loaded snapshot ← %s  (%d frames, state_dim=%d, joint_dof=%d)",
            path, snapshot["meta"]["T"],
            snapshot["meta"]["state_dim"], snapshot["meta"]["joint_dof"],
        )
        return snapshot


# ── High-level API ─────────────────────────────────────────────────────────

def run_stac(
    keypoints: np.ndarray,
    states: np.ndarray,
    fps: float = 50.0,
    smooth_sigma: float = 1.0,
    source_path: Optional[str] = None,
) -> STACResult:
    """
    Run STAC registration: keypoints → joint angles, package with states.

    Parameters
    ----------
    keypoints : np.ndarray, (T, 23, 3)
    states : np.ndarray, (T, 116) — from dannce_loader
    fps : float
    smooth_sigma : float
    source_path : str, optional

    Returns
    -------
    STACResult
    """
    assert keypoints.shape[0] == states.shape[0], \
        f"Frame count mismatch: keypoints {keypoints.shape[0]} vs states {states.shape[0]}"

    joint_angles = batch_keypoints_to_joint_angles(keypoints, smooth_sigma=smooth_sigma)

    return STACResult(
        joint_angles=joint_angles,
        keypoints=keypoints,
        states=states,
        fps=fps,
        source_path=source_path,
    )
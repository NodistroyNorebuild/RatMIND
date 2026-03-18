"""
dannce_loader.py
================
Load DANNCE 3D keypoint data and construct 93-dim state vectors.

State vector layout (93-dim):
    [0:69]   — de-centred keypoint positions (23 joints × 3 coords)
    [69:92]  — keypoint speeds (23 joints, scalar per joint)
    [92:93] — centre-of-mass velocity (scalar)

Expected DANNCE input format:
    .mat / .hdf5 / .npy file with shape (T, 23, 3) — T frames, 23 keypoints, xyz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
N_KEYPOINTS = 23
POS_DIM = N_KEYPOINTS * 3        # 69
SPEED_DIM = N_KEYPOINTS           # 23
COM_VEL_DIM = 1                   # 1
STATE_DIM = POS_DIM + SPEED_DIM + COM_VEL_DIM  # 93

# Default DANNCE keypoint names (rat, 23 markers)
DEFAULT_KEYPOINT_NAMES: list[str] = [
    "nose", "head_midline", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "spine_mid",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "spine_posterior", "hip_midline",
    "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "tail_base", "tail_mid", "tail_tip",
    "spine_anterior",
]

assert len(DEFAULT_KEYPOINT_NAMES) == N_KEYPOINTS


# ── Data container ─────────────────────────────────────────────────────────
@dataclass
class DANNCESequence:
    """One continuous recording session."""
    raw_keypoints: np.ndarray          # (T, 23, 3)  — original coordinates
    fps: float = 50.0                  # DANNCE default capture rate
    keypoint_names: list[str] = field(default_factory=lambda: list(DEFAULT_KEYPOINT_NAMES))
    source_path: Optional[str] = None

    # ── Computed on demand (lazy) ──
    _positions: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _speeds: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _com_vel: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _states: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        kp = self.raw_keypoints
        if kp.ndim != 3 or kp.shape[1] != N_KEYPOINTS or kp.shape[2] != 3:
            raise ValueError(
                f"Expected shape (T, {N_KEYPOINTS}, 3), got {kp.shape}"
            )
        if len(self.keypoint_names) != N_KEYPOINTS:
            raise ValueError(
                f"Need {N_KEYPOINTS} keypoint names, got {len(self.keypoint_names)}"
            )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def T(self) -> int:
        return self.raw_keypoints.shape[0]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def positions(self) -> np.ndarray:
        """De-centred keypoint positions (T, 69). COM subtracted per frame."""
        if self._positions is None:
            self._compute_all()
        return self._positions  # type: ignore[return-value]

    @property
    def speeds(self) -> np.ndarray:
        """Per-joint speed scalars (T, 23)."""
        if self._speeds is None:
            self._compute_all()
        return self._speeds  # type: ignore[return-value]

    @property
    def com_velocity(self) -> np.ndarray:
        """Centre-of-mass velocity scalar (T, 1)."""
        if self._com_vel is None:
            self._compute_all()
        return self._com_vel  # type: ignore[return-value]

    @property
    def states(self) -> np.ndarray:
        """Full 93-dim state vectors (T, 93)."""
        if self._states is None:
            self._compute_all()
        return self._states  # type: ignore[return-value]

    # ── Core computation ───────────────────────────────────────────────────

    def _compute_all(self) -> None:
        kp = self.raw_keypoints.astype(np.float64)  # (T, 23, 3)
        T = kp.shape[0]

        # 1) Centre of mass per frame (simple mean over joints)
        com = kp.mean(axis=1, keepdims=True)  # (T, 1, 3)

        # 2) De-centred positions → flatten to 69-dim
        decentred = kp - com                       # (T, 23, 3)
        self._positions = decentred.reshape(T, -1)  # (T, 69)

        # 3) Per-joint speed: ||Δp|| / dt  (forward diff, first frame = 0)
        dp = np.diff(kp, axis=0)                    # (T-1, 23, 3)
        joint_disp = np.linalg.norm(dp, axis=2)     # (T-1, 23)
        joint_speed = joint_disp / self.dt
        # Pad first frame with zeros
        self._speeds = np.vstack([np.zeros((1, N_KEYPOINTS)), joint_speed])  # (T, 23)


        # 4) COM velocity scalar: ||Δcom|| / dt
        com_flat = com.squeeze(1)                     # (T, 3)
        dcom = np.diff(com_flat, axis=0)              # (T-1, 3)
        com_speed = np.linalg.norm(dcom, axis=1, keepdims=True) / self.dt  # (T-1, 1)
        self._com_vel = np.vstack([np.zeros((1, 1)), com_speed])           # (T, 1)

        # 5) Concatenate → 93-dim state
        self._states = np.concatenate(
            [self._positions, self._speeds, self._com_vel],
            axis=1,
        )
        assert self._states.shape == (T, STATE_DIM), (
            f"State shape mismatch: {self._states.shape} vs expected ({T}, {STATE_DIM})"
        )

    def invalidate_cache(self) -> None:
        """Call if raw_keypoints is mutated in-place."""
        self._positions = self._speeds = self._com_vel = self._states = None

    # ── Slicing / windowing ────────────────────────────────────────────────

    def get_window(self, start: int, length: int) -> np.ndarray:
        """Return a (length, 93) state window. Clamps to valid range."""
        end = min(start + length, self.T)
        start = max(0, start)
        return self.states[start:end]

    def iter_windows(self, length: int, stride: int = 1):
        """Yield (start_idx, window) tuples."""
        for i in range(0, self.T - length + 1, stride):
            yield i, self.states[i : i + length]


# ── I/O helpers ────────────────────────────────────────────────────────────

def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _load_mat(path: Path) -> np.ndarray:
    """Load DANNCE .mat (v7.3 / HDF5 or legacy)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            # DANNCE convention: variable usually named 'pred' or 'predictions'
            for key in ("pred", "predictions", "keypoints", "data"):
                if key in f:
                    arr = np.array(f[key])
                    # h5py may store as (3, 23, T) — transpose if needed
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        arr = arr.transpose(2, 1, 0)
                    return arr
            raise KeyError(f"No recognised dataset in {path}. Keys: {list(f.keys())}")
    except Exception:
        from scipy.io import loadmat
        mat = loadmat(str(path))
        for key in ("pred", "predictions", "keypoints", "data"):
            if key in mat:
                arr = mat[key]
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = arr.transpose(2, 1, 0)
                return arr
        raise KeyError(f"No recognised variable in {path}. Keys: {[k for k in mat if not k.startswith('__')]}")


def _load_hdf5(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        for key in ("pred", "predictions", "keypoints", "data"):
            if key in f:
                arr = np.array(f[key])
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = arr.transpose(2, 1, 0)
                return arr
        raise KeyError(f"No recognised dataset in {path}. Keys: {list(f.keys())}")


_LOADERS = {
    ".npy": _load_npy,
    ".mat": _load_mat,
    ".h5": _load_hdf5,
    ".hdf5": _load_hdf5,
}


def load_dannce(
    path: Union[str, Path],
    fps: float = 50.0,
    keypoint_names: Optional[list[str]] = None,
) -> DANNCESequence:
    """
    Load a DANNCE keypoint file and return a DANNCESequence.

    Parameters
    ----------
    path : str or Path
        Path to .npy / .mat / .h5 / .hdf5 file with shape (T, 23, 3).
    fps : float
        Capture frame rate (default 50 Hz, standard DANNCE).
    keypoint_names : list[str], optional
        Override default 23 keypoint names.

    Returns
    -------
    DANNCESequence
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DANNCE file not found: {path}")

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type '{suffix}'. Use: {list(_LOADERS.keys())}")

    raw = loader(path).astype(np.float64)
    if raw.ndim != 3 or raw.shape[1] != N_KEYPOINTS or raw.shape[2] != 3:
        raise ValueError(
            f"Loaded array shape {raw.shape} does not match (T, {N_KEYPOINTS}, 3)"
        )

    logger.info("Loaded %s: %d frames @ %.1f fps", path.name, raw.shape[0], fps)

    return DANNCESequence(
        raw_keypoints=raw,
        fps=fps,
        keypoint_names=keypoint_names or list(DEFAULT_KEYPOINT_NAMES),
        source_path=str(path),
    )


def load_multiple(
    paths: Sequence[Union[str, Path]],
    fps: float = 50.0,
) -> list[DANNCESequence]:
    """Load multiple DANNCE files."""
    return [load_dannce(p, fps=fps) for p in paths]


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    T_test = 500
    fake_kp = np.random.randn(T_test, N_KEYPOINTS, 3) * 10  # mm scale
    # Add a slow drift to COM so com_velocity is non-trivial
    drift = np.linspace(0, 50, T_test).reshape(-1, 1, 1) * np.array([1, 0.5, 0])
    fake_kp += drift

    seq = DANNCESequence(raw_keypoints=fake_kp, fps=50.0)
    s = seq.states

    print(f"Frames          : {seq.T}")
    print(f"State shape     : {s.shape}")
    print(f"State dim       : {s.shape[1]}  (expect {STATE_DIM})")
    print(f"Positions range : [{seq.positions.min():.2f}, {seq.positions.max():.2f}]")
    print(f"Speeds range    : [{seq.speeds.min():.2f}, {seq.speeds.max():.2f}]")
    print(f"COM vel range   : [{seq.com_velocity.min():.2f}, {seq.com_velocity.max():.2f}]")

    # Window iterator
    wins = list(seq.iter_windows(length=64, stride=32))
    print(f"Windows (64/32) : {len(wins)}")
    print("✅ dannce_loader sanity check passed.")

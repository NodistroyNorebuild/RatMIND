"""
mock_generator_stac.py
======================
Generate mock STAC data by running the full pipeline:
    mock DANNCE keypoints → dannce_loader → stac_loader → .pt snapshot

This is the authoritative way to produce test data for downstream modules.

Usage:
    python mock_generator_stac.py                              # defaults
    python mock_generator_stac.py --frames 5000 --seed 42
    python mock_generator_stac.py --out snapshot_train.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_stac_snapshot(
    T: int = 5000,
    fps: float = 50.0,
    arena_radius: float = 300.0,
    seed: Optional[int] = None,
    smooth_sigma: float = 1.0,
    out_path: Optional[str] = None,
) -> dict:
    """
    End-to-end: mock keypoints → 116-dim states + 38-dim joint angles → .pt

    Parameters
    ----------
    T : int
        Number of frames.
    fps : float
        Frame rate.
    arena_radius : float
        Arena radius for mock trajectory.
    seed : int, optional
        Random seed for reproducibility.
    smooth_sigma : float
        Temporal smoothing for joint angles.
    out_path : str, optional
        If provided, save snapshot to this path.

    Returns
    -------
    snapshot : dict
        Keys: states (torch.Tensor), joint_angles (torch.Tensor), meta (dict)
    """
    from .mock_generator_dannce import generate_mock_session
    from .dannce_loader import DANNCESequence
    from .stac_loader import run_stac

    logger.info("━━━ Generating STAC snapshot (T=%d, seed=%s) ━━━", T, seed)

    # Step 1: Mock DANNCE keypoints
    keypoints, labels = generate_mock_session(T=T, fps=fps, arena_radius=arena_radius, seed=seed)
    logger.info("Mock keypoints: %s", keypoints.shape)

    # Step 2: DANNCE loader → 116-dim states
    seq = DANNCESequence(raw_keypoints=keypoints, fps=fps)
    states = seq.states
    logger.info("States: %s", states.shape)

    # Step 3: STAC → joint angles
    result = run_stac(
        keypoints=keypoints,
        states=states,
        fps=fps,
        smooth_sigma=smooth_sigma,
        source_path=f"mock_session(T={T}, seed={seed})",
    )
    logger.info("Joint angles: %s", result.joint_angles.shape)

    # Step 4: Save if requested
    if out_path is not None:
        result.save_snapshot(out_path)

    # Step 5: Return the snapshot dict
    import torch
    snapshot = {
        "states": torch.from_numpy(result.states).float(),
        "joint_angles": torch.from_numpy(result.joint_angles).float(),
        "meta": {
            "T": result.T,
            "fps": fps,
            "state_dim": states.shape[1],
            "joint_dof": result.joint_angles.shape[1],
            "source": f"mock_session(T={T}, seed={seed})",
            "behaviour_labels": labels.tolist(),
        },
    }
    return snapshot


def validate_snapshot(snapshot: dict) -> None:
    """Sanity checks on a snapshot dict."""
    import torch

    states = snapshot["states"]
    angles = snapshot["joint_angles"]
    meta = snapshot["meta"]

    T = meta["T"]
    assert states.shape == (T, 93), f"States shape: {states.shape}"
    assert angles.shape == (T, 38), f"Angles shape: {angles.shape}"
    assert torch.isfinite(states).all(), "States contain NaN/Inf"
    assert torch.isfinite(angles).all(), "Angles contain NaN/Inf"

    # Joint angle ranges
    q = angles.numpy()
    from .stac_loader import JOINT_NAMES_DOF, JOINT_LIMITS
    for j, name in enumerate(JOINT_NAMES_DOF):
        lo, hi = JOINT_LIMITS[name]
        assert q[:, j].min() >= lo - 1e-6, f"{name} below limit: {q[:, j].min():.4f} < {lo}"
        assert q[:, j].max() <= hi + 1e-6, f"{name} above limit: {q[:, j].max():.4f} > {hi}"

    print("=" * 60)
    print("  STAC Snapshot — Validation Report")
    print("=" * 60)
    print(f"  Frames         : {T} ({T / meta['fps']:.1f}s @ {meta['fps']} fps)")
    print(f"  States         : {states.shape}  range [{states.min():.1f}, {states.max():.1f}]")
    print(f"  Joint angles   : {angles.shape}  range [{angles.min():.3f}, {angles.max():.3f}] rad")

    # Per-group statistics
    groups = {
        "Root trans": (0, 3), "Root rot": (3, 6),
        "Spine": (6, 12), "Head": (12, 15),
        "L-forelimb": (15, 20), "R-forelimb": (20, 25),
        "L-hindlimb": (25, 30), "R-hindlimb": (30, 35),
        "Tail": (35, 38),
    }
    for gname, (i0, i1) in groups.items():
        g = q[:, i0:i1]
        print(f"  {gname:14s} : mean={g.mean():+.3f}  std={g.std():.3f}  "
              f"range=[{g.min():.3f}, {g.max():.3f}]")

    print("  ✅ All checks passed.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock STAC snapshot (.pt)")
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth", type=float, default=1.0, help="Smoothing sigma")
    parser.add_argument("--out", type=str, default="./mock_data/stac_snapshot.pt")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        from .stac_loader import STACResult
        snapshot = STACResult.load_snapshot(args.out)
        validate_snapshot(snapshot)
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    snapshot = generate_stac_snapshot(
        T=args.frames, fps=args.fps, seed=args.seed,
        smooth_sigma=args.smooth, out_path=args.out,
    )
    validate_snapshot(snapshot)

    size_mb = Path(args.out).stat().st_size / 1024 / 1024
    print(f"\n💾 Snapshot saved: {args.out} ({size_mb:.1f} MB)")
    print(f"   下游使用: snapshot = torch.load('{args.out}')")
    print(f"   snapshot['states'].shape      = {snapshot['states'].shape}")
    print(f"   snapshot['joint_angles'].shape = {snapshot['joint_angles'].shape}")


if __name__ == "__main__":
    main()
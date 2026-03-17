"""
configs/generate_snapshot.py
=============================
一键生成 STAC pipeline 断点文件 (stac_snapshot.pt)

用法：
    python configs/generate_snapshot.py
    python configs/generate_snapshot.py --frames 10000 --seed 0
    python configs/generate_snapshot.py --out ./mock_data/train.pt --frames 20000
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 import 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import generate_stac_snapshot, validate_snapshot


def main():
    parser = argparse.ArgumentParser(
        description="生成 STAC pipeline 断点文件 (.pt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames", type=int, default=5000, help="帧数")
    parser.add_argument("--fps", type=float, default=50.0, help="帧率 (Hz)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--smooth", type=float, default=1.0, help="关节角平滑 sigma")
    parser.add_argument("--out", type=str, default="./mock_data/stac_snapshot.pt",
                        help="输出路径")
    parser.add_argument("--no_validate", action="store_true", help="跳过验证")
    args = parser.parse_args()

    # 创建输出目录
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RatMIND — 生成 STAC Snapshot")
    print("=" * 60)
    print(f"  帧数   : {args.frames} ({args.frames / args.fps:.1f}s)")
    print(f"  帧率   : {args.fps} Hz")
    print(f"  种子   : {args.seed}")
    print(f"  平滑   : σ={args.smooth}")
    print(f"  输出   : {args.out}")
    print("=" * 60)

    snapshot = generate_stac_snapshot(
        T=args.frames,
        fps=args.fps,
        seed=args.seed,
        smooth_sigma=args.smooth,
        out_path=str(out_path),
    )

    if not args.no_validate:
        validate_snapshot(snapshot)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print()
    print("─" * 60)
    print(f"  ✅ 已生成: {out_path}")
    print(f"  📦 大小  : {size_mb:.2f} MB")
    print(f"  📐 states      : {snapshot['states'].shape}")
    print(f"  📐 joint_angles: {snapshot['joint_angles'].shape}")
    print("─" * 60)
    print()
    print("  下游使用:")
    print(f"    import torch")
    print(f"    snap = torch.load('{out_path}')")
    print(f"    states = snap['states']           # {snapshot['states'].shape}")
    print(f"    angles = snap['joint_angles']     # {snapshot['joint_angles'].shape}")
    print(f"    meta   = snap['meta']             # T, fps, joint_names, ...")
    print()


if __name__ == "__main__":
    main()
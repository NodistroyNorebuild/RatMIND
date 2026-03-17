"""
configs/fit_irl.py
===================
读取 stac_snapshot.pt → 拟合 MaxEnt IRL → 保存 irl_result.pt

用法：
    python configs/fit_irl.py
    python configs/fit_irl.py --hidden_dim 128 --max_iter 500
    python configs/fit_irl.py --snapshot ./mock_data/train.pt --out ./results/irl.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import STACResult
from upstream.irl import MaxEntIRL
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="拟合 MaxEnt IRL 并保存结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot", type=str, default="./mock_data/stac_snapshot.pt",
                        help="输入: stac_snapshot.pt 路径")
    parser.add_argument("--out", type=str, default="./mock_data/irl_result.pt",
                        help="输出: irl_result.pt 路径")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="MLP 隐层宽度")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="残差块数量")
    parser.add_argument("--max_iter", type=int, default=200,
                        help="训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch 大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("  RatMIND — MaxEnt IRL Fitting")
    print("=" * 60)

    snap = STACResult.load_snapshot(args.snapshot)
    states = np.array(snap["states"])
    meta_in = snap["meta"]

    print(f"  输入       : {args.snapshot}")
    print(f"  帧数       : {meta_in['T']} ({meta_in['T'] / meta_in['fps']:.1f}s)")
    print(f"  状态维度   : {states.shape[1]}")
    print(f"  hidden_dim : {args.hidden_dim}")
    print(f"  n_blocks   : {args.n_blocks}")
    print(f"  max_iter   : {args.max_iter}")
    print(f"  batch_size : {args.batch_size}")
    print("=" * 60)

    irl = MaxEntIRL(
        D=states.shape[1],
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )
    losses = irl.fit(
        demo_states=states,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    irl.save_result(out_path, states)

    u = irl.forward(states)
    stats = irl.utility_stats(states)
    size_kb = out_path.stat().st_size / 1024

    print()
    print("─" * 60)
    print(f"  ✅ 已保存: {out_path} ({size_kb:.1f} KB)")
    print(f"  Utilities    : {u.shape}")
    print(f"  u(s) mean    : {stats['mean']:.3f}")
    print(f"  u(s) std     : {stats['std']:.3f}")
    print(f"  u(s) range   : [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Final loss   : {losses[-1]:.4f}")
    print("─" * 60)
    print()
    print("  下游使用:")
    print(f"    from upstream.irl import MaxEntIRL")
    print(f"    result = MaxEntIRL.load_result('{out_path}')")
    print(f"    utilities = result['utilities']  # {u.shape}")
    print()


if __name__ == "__main__":
    main()
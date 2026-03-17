"""
configs/fit_hmm.py
===================
读取 stac_snapshot.pt → 拟合 HMM → 保存 hmm_result.pt

用法：
    python configs/fit_hmm.py
    python configs/fit_hmm.py --K 100 --n_iter 200
    python configs/fit_hmm.py --snapshot ./mock_data/train.pt --out ./results/hmm.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import STACResult
from upstream.hmm import GaussianHMM
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="拟合 HMM 并保存结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot", type=str, default="./mock_data/stac_snapshot.pt",
                        help="输入: stac_snapshot.pt 路径")
    parser.add_argument("--out", type=str, default="./mock_data/hmm_result.pt",
                        help="输出: hmm_result.pt 路径")
    parser.add_argument("--K", type=int, default=20, help="隐状态数 (debug=20, full=100)")
    parser.add_argument("--n_iter", type=int, default=50, help="EM 最大迭代")
    parser.add_argument("--tol", type=float, default=1e-4, help="收敛阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # ── 加载 snapshot ──
    print("=" * 60)
    print("  RatMIND — HMM Fitting")
    print("=" * 60)

    snap = STACResult.load_snapshot(args.snapshot)
    states = np.array(snap["states"])
    meta_in = snap["meta"]

    print(f"  输入     : {args.snapshot}")
    print(f"  帧数     : {meta_in['T']} ({meta_in['T'] / meta_in['fps']:.1f}s)")
    print(f"  状态维度 : {states.shape[1]}")
    print(f"  K        : {args.K}")
    print(f"  n_iter   : {args.n_iter}")
    print(f"  tol      : {args.tol}")
    print("=" * 60)

    # ── 拟合 HMM ──
    hmm = GaussianHMM(K=args.K, D=states.shape[1], seed=args.seed)
    lls = hmm.fit(states, n_iter=args.n_iter, tol=args.tol, init_from_data=True)

    # ── 推断 ──
    posteriors = hmm.forward(states)
    viterbi_path = hmm.viterbi(states)
    entropy = hmm.posterior_entropy(posteriors)
    usage = hmm.state_usage(posteriors)
    active = int((usage > 0.01).sum())

    # ── 保存结果 ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hmm.save_result(out_path, states, posteriors=posteriors)

    # ── 输出摘要 ──
    size_kb = out_path.stat().st_size / 1024
    print()
    print("─" * 60)
    print(f"  ✅ 已保存: {out_path} ({size_kb:.1f} KB)")
    print(f"  Posteriors    : {posteriors.shape}")
    print(f"  Viterbi path  : {viterbi_path.shape}")
    print(f"  Active states : {active} / {args.K}")
    print(f"  Mean entropy  : {entropy.mean():.4f}")
    print(f"  EM iterations : {len(lls)}")
    print(f"  Final LL      : {lls[-1]:.2f}")
    print("─" * 60)
    print()
    print("  下游使用:")
    print(f"    from upstream.hmm import GaussianHMM")
    print(f"    hmm = GaussianHMM(K={args.K}, D={states.shape[1]})")
    print(f"    hmm.load('{out_path}')")
    print(f"    posteriors = hmm.forward(states)  # {posteriors.shape}")
    print()


if __name__ == "__main__":
    main()
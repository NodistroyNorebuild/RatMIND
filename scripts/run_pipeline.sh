#!/bin/bash
# ================================================================
#  RatMIND — 阶段一完整 pipeline
#  Step 1: STAC snapshot (mock 数据 → states + joint_angles)
#  Step 2: HMM (states → 隐状态后验 zₜ)
#  Step 3: IRL (states → 效用值 u(sₜ))
#
#  用法:
#    bash scripts/run_pipeline.sh                          # 默认
#    bash scripts/run_pipeline.sh --K 100 --frames 10000   # 正式规模
#    bash scripts/run_pipeline.sh --skip_snapshot           # 跳过 Step 1
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── 默认参数 ──
FRAMES=5000
SEED=42
K=20
HMM_ITER=50
IRL_HIDDEN=128
IRL_ITER=200
IRL_BATCH=256
SKIP_SNAPSHOT=false

SNAPSHOT="./mock_data/stac_snapshot.pt"
HMM_OUT="./mock_data/hmm_result.pt"
IRL_OUT="./mock_data/irl_result.pt"

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --frames)        FRAMES="$2"; shift 2;;
        --seed)          SEED="$2"; shift 2;;
        --K)             K="$2"; shift 2;;
        --hmm_iter)      HMM_ITER="$2"; shift 2;;
        --irl_hidden)    IRL_HIDDEN="$2"; shift 2;;
        --irl_iter)      IRL_ITER="$2"; shift 2;;
        --irl_batch)     IRL_BATCH="$2"; shift 2;;
        --skip_snapshot) SKIP_SNAPSHOT=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RatMIND — Phase 1 Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Frames       : $FRAMES"
echo "  Seed         : $SEED"
echo "  HMM K        : $K  (n_iter=$HMM_ITER)"
echo "  IRL hidden   : $IRL_HIDDEN  (max_iter=$IRL_ITER)"
echo "  Skip snapshot: $SKIP_SNAPSHOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: STAC Snapshot ──
if [ "$SKIP_SNAPSHOT" = false ]; then
    echo "▶ Step 1/3: Generating STAC snapshot..."
    echo "─────────────────────────────────────────"
    python configs/generate_snapshot.py \
        --frames "$FRAMES" \
        --seed "$SEED" \
        --out "$SNAPSHOT"
    echo ""
else
    echo "▶ Step 1/3: Skipped (using existing $SNAPSHOT)"
    echo ""
fi

# ── Step 2: HMM ──
echo "▶ Step 2/3: Fitting HMM..."
echo "─────────────────────────────────────────"
python configs/fit_hmm.py \
    --snapshot "$SNAPSHOT" \
    --out "$HMM_OUT" \
    --K "$K" \
    --n_iter "$HMM_ITER" \
    --seed "$SEED"
echo ""

# ── Step 3: IRL ──
echo "▶ Step 3/3: Fitting MaxEnt IRL..."
echo "─────────────────────────────────────────"
python configs/fit_irl.py \
    --snapshot "$SNAPSHOT" \
    --out "$IRL_OUT" \
    --hidden_dim "$IRL_HIDDEN" \
    --max_iter "$IRL_ITER" \
    --batch_size "$IRL_BATCH" \
    --seed "$SEED"
echo ""

# ── Summary ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Phase 1 Pipeline Complete"
echo ""
echo "  Outputs:"
echo "    Snapshot : $SNAPSHOT"
echo "    HMM      : $HMM_OUT"
echo "    IRL      : $IRL_OUT"
echo ""
echo "  下游使用:"
echo "    snap = torch.load('$SNAPSHOT')"
echo "    hmm.load('$HMM_OUT')"
echo "    irl_result = MaxEntIRL.load_result('$IRL_OUT')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
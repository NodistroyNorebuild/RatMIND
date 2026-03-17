#!/bin/bash
# ================================================================
#  RatMIND — 生成 STAC pipeline 断点文件
#
#  用法:
#    bash scripts/generate_snapshot.sh
#    bash scripts/generate_snapshot.sh 10000 0      # 10000帧, seed=0
#    bash scripts/generate_snapshot.sh 20000 42 ./mock_data/train.pt
# ================================================================

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 参数（带默认值）
FRAMES="${1:-5000}"
SEED="${2:-42}"
OUT="${3:-$PROJECT_ROOT/mock_data/stac_snapshot.pt}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RatMIND — Generate STAC Snapshot"
echo "  Frames : $FRAMES"
echo "  Seed   : $SEED"
echo "  Output : $OUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

python configs/generate_snapshot.py \
    --frames "$FRAMES" \
    --seed "$SEED" \
    --out "$OUT"

echo ""
echo "Done. File: $OUT"
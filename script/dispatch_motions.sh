#!/bin/bash
# 0-07_ 로 시작하는 모션 12개를 각각 하나의 policy 학습 job 으로 dispatch
# 사용: bash script/dispatch_motions.sh
set -e

MOTION_DIR="$(dirname "$0")/../motionData"

for f in "$MOTION_DIR"/0-07_*.pkl; do
    motion=$(basename "$f" .pkl)
    echo ">>> dispatch: $motion"
    nomad job dispatch -meta motion="$motion" kapex-mt-batch
done

echo
echo "전체 dispatch 상태:"
nomad job status kapex-mt-batch | tail -20

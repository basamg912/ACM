#!/bin/bash
# 0-07_ 로 시작하는 모션 12개를 각각 하나의 policy 학습 job 으로 dispatch
# 사용: bash script/train/dispatch_motions.sh
# - -detach: 배치 대기 없이 등록만 (노드가 꽉 차 있어도 pending 으로 큐잉됨)
# - 이미 pending/running 인 모션은 건너뜀 (재실행 안전)
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MOTION_DIR="${ACM_ROOT}/motionData"

# 현재 살아있는(dead 아닌) dispatch 들의 motion 목록
active_motions=$(
  for j in $(nomad job status kapex-mt-batch 2>/dev/null | awk '/dispatch-/{print $1}'); do
    st=$(nomad job status "$j" | awk '/^Status/{print $3; exit}')
    if [ "$st" != "dead" ]; then
      nomad job inspect "$j" 2>/dev/null | grep -o '"motion": *"[^"]*"' | head -1 | cut -d'"' -f4
    fi
  done
)

for f in "$MOTION_DIR"/0-07_*.pkl; do
    motion=$(basename "$f" .pkl)
    if echo "$active_motions" | grep -qx "$motion"; then
        echo "--- skip (already active): $motion"
        continue
    fi
    echo ">>> dispatch: $motion"
    nomad job dispatch -detach -meta motion="$motion" kapex-mt-batch || echo "!!! dispatch failed: $motion"
done

echo
echo "전체 dispatch 상태:"
nomad job status kapex-mt-batch | sed -n '/Parameterized Job Summary/,$p'

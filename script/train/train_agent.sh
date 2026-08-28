#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ASAP_ROOT="${ACM_ROOT}/ASAP"

if (($# == 0)); then
    echo "Usage: $0 <motion.pkl|motion-name.pkl> [hydra overrides ...]" >&2
    exit 2
fi

motion_arg="$1"
shift

if [[ "${motion_arg}" = /* ]]; then
    motion_file="${motion_arg}"
else
    candidates=(
        "${ACM_ROOT}/${motion_arg}"
        "${ACM_ROOT}/motionData/${motion_arg}"
        "${ASAP_ROOT}/${motion_arg}"
    )
    motion_file=""
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            motion_file="${candidate}"
            break
        fi
    done
fi

if [[ -z "${motion_file:-}" || ! -f "${motion_file}" ]]; then
    echo "Motion file not found: ${motion_arg}" >&2
    exit 2
fi

cd "${ASAP_ROOT}"
set -x
python humanoidverse/train_agent.py \
    +simulator=isaacsim \
    +exp=motion_tracking \
    +domain_rand=NO_domain_rand \
    +rewards=motion_tracking/reward_motion_tracking_dm_2real \
    +robot=kapex/kapex_31dof \
    +terrain=terrain_locomotion_plane \
    +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
    num_envs=4096 \
    project_name=MotionTracking \
    experiment_name=MotionTracking_CR7 \
    "robot.motion.motion_file=${motion_file}" \
    rewards.reward_penalty_curriculum=True \
    rewards.reward_penalty_degree=0.00001 \
    env.config.resample_motion_when_training=False \
    env.config.termination.terminate_when_motion_far=True \
    env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
    env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
    env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
    robot.asset.self_collisions=0 \
    "$@"

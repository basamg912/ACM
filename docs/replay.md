# open-loop action replay (sim2sim 재현성 체크)

mujoco deploy 로 수집한 rollout(motionData/*.pt)의 action 만을 isaaclab(isaacsim)에서
open-loop 재생하여, 수집 궤적(qpos)이 source simulator 에서 재현되는지 검증한다.
정책은 로드하지 않는다. 산출물: <stem>_replay_isaacsim.{pt} + _compare.{json,png,mp4}

# 실행 환경: hvlab + isaac sim 바이너리 (train_agent.py isaacsim 학습과 동일)
conda activate hvlab
source /home/kist/work/workspace/ASAP/IsaacLab/_isaac_sim/setup_conda_env.sh

# 단일 rollout replay (ASAP 루트에서)
MUJOCO_GL=egl python humanoidverse/replay_agent.py \
  +simulator=isaacsim \
  +replay_data=motionData/locomotion_run0.pt \
  headless=True save_video=True

# 디렉토리 전체 (locomotion_*.pt 순회, isaacsim 세션 1회 재사용)
MUJOCO_GL=egl python humanoidverse/replay_agent.py \
  +simulator=isaacsim +replay_data=motionData headless=True save_video=True

# 옵션
#   +checkpoint=<path>       config 탐색용 (기본: replay 데이터에 저장된 경로)
#   +init_from_data=False    상태 주입 없이 기본 reset 에서 재생 (초기 상태가 달라짐:
#                            reset 시 dof_pos U(0.5,1.5) 랜덤화가 eval 에서도 걸림)
#   +simulator=mujoco        같은 시뮬레이터 재생 (sanity check; 오차 하한 측정)

# 초기 상태 정렬 원리
#   qpos[k] 는 actions[k] 적용 직후 상태 → qpos[0](위치) + obs[1] 슬라이스(속도:
#   dof_vel, base_ang_vel, critic_obs 의 base_lin_vel; noise_scales 전부 0이라 정확값)
#   로 step-1 상태를 복원해 주입하고 actions[1:] 재생. obs 슬라이스 순서는
#   _post_config_observation_callback 의 sorted() 순서.

# 비교만 다시 실행 (isaacsim 불필요, hvgym 등에서)
python humanoidverse/utils/replay_compare.py \
  motionData/locomotion_run0.pt motionData/locomotion_run0_replay_isaacsim.pt

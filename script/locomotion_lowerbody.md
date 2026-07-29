# KAPEX 하체+허리 RL locomotion (상체는 PD 로 default pose 유지)

전신 31-DoF RL 이 제어가 안 되어, policy 는 하체 17-DoF(다리 14+허리 3)만 출력하고
팔 14-DoF 는 action=0 (→ PD target = default_dof_pos) 으로 고정하는 구성.
추가 파일: agents/decouple/ppo_lowerbody.py, config/algo/ppo_lowerbody.yaml,
config/exp/locomotion_lowerbody.yaml (기존 파이프라인 무수정)

python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion_lowerbody \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_history_wolinvel \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=lowerbody_v1 \
headless=True

# 참고
# - obs 의 'actions' 성분은 여전히 31차원(뒤 14개는 항상 0) — obs config 수정 불필요
# - deploy_agent.py 수집 데이터의 actions 는 17차원으로 저장됨.
#   replay_agent.py 는 자동으로 상체 0 패딩 후 재생 (padding 로그 확인)
# - reward 의 upperbody_joint_angle_freeze 는 이 구성에선 policy 가 제어 불가한 항이므로
#   0 으로 꺼도 됨 (PD 홀드로 편차가 작아 유지해도 무해)

policy deploy :
    `deploy_agent.py`

convert motion data for split:
    `make_delta_a_motion.py`

train delta action policy:
python humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=train_delta_a_open_loop \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/delta_a/reward_delta_a_openloop \
  +robot=kapex/kapex_31dof \
  +terrain=terrain_locomotion_plane \
  +obs=delta_a/open_loop \
  num_envs=4096 \
  project_name=DeltaA_Training \
  experiment_name=kapex_openloop_deltaA \
  robot.motion.motion_file=motionData/delta_a/kapex_walk.pkl \
  env.config.max_episode_length_s=1.0 \
  rewards.reward_scales.penalty_minimal_action_norm=-0.1 \
  env.config.resample_motion_when_training=True \
  env.config.resample_time_interval_s=10000

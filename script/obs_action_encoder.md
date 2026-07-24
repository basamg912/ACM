python humanoidverse/train_agent.py \
  +simulator=isaacsim +exp=locomotion_history_encoder \
  +domain_rand=NO_domain_rand +rewards=loco/reward_kapex_locomotion \
  +robot=kapex/kapex_31dof +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_history_encoder \
  num_envs=4096 algo.config.num_learning_iterations=5 headless=True \
  project_name=smoke experiment_name=hist_enc_smoke

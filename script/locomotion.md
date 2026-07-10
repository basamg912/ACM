python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion \
+obs=loco/leggedloco_obs_history_wolinvel \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=w_h1_reward \
headless=True

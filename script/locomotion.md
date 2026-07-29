python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_history_wolinvel \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=w_h1_reward \
headless=True

# eval
python humanoidverse/eval_agent.py +checkpoint=/home/kist/work/workspace/ACM/ASAP/logs/kapex_locomotion/20260728_125052-w_h1_reward-locomotion-KAPEX_wo_hand_head/model_10000.pt +num_envs=1 +robot.asset.xml_file=kapex/kapex_play.xml +algo.config.eval_command=[0.0,0.0,0.0,0.0]

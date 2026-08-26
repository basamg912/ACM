# Train
python humanoidverse/train_agent.py +simulator=isaacsim +exp=locomotion env=locomotion_cmd_curriculum +domain_rand=NO_domain_rand +rewards=loco/reward_g1_locomotion +robot=g1/g1_29dof_anneal_23dof +terrain=terrain_locomotion_plane +obs=loco/leggedloco_obs_history_wolinvel_wphase num_envs=4096 project_name=G1_cmd_cur experiment_name=g1_reward_setging headless=True

# eval (Kapex)
python humanoidverse/eval_agent.py +checkpoint=/home/kist/work/workspace/ACM/ASAP/logs/kapex_locomotion/20260728_125052-w_h1_reward-locomotion-KAPEX_wo_hand_head/model_10000.pt +num_envs=1 +robot.asset.xml_file=kapex/kapex_play.xml +algo.config.eval_command=[0.0,0.0,0.0,0.0]

# G1 (no cmd cur, base)
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py +simulator=isaacsim +exp=locomotion +domain_rand=NO_domain_rand +rewards=loco/reward_g1_locomotion  +robot=g1/g1_29dof_anneal_23dof +terrain=terrain_locomotion_plane +obs=loco/leggedloco_obs_history_wolinvel num_envs=4096 project_name=G1_reward experiment_name=G123dof_loco headless=True

# G1 (vel cur)
python humanoidverse/train_agent.py +simulator=isaacsim +exp=locomotion env=locomotion_cmd_curriculum +domain_rand=NO_domain_rand +rewards=loco/reward_g1_locomotion +robot=g1/g1_29dof_anneal_23dof +terrain=terrain_locomotion_plane +obs=loco/leggedloco_obs_history_wolinvel num_envs=4096 project_name=G1_loco_cmd_cur experiment_name=G1_loco_cmd_cur_base headless=True

# push eval
python humanoidverse/eval_agent.py +checkpoint=/path/to/model.pt +simulator=mujoco +num_envs=1 keyboard_push.force_newtons=150 keyboard_push.duration_seconds=0.2


# Nomad
nomad job run -var 'extra_args=["env=locomotion_cmd_curriculum", "domain_rand=domain_rand_base", "auto_load_latest=False"]' -var 'disabled_groups=["baseline"]' -var 'wandb_api_key=<api_key>' script/g1_hist_ablation.nomad

extra_args=[..., "++algo.config.detach_encoder_for_policy=False"]

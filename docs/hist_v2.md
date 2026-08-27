# History Encoder V2 (student-teacher concurrent distillation)

구조 (DreamWaQ CENet 분해형):
- Student(배포 사용): [history 495] -> MLP-mixer -> [v_base(3), latent z(16)] -> policy 입력에 concat
- Teacher(학습 전용): [next obs 68] -> VAE -> latent -> decoder -> next obs 복원
- Loss: teacher = recon + beta*KL(free bits) / student = vel MSE(GT base_lin_vel) + latent MSE(teacher mu, detach)
- detach_encoder_for_policy=True: PPO gradient 는 encoder 로 흐르지 않음 (supervised 전용)

추가 파일 (기존 v1 파이프라인 무수정):
- agents/modules/ppo_hist_v2_modules.py (StudentEncoder / TeacherVAE / PPOActorWithStudentEncoder)
- agents/ppo_hist_v2/ppo_hist_v2.py (PPOHistV2) + inference_wrapper.py (ONNX/JIT export 용)
- config/algo/ppo_hist_v2.yaml, config/exp/locomotion_hist_v2.yaml

# 학습 (isaacsim: conda activate hvlab + source _isaac_sim/setup_conda_env.sh)
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion_hist_v2 \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_history_encoder \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=hist_v2 \
headless=True

# 주요 knob (config/algo/ppo_hist_v2.yaml)
#   encoder_config.hidden_dims=[256,128] (feature mixing) / channel_hidden_dims=[64] (channel mixing)
#     — 분리 지정. channel(시계열 5차원) 쪽이 행 75회 반복되는 연산 병목 (2.9M→0.36M MAC)
#   vel_coef=1.0 latent_coef=0.5 latent_coef_warmup_iters=500 (초기 teacher latent 노이즈 회피)
#   recon_coef=1.0 vae_beta=0.01 vae_free_bits=0.1 (teacher)
#   detach_encoder_for_policy=True (False 로 두면 v1 처럼 joint gradient 비교 실험 가능)
#   latent_dim 변경 시 module_dict.actor.input_dim: [actor_obs, vel_dim+latent_dim] 동기 필수

# TB 지표: Loss/vel_est, Loss/latent_match, Loss/teacher_recon, Loss/teacher_kl, Loss/teacher_latent_std
# 체크포인트에 teacher 포함(resume 용) — deploy/eval 은 student 만 사용
# command curriculum 병행: env=locomotion_cmd_curriculum 추가

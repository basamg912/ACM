# History Encoder V3 (V2 + contrastive projection heads)

V2(student-teacher)에 contrastive 헤드를 추가:
- h(z_s): student latent 헤드 (actor 소유), g(z_t): teacher latent 헤드 (teacher 소유)
- contrastive loss = 양방향 InfoNCE(h(z_s), g(t_mu.detach())), cosine/temperature
- 목적: VAE(KL) 압력이 만드는 teacher latent 좌표 노이즈를 MSE 로 쫓지 않고
  배치 negative 대비 상대 유사도(판별 구조)만 정렬
- recon 은 z_t 유지, policy 입력은 z_s 유지, teacher encoder 는 recon 으로만 학습(detach)
- 기본값 latent_coef=0 (v2 MSE 를 contrastive 로 대체; >0 으로 병행 실험 가능)

추가 파일 (v1/v2 무수정):
- agents/modules/ppo_hist_v3_modules.py (heads + info_nce_loss)
- agents/ppo_hist_v3/ppo_hist_v3.py (PPOHistV3, PPOHistV2 상속)
- config/algo/ppo_hist_v3.yaml, config/exp/locomotion_hist_v3.yaml

# 학습 (isaacsim: conda activate hvlab + source _isaac_sim/setup_conda_env.sh)
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion_hist_v3 \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_history_encoder \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=hist_v3 \
headless=True

# 주요 knob
#   contrastive_coef=0.5 contrastive_temperature=0.1 contrastive_batch_size=1024
#   projection_config.proj_dim=16 hidden_dims=[32]
#   latent_coef_warmup_iters=500 (contrastive 에도 동일 warmup 적용)
# TB: Loss/contrastive (random ~ log(batch)≈6.9 에서 감소해야 정렬 학습 중)
# 주의: v2 체크포인트는 head 파라미터가 없어 v3 로 strict load 불가 (새로 학습)
# deploy/eval 은 student 만 사용 — v2 inference_wrapper 그대로 호환

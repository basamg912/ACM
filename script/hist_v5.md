# History Encoder V5 (V3 + teacher 를 context 조건부 CVAE 로)

V3(contrastive student-teacher)에서 teacher 구조만 교체:
1. 입력 = student 와 동일한 window 의 history_t + o_{t+1}
2. context c = MLP-mixer(history_t) — student 인코더와 같은 mixer 구조(가중치는 teacher 소유)
3. encoder(c, o_{t+1}) → (t_mu, t_logvar)
4. decoder(c, t_mu) → o_{t+1} recon — **t_mu 직접 decode** (reparam 샘플이 decode 경로에서
   빠져 teacher forward 는 결정론적, t_logvar 는 KL 로만 학습)
- student 경로 / contrastive(InfoNCE) / vel loss / optimizer 순서 전부 v3 와 동일
- 의미: recon 중 "history 로 예측 가능한 부분"은 c 가 담당 → z_t 는 o_{t+1} 이 새로
  가져오는 **잔차 정보**만 인코딩, student z_s 는 그 잔차 표현에 정렬됨
- 데이터: teacher history = student 가 본 encoder_obs 그대로(자동 저장),
  o_{t+1} = v2 부터 캡처하던 next_obs_target → **rollout/storage/obs 변경 없음**

추가 파일 (v1/v2/v3 무수정):
- agents/modules/ppo_hist_v5_modules.py (TeacherContextCVAEContrastive)
- agents/ppo_hist_v5/ppo_hist_v5.py (PPOHistV5, PPOHistV3 상속)
- config/algo/ppo_hist_v5.yaml, config/exp/locomotion_hist_v5.yaml

# 학습 (isaacsim: conda activate hvlab + source _isaac_sim/setup_conda_env.sh)
python humanoidverse/train_agent.py \
+simulator=isaacsim \
+exp=locomotion_hist_v5 \
+domain_rand=domain_rand_base \
+rewards=loco/reward_kapex_locomotion \
+robot=kapex/kapex_31dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_history_encoder \
num_envs=4096 \
project_name=kapex_locomotion \
experiment_name=hist_v5 \
headless=True

# 주요 knob
#   teacher_config.context_dim=32 — history 압축 차원 (c)
#   teacher_config.mixer_hidden_dims=[256,128] mixer_channel_hidden_dims=[64] — context mixer
#   teacher_config.enc_hidden_dims=[128,64] dec_hidden_dims=[128,256]
#   나머지(contrastive_coef 등)는 v3 와 동일
# TB: v3 와 동일 (Loss/teacher_recon, Loss/teacher_kl, Loss/contrastive, Loss/latent_match, ...)
#   teacher_recon 이 v3 보다 빨리/낮게 떨어지는 게 정상 (c 가 예측 가능분을 흡수)
#   z_t 붕괴 여부는 Loss/teacher_kl 이 free-bits floor(0.1×16=1.6) "아래로" 내려가는지로 감시
# 주의: v3 체크포인트는 teacher 구조가 달라 strict load 불가 (actor/critic 은 구조 동일)
# deploy/eval 은 student 만 사용 — v2 inference_wrapper 그대로 호환

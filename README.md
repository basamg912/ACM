# ACM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Nomad](https://img.shields.io/badge/Nomad-00CA8E?style=flat&logo=nomad&logoColor=white)
![NVIDIA Isaac Lab](https://img.shields.io/badge/NVIDIA%20Isaac%20Lab-76B900?style=flat&logo=nvidia&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-1A1A1A?style=flat)
![Hydra](https://img.shields.io/badge/Hydra-89b8cd?style=flat)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat)
![Weights & Biases](https://img.shields.io/badge/W%26B-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)
![Conda](https://img.shields.io/badge/Conda-44A833?style=flat&logo=anaconda&logoColor=white)

휴머노이드 로봇(KAPEX, Unitree G1) 강화학습 정책 학습을 위한 프로젝트. Sim-to-Real 강건성 향상을 위한 History Encoder 학습 진행.
HumanoidVerse 기반 시뮬레이션 학습 및 평가 프레임워크와 모션 리타겟팅 프레임워크를 서브모듈로 사용.

Docker/Nomad 를 이용해 GPU 클러스터 (Nvidia DGX Spark 및 RTX 5090 Node Pool) 를 구성하고, 병렬 학습을 진행해 학습 효율 높임.


> [!note]
> KAPEX 와 관련된 모델 파일 및 Config 는 보안 상의 이유로 공개하지 않습니다.

## 기여

- **IsaacGym → IsaacLab 포팅** <br>: IsaacGym 기반 HumanoidVerse 를 IsaacLab latest 2.3.0 에서 동작하도록 포팅 및 Reward 재설계
- **MuJoCo Simulator 추가** <br>: (IsaacLab 학습 이후 Sim-to-Sim Target simulator 로 사용)
- **History Encoder 구조를 통한 Sim-to-Real(Sim) 강건성** <br>: End-to-End History Encoder 학습을 통한 Sim-to-Real(Sim) 강건성 확인 (총 5가지 variant)
- **Sim-to-Sim 강건성 실험 설계 및 시각화** <br>: (관측 오염 · 관절 dropout · 간헐적 glitch · 외란 push · input saliency · latent t-SNE)
- **GMR 을 이용한 Motion Retargeting 추가** <br>: (기존 프레임워크 : H2O (Human to Humanoid) )
- **클러스터 운용 편의 기능** <br>: Ablation study 를 위한 Nomad job, 체크포인트 auto-resume


## 구성

- `ASAP` (submodule): IsaacLab 기반 강화학습 프레임워크. Locomotion, motion tracking, history encoder 등 정책 학습을 담당.
- `GMR` (submodule): 사람 모션을 로봇 모션으로 리타겟팅. Motion tracking 학습에 쓰이는 모션 데이터를 생성.
- `motionData`: 리타겟팅된 모션 데이터.
- `script`: 학습 실행 커맨드, Nomad job 정의, 배치 dispatch 스크립트 모음.
- `test`: 테스트 코드. (프레임워크 단위 테스트는 `ASAP/tests` 에 pytest 로 존재)
- `Dockerfile`, `compose.yaml`: 학습 컨테이너 환경 정의.

## 학습 태스크

- Locomotion: 커맨드(속도/방향)를 추종하는 보행 정책 학습.
- Motion Tracking: GMR로 리타겟팅한 모션을 모방하는 정책 학습.
- History Encoder: 아래 별도 절 참고.
- 이 외 delta action, force control, lowerbody decouple 등 실험적 variant들이 함께 관리됨.

## History Encoder

Proprioceptive history 를 이용한 latent 학습. 이를 통해 sim-to-real 전이를 돕는 encoder 실험 계열.
배포/평가 시에는 모든 Variants 가 History Encoder 만 사용하므로 추론 경로는 Baseline 과 동일.

아래와 같이 여러 Variants 로 구성되어 있으며, ablation study 를 통해 각 요소들의 평가를 진행. 

- Baseline: 현재 관측과 Raw History 를 입력으로 받는 정책을 학습.
- V1: History 를 압축하는 Encoder 가 Recon/KL loss 및 PPO loss 로 학습.
- V2: Student-teacher 구조로 분리. 배포용 student encoder는 velocity 추정과 teacher latent 와의 MSE loss 로 학습, teacher는 다음 상태에 대한 Recon/KL loss 로 학습.
- V3: V2에 Latent 를 MSE loss 가 아닌 contrastive loss 를 적용. MSE 의 좌표별 회귀 대신 유사도 기반 Pos/Neg sample 분류를 학습 신호로 사용함으로써, Encoder 의 표현학습에 자유도를 높이고, Collapsing 을 명시적으로 억제.
- V4: Teacher 의 VAE(recon/KL)를 제거하고 privileged obs 를 latent 로 encoding.
  Teacher 학습 신호를 `teacher_mode` 로 선택 (`critic` / `critic_aux` / `frozen` / `vicreg`).
  Latent Collapsing 방지와 무편향 baseline 유지.
- V5: V3의 teacher 를 context 조건부 CVAE 로 교체. `c = mixer(history_t)` 를 조건으로 두어
  history 로 예측 가능한 부분은 c 가 흡수하고, latent 은 `o_{t+1}` 의 잔차 정보만 인코딩 유도.
  Student 경로 / contrastive / obs 구성은 V3 와 동일.

Variant 별 실행 커맨드와 설명은 `script/hist_v2.md`, `hist_v3.md`, `hist_v5.md` 참고.

### Ablation (Nomad)

`script/g1_hist_ablation.nomad` 로 baseline + v1~v5 총 6개 조합을 서로 다른 GPU 노드에
동시 배치해 하나의 `project_name` 아래에서 비교한다.

```bash
export NOMAD_VAR_user=$USER
export NOMAD_VAR_wandb_api_key=<wandb api key>   # 생략 시 TensorBoard only
nomad job run script/g1_hist_ablation.nomad

# 6개 전부에 공통 override / 특정 그룹만 제외
nomad job run -var 'extra_args=["env=locomotion_cmd_curriculum", "domain_rand=domain_rand_base"]' \
              -var 'disabled_groups=["baseline"]' script/g1_hist_ablation.nomad
```

- `auto_load_latest` 기본 활성: 같은 experiment 의 이전 run dir 에서 최신 `model_*.pt` 를 load 해 학습. 학습 config 변경으로 구조가 달라진 경우 run dir 는 통째로 스킵.
- `distinct_hosts` 제약으로 그룹당 노드 1개씩 배치, 가용 노드가 모자라면 pending 대기.

### 강건성 · 해석 실험

History encoder 가 실제로 이득이 있는지 확인하기 위한 평가 스크립트 (`ASAP/humanoidverse`).
결과 플롯과 생성 스크립트는 `ASAP/ckpt/hist_ablation` 에 있다.

| 스크립트 | 설명 |
| --- | --- |
| `eval_obs_corruption.py` | 매 스텝 지속되는 관측 오염(gauss/bias)에 대한 낙상률 |
| `eval_joint_dropout.py` | 관절 단위 센서 고장 (`zero` / `freeze`), dof_pos·dof_vel 채널별 |
| `eval_intermittent_noise.py` | 간헐적 glitch — duty(오염 비율)와 burst(연속 길이)를 분리해 비교 |
| `eval_push_robustness.py` | Robot base 에 외란을 주입 → P(fall \| push) |
| `collect_obs_stats.py` | 정책이 실제로 보는 obs 기록 → 차원별 표준편차 · on-policy Jacobian |
| `collect_encoder_latents.py` | 커맨드별 encoder latent/vel head 기록 (t-SNE 군집 분석용) |
| `scripts/plot_latent_tsne.py` | 수집한 latent 의 t-SNE 시각화 |

관측 오염은 학습 과정에서 사용되는 관측 노이즈 경로(`helpers.parse_observation`) 를 그대로 사용해
현재 obs 와 history 모두 학습 환경과 분리된 순수 센서 결함에 의한 노이즈성 신호가 전파.

## 학습 설정

- **Observation**: `leggedloco_obs_*` 계열로 관리. `wolinvel`(base lin vel 제외), `history_encoder`
  (encoder_obs/recon_target 그룹 추가), `_v4`(privileged teacher_obs 그룹), `_wphase`
  (gait 위상 `cos_phase`/`sin_phase` 포함) 조합.
- **Command curriculum** (`env=locomotion_cmd_curriculum`): `tracking_lin_vel` 보상이 임계값을
  넘으면 lin_vel x/y 커맨드 범위를 단계적으로 확장하는 Unitree RL Lab 방식 구현.
- **Reward curriculum**: penalty scale 및 dof_pos/dof_vel/torque soft limit 을 학습 진행에 따라
  level up/down. 기존 ASAP 의 IsaacGym 학습에서 사용하던 방식.
- **Domain randomization** (`domain_rand=domain_rand_base`): reset 자세/루트 상태, push,
  base CoM, link mass, PD gain, friction, RFI torque, control delay 랜덤화.
- **Logging**: TensorBoard 기본, `+opt=wandb` 로 W&B 병행 (`wandb_api_key` 전달 시 Nomad job 이 자동 부착).

## 실행 방식

- 로컬: `docker compose build` / `docker compose up` 으로 컨테이너에 진입해 학습 스크립트를 직접 실행. 커맨드 예시는 `script/*.md`, `script/*.sh` 참고.
- 클러스터: `script/*.nomad` job spec을 통해 GPU 노드(dgx-spark, rtx-gpu 등)에 학습/모션 리타겟팅 job을 등록·디스패치.

## 지원 로봇

- G1 (29dof / anneal 23dof)
- ~~KAPEX(31dof)~~

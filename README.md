# ACM

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Nomad](https://img.shields.io/badge/Nomad-00CA8E?style=flat&logo=nomad&logoColor=white)
![NVIDIA Isaac Lab](https://img.shields.io/badge/NVIDIA%20Isaac%20Lab-76B900?style=flat&logo=nvidia&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-1A1A1A?style=flat)
![Hydra](https://img.shields.io/badge/Hydra-89b8cd?style=flat)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat)
![Conda](https://img.shields.io/badge/Conda-44A833?style=flat&logo=anaconda&logoColor=white)

휴머노이드 로봇(KAPEX, G1) 강화학습 정책 학습을 위한 프로젝트.
HumanoidVerse 기반 시뮬레이션 학습 프레임워크와 모션 리타겟팅 프레임워크를 서브모듈로 사용.
학습 시 Docker/Nomad 로 GPU 클러스터 (Spark 6대 및 RTX 5090 pool) 에서 병렬 학습 진행.

> [!note]
> KAPEX 와 관련된 모델 파일 및 Config 는 보안 상의 이유로 공개하지 않습니다.

## 기여

- IsaacGym 학습 프레임워크를 IsaacLab latest 5.1.0 동작하도록 포팅
- MuJoCo Simulator 추가 (IsaacLab 학습 이후 Sim-to-Sim Target simulator 로 사용)
- End-to-End History Encoder 학습을 통한 Sim-to-Real(Sim) 강건성 확인 (총 3가지 variant)
- Unitree G1 및 Kapex 에서 사용 가능한 Locomotion Reward 설계
- GMR 을 이용한 Motion Retargeting 추가 (기존 프레임워크 : H2O (Human to Humanoid) )
- Only Lowerbody decoupled RL 환경 구성 (Loco-manipulation 을 위한 설계)


## 구성

- `ASAP` (submodule): IsaacLab 기반 강화학습 프레임워크. Locomotion, motion tracking, history encoder 등 정책 학습을 담당.
- `GMR` (submodule): 사람 모션을 로봇 모션으로 리타겟팅. Motion tracking 학습에 쓰이는 모션 데이터를 생성.
- `motionData`: 리타겟팅된 모션 데이터.
- `script`: 학습 실행 커맨드, Nomad job 정의, 배치 dispatch 스크립트 모음.
- `test`: 테스트 코드.
- `Dockerfile`, `compose.yaml`: 학습 컨테이너 환경 정의.

## 학습 태스크

- Locomotion: 커맨드(속도/방향)를 추종하는 보행 정책 학습.
- Motion Tracking: GMR로 리타겟팅한 모션을 모방하는 정책 학습.
- History Encoder: 아래 별도 절 참고.
- 이 외 delta action, force control, lowerbody decouple 등 실험적 variant들이 함께 관리됨.

## History Encoder

Proprioceptive history 를 이용한 latent 학습. 이를 통해 sim-to-real 전이를 돕는 encoder 실험 계열.

- Baseline: History encoder 없이 현재 관측만으로 정책을 학습.
- V1: Encoder를 정책과 하나로 묶어 end-to-end로 함께 학습. Recon/KL loss가 정책 loss와 같은 optimizer로 흐름.
- V2: Student-teacher 구조로 분리. 배포용 student encoder는 velocity 추정과 teacher latent 정렬(MSE)로 지도학습되고, teacher는 다음 상태 복원(VAE)으로 별도 학습.
- V3: V2에 contrastive(InfoNCE) latent 정렬을 추가. Teacher-student latent를 좌표별 회귀 대신 판별적 유사도로 정렬해 sim-to-real 전이 강건성을 높이는 실험.

## 실행 방식

- 로컬: `docker compose build` / `docker compose up` 으로 컨테이너에 진입해 학습 스크립트를 직접 실행. 커맨드 예시는 `script/*.md`, `script/*.sh` 참고.
- 클러스터: `script/*.nomad` job spec을 통해 GPU 노드(dgx-spark, rtx-gpu 등)에 학습/모션 리타겟팅 job을 등록·디스패치.

## 대상 로봇

- KAPEX (31dof)
- G1 (29dof / anneal 23dof)

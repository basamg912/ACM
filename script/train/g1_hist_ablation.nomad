// Nomad job: G1 reward 기준 history encoder ablation (6개 조합 동시 실행)
//   baseline : PPO                      (+exp=locomotion,                 obs=wolinvel)
//   hist-v1  : VAE history encoder      (+exp=locomotion_history_encoder, obs=history_encoder)
//   hist-v2  : student-teacher distill  (+exp=locomotion_hist_v2,         obs=history_encoder)
//   hist-v3  : v2 + contrastive heads   (+exp=locomotion_hist_v3,         obs=history_encoder)
//   hist-v4  : task critic teacher + PPO/velocity/contrastive student
//                                       (+exp=locomotion_hist_v4,         obs=history_encoder_v4)
//   hist-v5  : v3 + teacher 를 context 조건부 CVAE 로 — c=mixer(history_t),
//              encoder(c, o_{t+1}), decoder(c, t_mu)→o_{t+1} recon
//                                       (+exp=locomotion_hist_v5,         obs=history_encoder)
//
// 사용법:
//   export NOMAD_VAR_user=$USER
//   nomad job run script/g1_hist_ablation.nomad
//   nomad job run -var node_pool=rtx-gpu -var image=161.122.114.87:5000/hvlab:v0-amd64 \
//     script/g1_hist_ablation.nomad
//   // 6개 전부에 공통 override 추가 (예: command curriculum):
//   nomad job run -var 'extra_args=["env=locomotion_cmd_curriculum"]' script/g1_hist_ablation.nomad
//   // wandb 로깅: 키를 넘기면 6개 전부에 +opt=wandb 가 자동으로 붙는다 (미지정 시 TB only)
//   export NOMAD_VAR_wandb_api_key=<https://wandb.ai/authorize 에서 발급>
//   nomad job run script/g1_hist_ablation.nomad
//   // 체크포인트 자동 승계: 모든 그룹에 auto_load_latest=True 기본 — 같은 experiment
//   //   (project/experiment_name/robot 동일)의 이전 run dir 에서 최신 model_*.pt 를 찾아
//   //   이어 학습한다 (노드 전원 차단 후 재시작 대비). 리워드/설정을 바꿔 처음부터
//   //   돌리려면 experiment_name 을 바꾸거나:
//   nomad job run -var 'extra_args=["auto_load_latest=False"]' script/g1_hist_ablation.nomad
//
// 비교 설계 노트:
//   - project_name 을 하나(G1_hist_ablation)로 묶어 TB 에서 6개 run 을 한 화면에 비교
//   - seed 는 base config 기본값(0)으로 6개 동일 — 변경 시 extra_args 로 일괄 적용
//   - baseline 만 obs 가 다름(wolinvel). v1/v2/v3/v5 의 actor_obs 구성은 wolinvel 과 동일하고
//     encoder_obs/recon_target 그룹만 추가됨 (obs dims 는 robot.dof_obs_size 기반이라 G1 23dof 자동 대응)
//   - v4 는 obs 파일이 다름(history_encoder_v4): critic history를 제거하고 teacher_obs에
//     privileged(base_pos_z, feet_contact_force)를 포함한다. actor_obs/encoder_obs는 동일.
//   - v5 는 v2/v3 와 obs 동일(history_encoder) — teacher 가 encoder_obs+next_obs_target 재사용,
//     rollout/storage/obs 추가 없음.
//   - 노드당 GPU 1장 전제 → distinct_hosts 로 6개 그룹을 서로 다른 노드에 배치.
//     가용 노드가 6개 미만이면 남는 그룹은 pending 으로 대기.

variable "acm_root" {
  type    = string
  default = "/home/sybae/work/ACM"
}

// dgx-spark(ARM64) → hvlab:v0,  rtx-gpu(x86_64) → hvlab:v0-amd64
variable "image" {
  type    = string
  default = "161.122.114.87:5000/hvlab:v0"
}

variable "node_pool" {
  type    = string
  default = "dgx-spark"
}

variable "user" {
  type        = string
  default     = ""
  description = "LDAP uid — NOMAD_VAR_user=$USER 로 export 하거나 -var=user=... 로 전달"
  validation {
    condition     = var.user != ""
    error_message = "Variable 'user' is required. Run: export NOMAD_VAR_user=$USER (or pass -var=user=<id>)."
  }
}

// 6개 그룹 전부에 덧붙일 공통 hydra override (예: ["env=locomotion_cmd_curriculum", "seed=1"])
variable "extra_args" {
  type    = list(string)
  default = []
}

// 제출 시점에 특정 그룹만 비활성화 (count=0 처리, 주석처리 불필요):
//   nomad job run -var 'disabled_groups=["hist-v1","hist-v4"]' script/g1_hist_ablation.nomad
// 이미 돌고 있는 잡은 파일 수정 없이: nomad job scale g1-hist-ablation <group> 0
variable "disabled_groups" {
  type    = list(string)
  default = []
}

// wandb 로깅. 키가 있으면 6개 전부 +opt=wandb 자동 활성화 — train_agent.py 가
// sync_tensorboard=True 로 init 하므로 TB 스칼라가 그대로 wandb 에 미러링된다.
// 키는 git 에 남지 않게 파일에 쓰지 말고 NOMAD_VAR_wandb_api_key 로만 전달할 것.
variable "wandb_api_key" {
  type        = string
  default     = ""
  description = "W&B API key — export NOMAD_VAR_wandb_api_key=... (비우면 wandb off)"
}

// 개인 계정이면 비워두기(null → API key 의 기본 entity 로 업로드).
// 기본 config(opt/wandb.yaml)의 entity 는 ASAP 원저자 팀(agileh2o)이라 그대로 쓰면 403.
variable "wandb_entity" {
  type    = string
  default = ""
}

locals {
  wandb_args = var.wandb_api_key == "" ? [] : [
    "+opt=wandb",
    "wandb.wandb_entity=${var.wandb_entity == "" ? "null" : var.wandb_entity}",
  ]
  // 이미지(hvlab:v0)의 wandb 0.24.0 은 TB sync 가 tfevents ~1MB 지점에서 영구 정지하는
  // 회귀가 있어 (0.25.0 에서 수정, wandb/wandb#11334) 학습 전에 업그레이드한다.
  // pip 실패(오프라인 등) 시에도 경고만 남기고 학습은 계속 진행.
  train_prefix = "pip install -qU 'wandb>=0.28' || echo '[WARN] wandb upgrade failed; charts may freeze after ~30min (wandb 0.24.0 TB-sync bug)'; exec python humanoidverse/train_agent.py"
}

job "g1-hist-ablation" {
  datacenters = ["dc1"]
  node_pool   = var.node_pool
  type        = "batch"
  namespace   = var.user

  constraint {
    operator = "distinct_hosts"
    value    = "true"
  }

  // ---------------------------------------------------------------
  // 1/6 baseline: plain PPO + wolinvel obs
  // ---------------------------------------------------------------
  group "baseline" {
    count = contains(var.disabled_groups, "baseline") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_wolinvel_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=baseline",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          // Isaac 캐시는 그룹별 분리 (동시 실행 시 쓰기 충돌 방지)
          "isaac-cache-kit-g1bl:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1bl:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1bl:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1bl:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1bl:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1bl:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1bl:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 2/6 hist-v1: concurrent VAE history encoder (joint grad)
  // ---------------------------------------------------------------
  group "hist-v1" {
    count = contains(var.disabled_groups, "hist-v1") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion_history_encoder",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v1",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-g1hv1:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1hv1:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1hv1:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1hv1:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1hv1:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1hv1:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1hv1:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 3/6 hist-v2: student(v,z)-teacher(next-obs VAE) distillation
  // ---------------------------------------------------------------
  group "hist-v2" {
    count = contains(var.disabled_groups, "hist-v2") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v2",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v2",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-g1hv2:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1hv2:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1hv2:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1hv2:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1hv2:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1hv2:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1hv2:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 4/6 hist-v3: v2 + contrastive projection heads (InfoNCE)
  // ---------------------------------------------------------------
  group "hist-v3" {
    count = contains(var.disabled_groups, "hist-v3") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v3",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v3",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-g1hv3:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1hv3:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1hv3:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1hv3:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1hv3:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1hv3:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1hv3:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 5/6 hist-v4: task critic teacher + PPO/velocity/contrastive student
  //   critic은 current state + teacher latent만 사용 (history는 student 전용)
  // ---------------------------------------------------------------
  group "hist-v4" {
    count = contains(var.disabled_groups, "hist-v4") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v4",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder_v4_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v4_task_contrastive",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-g1hv4:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1hv4:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1hv4:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1hv4:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1hv4:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1hv4:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1hv4:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 6/6 hist-v5: v3 + teacher 를 context 조건부 CVAE 로
  //   c=mixer(history_t, student 와 동일 window), encoder(c, o_{t+1}),
  //   decoder(c, t_mu)→o_{t+1} recon. obs 는 v2/v3 와 동일(history_encoder).
  // ---------------------------------------------------------------
  group "hist-v5" {
    count = contains(var.disabled_groups, "hist-v5") ? 0 : 1
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
	image_pull_timeout = "30m"
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "bash"
        args = ["-c", join(" ", concat([
          local.train_prefix,
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v5",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder_wphase",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v5",
          "headless=True",
          "auto_load_latest=True",
        ], local.wandb_args, var.extra_args))]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-g1hv5:/isaac-sim/kit/cache",
          "isaac-cache-ov-g1hv5:/home/hvlab/.cache/ov",
          "isaac-cache-pip-g1hv5:/home/hvlab/.cache/pip",
          "isaac-cache-gl-g1hv5:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-g1hv5:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-g1hv5:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-g1hv5:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
        WANDB_API_KEY              = var.wandb_api_key
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }
}

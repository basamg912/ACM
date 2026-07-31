// Nomad job: G1 reward 기준 history encoder ablation (4개 조합 동시 실행)
//   baseline : PPO                      (+exp=locomotion,                 obs=wolinvel)
//   hist-v1  : VAE history encoder      (+exp=locomotion_history_encoder, obs=history_encoder)
//   hist-v2  : student-teacher distill  (+exp=locomotion_hist_v2,         obs=history_encoder)
//   hist-v3  : v2 + contrastive heads   (+exp=locomotion_hist_v3,         obs=history_encoder)
//
// 사용법:
//   export NOMAD_VAR_user=$USER
//   nomad job run script/g1_hist_ablation.nomad
//   nomad job run -var node_pool=rtx-gpu -var image=161.122.114.87:5000/hvlab:v0-amd64 \
//     script/g1_hist_ablation.nomad
//   // 4개 전부에 공통 override 추가 (예: command curriculum):
//   nomad job run -var 'extra_args=["env=locomotion_cmd_curriculum"]' script/g1_hist_ablation.nomad
//
// 비교 설계 노트:
//   - project_name 을 하나(G1_hist_ablation)로 묶어 TB 에서 4개 run 을 한 화면에 비교
//   - seed 는 base config 기본값(0)으로 4개 동일 — 변경 시 extra_args 로 일괄 적용
//   - baseline 만 obs 가 다름(wolinvel). v1/v2/v3 의 actor_obs 구성은 wolinvel 과 동일하고
//     encoder_obs/recon_target 그룹만 추가됨 (obs dims 는 robot.dof_obs_size 기반이라 G1 23dof 자동 대응)
//   - 노드당 GPU 1장 전제 → distinct_hosts 로 4개 그룹을 서로 다른 노드에 배치.
//     가용 노드가 4개 미만이면 남는 그룹은 pending 으로 대기.

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

// 4개 그룹 전부에 덧붙일 공통 hydra override (예: ["env=locomotion_cmd_curriculum", "seed=1"])
variable "extra_args" {
  type    = list(string)
  default = []
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
  // 1/4 baseline: plain PPO + wolinvel obs
  // ---------------------------------------------------------------
  group "baseline" {
    count = 1
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

        command = "python"
        args = concat([
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=locomotion",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_wolinvel",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=baseline",
          "headless=True",
        ], var.extra_args)

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
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 2/4 hist-v1: concurrent VAE history encoder (joint grad)
  // ---------------------------------------------------------------
  group "hist-v1" {
    count = 1
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

        command = "python"
        args = concat([
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=locomotion_history_encoder",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v1",
          "headless=True",
        ], var.extra_args)

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
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 3/4 hist-v2: student(v,z)-teacher(next-obs VAE) distillation
  // ---------------------------------------------------------------
  group "hist-v2" {
    count = 1
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

        command = "python"
        args = concat([
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v2",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v2",
          "headless=True",
        ], var.extra_args)

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
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }

  // ---------------------------------------------------------------
  // 4/4 hist-v3: v2 + contrastive projection heads (InfoNCE)
  // ---------------------------------------------------------------
  group "hist-v3" {
    count = 1
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

        command = "python"
        args = concat([
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=locomotion_hist_v3",
          "+domain_rand=NO_domain_rand",
          "+rewards=loco/reward_g1_locomotion",
          "+robot=g1/g1_29dof_anneal_23dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=loco/leggedloco_obs_history_encoder",
          "num_envs=4096",
          "project_name=G1_hist_ablation",
          "experiment_name=hist_v3",
          "headless=True",
        ], var.extra_args)

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
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }
}

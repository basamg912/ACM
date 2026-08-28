// Nomad job: locomotion + motion tracking 학습을 동시에 실행
// 사용법:
//   nomad job run script/train/train_jobs.nomad
//   nomad job run -var node_pool=rtx-gpu -var image=161.122.114.87:5000/hvlab:v0-amd64 script/train/train_jobs.nomad
// 전제:
//   - 이미지 hvlab:v0 가 노드에 빌드되어 있음 (docker compose build)
//   - Nomad docker 드라이버에서 volumes 허용 필요 (client 설정):
//       plugin "docker" { config { volumes { enabled = true } } }
//   - docker nvidia runtime 사용 (device plugin 불필요)

variable "acm_root" {
  type    = string
  default = "/home/sybae/work/ACM"
}

// 노드 아키텍처에 맞는 이미지를 골라야 함:
//   dgx-spark(ARM64) → hvlab:v0,  rtx-gpu(x86_64) → hvlab:v0-amd64
variable "image" {
  type    = string
  default = "161.122.114.87:5000/hvlab:v0"
}

variable "node_pool" {
  type    = string
  default = "dgx-spark"
}

job "kapex-train" {
  datacenters = ["dc1"]
  node_pool   = var.node_pool
  type        = "batch"

  // DGX Spark는 노드당 GPU 1장(GB10)이므로 두 그룹을 서로 다른 노드에 배치
  constraint {
    operator = "distinct_hosts"
    value    = "true"
  }

  // ---------------------------------------------------------------
  // Job 1: locomotion (docs/locomotion.md)
  // ---------------------------------------------------------------
  group "locomotion" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }

    task "train" {
      driver = "docker"
      // 호스트 디렉토리(UID 1000009 소유)에 로그를 쓰기 위해 root로 실행
      user   = "root"

      config {
        image        = var.image
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "python"
        args = [
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=locomotion",
          "+domain_rand=domain_rand_base",
          "+rewards=loco/reward_kapex_locomotion",
          "+robot=kapex/kapex_31dof",
          "+terrain=terrain_locomotion",
          "+obs=loco/leggedloco_obs_history_wolinvel",
          "num_envs=4096",
          "project_name=kapex_locomotion",
          "experiment_name=w_h1_reward",
          "headless=True",
        ]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          // Isaac 캐시는 그룹별로 분리 (동시 실행 시 쓰기 충돌 방지)
          "isaac-cache-kit-loco:/isaac-sim/kit/cache",
          "isaac-cache-ov-loco:/home/hvlab/.cache/ov",
          "isaac-cache-pip-loco:/home/hvlab/.cache/pip",
          "isaac-cache-gl-loco:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-loco:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-loco:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-loco:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA             = "Y"
        PRIVACY_CONSENT         = "Y"
        NVIDIA_VISIBLE_DEVICES  = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
      }

      resources {
        cpu    = 8000  # MHz
        memory = 16000 # MB
      }
    }
  }

  // ---------------------------------------------------------------
  // Job 2: motion tracking (docs/motion_tracking.md)
  // ---------------------------------------------------------------
  group "motion-tracking" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }

    task "train" {
      driver = "docker"
      // 호스트 디렉토리(UID 1000009 소유)에 로그를 쓰기 위해 root로 실행
      user   = "root"

      config {
        image        = var.image
        runtime      = "nvidia"
        network_mode = "host"
        work_dir     = "/workspace/ASAP"

        command = "python"
        args = [
          "humanoidverse/train_agent.py",
          "+simulator=isaacsim",
          "+exp=motion_tracking",
          "+domain_rand=NO_domain_rand",
          "+rewards=motion_tracking/reward_motion_tracking_dm_2real",
          "+robot=kapex/kapex_31dof",
          "+terrain=terrain_locomotion_plane",
          "+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history",
          "num_envs=4096",
          "project_name=MotionTracking",
          "experiment_name=MotionTracking_Kapex_walking",
          "headless=True",
          "rewards.reward_penalty_curriculum=True",
          "rewards.reward_penalty_degree=0.00001",
          "env.config.resample_motion_when_training=False",
          "env.config.termination.terminate_when_motion_far=True",
          "env.config.termination_curriculum.terminate_when_motion_far_curriculum=True",
          "env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3",
          "env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025",
          "robot.asset.self_collisions=0",
        ]

        volumes = [
          "${var.acm_root}/ASAP:/workspace/ASAP",
          "${var.acm_root}/GMR:/workspace/GMR",
          "${var.acm_root}/motionData:/workspace/motionData",
          "${var.acm_root}/script:/workspace/script",
          "${var.acm_root}/logs:/workspace/logs",
          "isaac-cache-kit-mt:/isaac-sim/kit/cache",
          "isaac-cache-ov-mt:/home/hvlab/.cache/ov",
          "isaac-cache-pip-mt:/home/hvlab/.cache/pip",
          "isaac-cache-gl-mt:/home/hvlab/.cache/nvidia/GLCache",
          "isaac-cache-compute-mt:/home/hvlab/.nv/ComputeCache",
          "isaac-logs-mt:/home/hvlab/.nvidia-omniverse/logs",
          "isaac-data-mt:/home/hvlab/.local/share/ov/data",
        ]
      }

      env {
        ACCEPT_EULA             = "Y"
        PRIVACY_CONSENT         = "Y"
        NVIDIA_VISIBLE_DEVICES  = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
      }

      resources {
        cpu    = 8000
        memory = 16000
      }
    }
  }
}

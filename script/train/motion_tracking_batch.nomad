// Parameterized job: 모션 하나당 policy 하나 학습
// 등록:   nomad job run script/train/motion_tracking_batch.nomad
// 실행:   nomad job dispatch -meta motion=0-07_01_stageii kapex-mt-batch
//        (일괄 실행은 script/train/dispatch_motions.sh 참고)
//
// 노드당 GPU 1장이므로 static port(gpu_lock) 예약으로 노드당 학습 1개만 뜨게 함.
// 남는 dispatch 는 자동으로 대기했다가 자리가 나면 실행됨.

variable "acm_root" {
  type    = string
  default = "/home/sybae/work/ACM"
}

variable "image" {
  type    = string
  default = "161.122.114.87:5000/hvlab:v0"
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
job "kapex-mt-batch" {
  datacenters = ["dc1"]
  node_pool   = "dgx-spark"
  type        = "batch"
  namespace = var.user
  parameterized {
    meta_required = ["motion"] // 확장자(.pkl) 제외한 모션 파일 이름
  }

  group "train" {
    count = 1

    // "GPU 잠금" 용 static port: 같은 포트는 노드당 하나만 예약 가능하므로
    // 이 job 의 dispatch 들끼리는 노드당 1개만 배치됨.
    // (distinct_hosts 는 dispatch 된 자식 job 간에는 적용되지 않아 사용 불가.
    //  이 포트를 쓰지 않는 다른 사용자의 job 은 같은 노드에 들어올 수 있음.)
    network {
      port "gpu_lock" {
        static = 28877
      }
    }

    restart {
      attempts = 0
      mode     = "fail"
    }

    // 노드 재부팅 등으로 잃은 alloc 을 다른 노드에 다시 스케줄하지 않음 (중복 학습 방지)
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "train" {
      driver = "docker"
      user   = "root"

      config {
        image        = var.image
        // 이미지가 커서(23GB) 첫 pull 시 기본 5분 제한을 초과할 수 있음
        image_pull_timeout = "30m"
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
          "experiment_name=MT_${NOMAD_META_motion}",
          "headless=True",
          "robot.motion.motion_file=/workspace/motionData/${NOMAD_META_motion}.pkl",
          "rewards.reward_penalty_curriculum=True",
          "rewards.reward_penalty_degree=0.00001",
          "env.config.resample_motion_when_training=False",
          "env.config.termination.terminate_when_motion_far=True",
          "env.config.termination_curriculum.terminate_when_motion_far_curriculum=True",
          "env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3",
          "env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025",
          "robot.asset.self_collisions=0",
          "algo.config.num_learning_iterations=100000",
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
        ACCEPT_EULA                = "Y"
        PRIVACY_CONSENT            = "Y"
        NVIDIA_VISIBLE_DEVICES     = "all"
        NVIDIA_DRIVER_CAPABILITIES = "all"
      }

      resources {
        cpu    = 8000
        // docker driver 는 이 값을 컨테이너 memory hard limit 으로도 사용하므로
        // 실제 사용량보다 여유있게. 노드당 1개 강제는 위 gpu_lock 포트가 담당.
        memory = 30000
      }
    }
  }
}

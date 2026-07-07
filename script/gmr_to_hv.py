"""GMR 출력 pkl → ASAP 모션 pkl 변환. hvgym 환경에서 실행할 것."""

import glob
import os
import sys

import numpy

# numpy 2.x로 저장된 pickle을 numpy 1.x에서 열기 위한 shim
sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
sys.modules["numpy._core.numeric"] = numpy.core.numeric
sys.modules["numpy._core.umath"] = numpy.core.umath
import pickle

import joblib


def convert(gmr_pkl_path, out_dir):
    with open(gmr_pkl_path, "rb") as f:
        d = pickle.load(f)
    key = "0-" + os.path.splitext(os.path.basename(gmr_pkl_path))[0]
    asap_data = {
        key: {
            "root_trans_offset": numpy.ascontiguousarray(d["root_pos"]),
            "root_rot": numpy.ascontiguousarray(d["root_rot"]),  # xyzw 그대로
            "dof": numpy.ascontiguousarray(d["dof_pos"]),  # 순서 1:1 일치 확인됨
            "fps": int(d["fps"]),
        }
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, key + ".pkl")
    joblib.dump(asap_data, out_path)  # numpy 1.x로 재저장 → ASAP 어디서든 로드 가능
    print(f"saved: {out_path}")


if __name__ == "__main__":
    out_dir = "humanoidverse/data/motions/kapex_31dof/GMR/singles"
    for p in sys.argv[1:] or glob.glob(
        "/home/kist/work/workspace/ASAP_isaaclab/GMR/result/**/*.pkl", recursive=True
    ):
        convert(p, out_dir)

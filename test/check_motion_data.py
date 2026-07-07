import os
import pickle
from pathlib import Path


def check_motion_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    data = dict(data)
    # keys -> ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
    print(data.keys())


ASSET_PATH = (
    Path(__file__).parent.parent
    / "GMR"
    / "result"
    / "Female1Running_c3d"
    / "C2_-_Run_to_stand_stageii.pkl"
)
if __name__ == "__main__":
    file_path = str(ASSET_PATH)
    check_motion_data(file_path)

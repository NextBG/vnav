# Cut the trajectories into smaller pieces of TRAJ_LEN frames.
import os
import shutil
from tqdm import tqdm

INPUT_TRAJ_ROOT_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/tokyotownwalk_shinjukupark/temp"
OUTPUT_TRAJ_ROOT_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/tokyotownwalk_shinjukupark/trajectories"

TRAJ_LEN = 3000

for input_traj_name in sorted(os.listdir(INPUT_TRAJ_ROOT_PATH)):
    input_traj_path = os.path.join(INPUT_TRAJ_ROOT_PATH, input_traj_name)
    # frame count
    frame_count = len(os.listdir(os.path.join(input_traj_path, "frames")))
    frame_count_to_cut = frame_count - frame_count % TRAJ_LEN

    for traj_index in range(0, frame_count_to_cut, TRAJ_LEN):
        # output traj path
        output_traj_path = os.path.join(OUTPUT_TRAJ_ROOT_PATH, f"{input_traj_name}_{traj_index//TRAJ_LEN:06d}", "frames")

        # skip if already exists and there are TRAJ_LEN frames in the folder
        if os.path.exists(output_traj_path):
            if len(os.listdir(output_traj_path)) == TRAJ_LEN:
                continue
            else:
                shutil.rmtree(output_traj_path)

        # create output traj path
        os.makedirs(output_traj_path)

        # copy frames
        for frame_index in tqdm(range(traj_index, traj_index + TRAJ_LEN), desc=f"{input_traj_name}_{traj_index//TRAJ_LEN:06d}"):
            frame_path = os.path.join(input_traj_path, "frames", f"{frame_index:06d}.jpg")
            output_frame_path = os.path.join(output_traj_path, f"{(frame_index%TRAJ_LEN):06d}.jpg")
            shutil.move(frame_path, output_frame_path)
# Count the number of files in frames directory of every sub-directory of a given path

import os
import time

DATASET_ROOT_PATH = "mount_point/datasets"
DATASET_NAME = "Tokyo 東京都/trajectories"

dataset_path = os.path.join(DATASET_ROOT_PATH, DATASET_NAME)

while True:
    print("-------------------------------------")
    total = 0
    for video_name in sorted(os.listdir(dataset_path))[:]:
        video_path = os.path.join(dataset_path, video_name)
        frames_dir_path = os.path.join(video_path, "frames")
        if os.path.exists(frames_dir_path):
            print(f"{video_name}: {len(os.listdir(frames_dir_path))}")
            total += len(os.listdir(frames_dir_path))
    print(f"Total: {total}")
    # delay 1s
    time.sleep(2)
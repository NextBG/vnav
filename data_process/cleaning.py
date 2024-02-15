import cv2
import os
from tqdm import tqdm
import multiprocessing

# Desolve videos into frames
# IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/videos"
IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/trajectories"

for traj_folder in sorted(os.listdir(IN_PATH)):
    # traj_est_file = os.path.join(IN_PATH, traj_folder, "traj_est.pkl")
    traj_processed_file = os.path.join(IN_PATH, traj_folder, "traj_processed.pkl")
    traj_vis_file = os.path.join(IN_PATH, traj_folder, "traj_vis.png")

    # Delete files if they exist
    # if os.path.exists(traj_est_file):
    #     os.remove(traj_est_file)
    if os.path.exists(traj_processed_file):
        os.remove(traj_processed_file)
    if os.path.exists(traj_vis_file):
        os.remove(traj_vis_file)
    print(f"Deleted traj files in {traj_folder}")
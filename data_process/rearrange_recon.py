import os
import shutil

IN_DIR = "/home/caoruixiang/datasets_mnt/vnav_datasets/recon"

# Create a new folder called trajectories
if not os.path.exists(os.path.join(IN_DIR, "trajectories")):
    os.makedirs(os.path.join(IN_DIR, "trajectories"))

for traj_name in os.listdir(IN_DIR):
    # Skip the trajectories folder
    if traj_name == "trajectories":
        continue

    # Move the trajectory folder to the trajectories folder
    if os.path.isdir(os.path.join(IN_DIR, traj_name)):
        shutil.move(os.path.join(IN_DIR, traj_name), os.path.join(IN_DIR, "trajectories", traj_name))

    # Rename the traj_data.pkl file to traj_processed.pkl
    if os.path.exists(os.path.join(IN_DIR, "trajectories", traj_name, "traj_data.pkl")):
        shutil.move(os.path.join(IN_DIR, "trajectories", traj_name, "traj_data.pkl"), os.path.join(IN_DIR, "trajectories", traj_name, "traj_processed.pkl"))

    # Move all images in the trajectory folder to the images folder named "frames"
    if os.path.isdir(os.path.join(IN_DIR, "trajectories", traj_name)):
        if not os.path.exists(os.path.join(IN_DIR, "trajectories", traj_name, "frames")):
            os.makedirs(os.path.join(IN_DIR, "trajectories", traj_name, "frames"))
        for img_name in os.listdir(os.path.join(IN_DIR, "trajectories", traj_name)):
            if img_name.endswith(".jpg"):
                shutil.move(os.path.join(IN_DIR, "trajectories", traj_name, img_name), os.path.join(IN_DIR, "trajectories", traj_name, "frames", img_name))


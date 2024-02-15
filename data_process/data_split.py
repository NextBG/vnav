import argparse
import os
import random

DATASET_NAME = "scalenet"

DATASETS_PATH = f"/home/caoruixiang/datasets_mnt/vnav_datasets/{DATASET_NAME}/trajectories"
DATA_SPLITS_PATH = f"/home/caoruixiang/datasets_mnt/vnav_datasets/{DATASET_NAME}/partitions"

def main(args: argparse.Namespace):
    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    traj_folder_names = [
        f for f in sorted(os.listdir(DATASETS_PATH))
        if os.path.isdir(os.path.join(DATASETS_PATH, f))
    ]

    # Randomly shuffle the names of the folders
    random.shuffle(traj_folder_names)

    # Split the names of the folders into train and evaluate sets
    split_index = int(args.split_ratio * len(traj_folder_names))
    train_traj_folder_names = traj_folder_names[:split_index]
    eval_traj_folder_names = traj_folder_names[split_index:]

    # Create th partition folder if it doesn't exist
    if not os.path.exists(DATA_SPLITS_PATH):
        os.makedirs(DATA_SPLITS_PATH)

    # Write the names of the train and evaluate folders to files
    with open(os.path.join(DATA_SPLITS_PATH, "train.txt"), "w") as f:
        for folder_name in train_traj_folder_names:
            f.write(folder_name + "\n")

    with open(os.path.join(DATA_SPLITS_PATH, "eval.txt"), "w") as f:
        for folder_name in eval_traj_folder_names:
            f.write(folder_name + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="Train/test split (default: 0.8)",
    )
        
    args = parser.parse_args()
    main(args)
    print("Done")
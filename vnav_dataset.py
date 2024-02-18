import io
import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class VnavDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        config: dict,
    ):
        """
        Main Vec Nav dataset class
        """
        # Paths
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(config["datasets_folder"], dataset_name)
        traj_names_file = os.path.join(self.dataset_folder, "partitions", f"{dataset_type}.txt")

        # Parameters
        self.image_size = config["image_size"]
        self.stride = config["stride"]
        self.pred_horizon = config["pred_horizon"]
        self.context_size = config["context_size"]
        self.max_goal_dist = config["max_goal_dist"]
        self.max_traj_len = config["max_traj_len"]
        self.cam_rot_th = config["cam_rot_th"]
        self.goal_rot_th = config["goal_rot_th"]
        self.step_dist = config["step_distance"]

        # Build index
        with open(traj_names_file, "r") as f:
            self.traj_names = f.read().splitlines()

        self.index_to_data = self._build_index()

        # Cache
        self.cache_file = os.path.join(self.dataset_folder, f"cache_{self.image_size[0]}x{self.image_size[1]}.lmdb")
        self.lmdb_env = None

    def _build_index(self):
        samples_index = []
        traj_len_to_use = min(self.max_traj_len, len(self.traj_names))
        for traj_name in tqdm(self.traj_names[:traj_len_to_use]):
            traj_data = self._get_trajectory(traj_name)
            # Skip if doesn't exist
            if traj_data is None:
                print(f"{traj_name} doesn't exist, skipped")
                continue

            traj_len = len(traj_data["positions"])

            begin_time = self.context_size * self.stride
            end_time = traj_len - self.pred_horizon * self.stride

            for curr_time in range(begin_time, end_time):
                # Yaw at T=0
                yaw_t0 = traj_data["yaws"][curr_time] 

                # Direction vector pos(T=1)-pos(T=0)
                pos_t0 = traj_data["positions"][curr_time]
                pos_t1 = traj_data["positions"][curr_time + self.stride]
                dir_vec = self._to_local_coords(pos_t1, pos_t0, yaw_t0)
                dir_vec_deg = np.arctan2(dir_vec[1], dir_vec[0]) * 180 / np.pi

                # Large yaw offset
                if np.abs(dir_vec_deg) > self.cam_rot_th:
                    continue

                # min and max goal distance
                max_goal_dist = min(self.max_goal_dist * self.stride, traj_len - curr_time - 1)
                max_goal_dist = (max_goal_dist+1) // self.stride

                # Sample goal
                for _ in range(3):
                    # Sample goal
                    goal_offset = np.random.randint(1, max_goal_dist+1)
                    goal_time = curr_time + goal_offset * self.stride

                    # Check yaw offset
                    goal_pos = traj_data["positions"][min(goal_time, len(traj_data["positions"]) - 1)]
                    goal_vec = self._to_local_coords(goal_pos, pos_t0, yaw_t0)
                    goal_rot_deg = np.arctan2(goal_vec[1], goal_vec[0]) * 180 / np.pi

                    # Large yaw offset
                    if np.abs(goal_rot_deg) > self.goal_rot_th:
                        break
                
                if np.abs(goal_rot_deg) > self.goal_rot_th:
                    # Add to index
                    samples_index.append((traj_name, curr_time, goal_time))

        return samples_index


    def _load_image(self, traj_name, time):
        # Open cache
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.cache_file, map_size=2**40, readonly=True) # 1TB cache

        # Load image
        with self.lmdb_env.begin() as txn:
            image_buffer = txn.get(f"{traj_name}_{time:06d}".encode())
            image_bytes = io.BytesIO(bytes(image_buffer))
        image = Image.open(image_bytes)
        image_tensor = TF.to_tensor(image)

        return image_tensor

    def _to_local_coords(self, points, origin, yaw0):
        # Rotation matrix
        R = np.array([
            [np.cos(yaw0), -np.sin(yaw0)],
            [np.sin(yaw0), np.cos(yaw0)],
        ])
        points = points - origin
        points = points @ R
        return points
    
    def _get_norm_scale(self, p: np.ndarray, d: float) -> np.ndarray:
        dist = np.linalg.norm(np.diff(p, axis=0), axis=1)

        # Outliers
        avg = np.mean(dist)
        std = np.std(dist)
        dist = dist[dist < (avg + std)]
        dist = dist[dist > (avg - std)]
        avg = np.mean(dist)

        # Scale
        scale = d / avg
        return scale

    def _compute_actions(self, traj_data, curr_time, goal_time):
        # Start and end time
        start_index = curr_time
        end_index = curr_time + self.pred_horizon * self.stride + 1

        # Get the actions, yaws, goal_vec
        positions = traj_data["positions"][start_index:end_index:self.stride] # [H+1, 2]
        yaws = traj_data["yaws"][start_index:end_index:self.stride] # [H+1, 1]
        goal_pos = traj_data["positions"][min(goal_time, len(traj_data["positions"]) - 1)]

        # Convert to local coordinates
        waypoints = self._to_local_coords(positions, positions[0], yaws[0])
        goal_pos = self._to_local_coords(goal_pos, positions[0], yaws[0])
        yaws = yaws - yaws[0]

        # Scale waypoints
        scale = self._get_norm_scale(waypoints, self.step_dist)
        waypoints *= scale
        goal_pos *= scale

        # Relative actions
        actions = waypoints[1:]
        yaws = yaws[1:]

        return actions, yaws, goal_pos
    
    def _get_trajectory(self, trajectory_name):
        #  traj_processed.pkl
        if os.path.exists(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl")):
            with open(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            return traj_data
        
        # traj_est.pkl
        if os.path.exists(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_est.pkl")):
            with open(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_est.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            return traj_data
        
        return None

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        # Index to data
        current_file, curr_time, goal_time = self.index_to_data[i]

        # Context images
        context_times = list(
            range(
                curr_time - (self.context_size-1)*self.stride,
                curr_time + 1,
                self.stride
                )
            )
        context = [(current_file, t) for t in context_times]
        loaded_images = [self._load_image(current_file, t) for current_file, t in context] # [3, H, W] * N
        obs_context = torch.stack(loaded_images, dim=0) # [N, 3, H, W]

        # Actions, yaws, goal vector
        curr_traj_data = self._get_trajectory(current_file)
        actions, yaws, goal_vec = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Metadata
        dataset_name: str = self.dataset_name
        if len(current_file.split("_")) == 1:
            video_index = int(current_file)
            traj_index = 0
        elif len(current_file.split("_")) == 2:
            video_index = int(current_file.split("_")[0])
            traj_index = int(current_file.split("_")[1])
        metadata = {
            "dataset_name": dataset_name,
            "traj_name": video_index,
            "traj_idx": traj_index,
            "timestep": curr_time,
        }

        # Return data
        return (
            torch.as_tensor(obs_context, dtype=torch.float32),
            torch.as_tensor(goal_vec, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(yaws, dtype=torch.float32),
            (metadata),
        )

    def __len__(self) -> int:
        return len(self.index_to_data)
    
    def __del__(self):
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

def vnav_collect_fn(batch):
    """
    Collect a batch of data from the dataset.
    """
    obs_context = torch.stack([x[0] for x in batch], dim=0)     # [B, N, 3, H, W]
    goal_vec = torch.stack([x[1] for x in batch], dim=0)        # [B, 2]
    actions = torch.stack([x[2] for x in batch], dim=0)         # [B, N, 2]
    yaws = torch.stack([x[3] for x in batch], dim=0)            # [B, N, 1]
    metadata = [x[4] for x in batch]

    return obs_context, goal_vec, actions, yaws, metadata

# Test
if __name__ == "__main__":
    
    vnav_dataset = VnavDataset(
        dataset_name="scalenet",
        dataset_type="train",
        config={
            "datasets_folder": "/home/caoruixiang/datasets_mnt/vnav_datasets",
            "image_size": (256, 144),
            "stride": 5,
            "pred_horizon": 8,
            "context_size": 3,
            "max_goal_dist": 20,
            "max_traj_len": -1,
            "cam_rot_th": 26.565051177,
        }
    )

    import matplotlib.pyplot as plt

    for i in range(100):
        # Take sample
        sample = vnav_dataset[np.random.randint(len(vnav_dataset))]

        obs_imgs, goal_vec, gt_actions, gt_yaws, metadata = sample

        # Figure
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Last image))
        axs[1].set_xlabel('y(m)')
        axs[1].set_ylabel('x(m)')
        axs[1].invert_xaxis()
        axs[1].axis('equal')
        axs[1].grid()

        # Add first point
        gt_actions = np.concatenate([np.array([[0, 0]]), gt_actions], axis=0)
        gt_yaws = np.concatenate([np.array([0]), gt_yaws], axis=0)

        # Actions (ROS)
        gt_x = gt_actions[:, 0]
        gt_y = gt_actions[:, 1]
        axs[1].plot(gt_y, gt_x, 'ro', markersize=2)
        axs[1].plot(gt_y, gt_x, 'r-', markersize=0.5)
        
        # Goal vector (ROS)
        axs[1].plot(goal_vec[1], goal_vec[0], 'bx', markersize=15)

        # Yaw (ROS)
        axs[1].quiver(
            gt_y,
            gt_x,
            -np.sin(gt_yaws),
            np.cos(gt_yaws),
            color='g', 
            alpha=0.5,
            width=0.005,
        ) # GT directions

        # Plot +- 30 degree lines 
        x = np.array([-2, 0, 2])
        y = np.array([4, 0, 4])
        axs[1].plot(x, y, 'b--', alpha=0.5)

        plt.show()
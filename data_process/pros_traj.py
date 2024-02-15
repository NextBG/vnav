import os
import pickle
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from multiprocessing import Pool

TRAJ_ROOT_DIR = '/home/caoruixiang/datasets_mnt/vnav_datasets/tokyotownwalk_shinjukupark/trajectories'

# STD_DIST = 0.10942

def quat_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]: # w, x, y, z to rx, ry, rz
    w, x, y, z = q
    rx = np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2))
    ry = np.arcsin(2*(w*y-z*x))
    rz = np.arctan2(2*(w*z+x*y), 1-2*(y**2+z**2))
    return rx, ry, rz

# def _scale(p: np.ndarray, targ_mean_dist: float) -> np.ndarray:
#     dist = np.linalg.norm(np.diff(p, axis=0), axis=1)

#     # Outliers
#     avg = np.mean(dist)
#     std = np.std(dist)
#     dist = dist[dist < (avg + std)]
#     dist = dist[dist > (avg - std)]
#     avg = np.mean(dist)

#     # Scale
#     scale = targ_mean_dist / avg
#     return p * scale

def process_traj(trajs: str):
    print(f'Processing {trajs}... ', end='')
    traj_path = os.path.join(TRAJ_ROOT_DIR, trajs, 'traj_est.pkl')

    # Skip if traj_est.pkl don't exist
    if not os.path.exists(traj_path):
        print(f'Trajectory does not exist, skipped')
        return
    
    # # Skip if traj_processed.pkl already exists
    # if os.path.exists(os.path.join(TRAJ_ROOT_DIR, trajs, 'traj_processed.pkl')):
    #     print(f'Aready processed, skipped')
    #     return
    
    with open(traj_path, 'rb') as f:
        raw_traj = pickle.load(f)

    # From opengl to ros coordinate
    x_cg = raw_traj[:,0] # Right
    y_cg = raw_traj[:,1] # Down
    z_cg = raw_traj[:,2] # Forward
    x_ros = z_cg # Forward
    y_ros = -x_cg # Left
    z_ros = -y_cg # Up

    # Positions
    p_ros = np.stack((x_ros, y_ros), axis=1)

    # # Scale trajectory
    # p_ros = _scale(p_ros, STD_DIST)

    # Yaws
    q_cg = raw_traj[:,3:]
    qw_cg = q_cg[:,0]
    qx_cg = q_cg[:,1]
    qy_cg = q_cg[:,2]
    qz_cg = q_cg[:,3]
    qw_ros = qw_cg
    qx_ros = qz_cg
    qy_ros = -qx_cg
    qz_ros = -qy_cg
    yaws = []
    q_ros = np.stack((qw_ros, qx_ros, qy_ros, qz_ros), axis=1)
    for q in q_ros:
        yaws.append(quat_to_euler(q)[2])
    yaws = np.array(yaws)

    # Raw
    pose_ros = np.stack((x_ros, y_ros, z_ros, qw_ros, qx_ros, qy_ros, qz_ros), axis=1)

    traj = {}
    traj['positions'] = p_ros
    traj['yaws'] = yaws
    traj['raw'] = pose_ros

    # Save the processed trajectory
    with open(os.path.join(TRAJ_ROOT_DIR, trajs, 'traj_processed.pkl'), 'wb') as f:
        pickle.dump(traj, f)

    # Plot the trajectory and save the image
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.invert_xaxis()
    ax.grid()

    # Plot the trajectory
    ax.plot(traj['positions'][:,1], traj['positions'][:,0], color='r', linewidth=0.1)
    # Plot every 100th point
    ax.plot(traj['positions'][:,1][::100], traj['positions'][:,0][::100], 'o', color='b', markersize=0.1)
    # Plot the direction vector
    vis_stride = 10
    ax.quiver(
        traj['positions'][:,1][::vis_stride], 
        traj['positions'][:,0][::vis_stride], 
        -np.sin(traj['yaws'][::vis_stride]),
        np.cos(traj['yaws'][::vis_stride]),
        color='g', 
        width=0.001)

    # legend
    ax.legend(['trajectory', 'every 100th point', 'direction vector', 'yaw angle'], loc='lower right', fontsize='xx-small')

    fig.savefig(os.path.join(TRAJ_ROOT_DIR, trajs, 'traj_vis.png'), dpi=1000)

    # Close the figure
    plt.close(fig)

    print(f'Done')

if __name__ == '__main__':
    trajs = sorted(os.listdir(TRAJ_ROOT_DIR))
    with Pool() as pool:
        pool.map(process_traj, trajs)

    print('All done!')
import os
import pickle
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

TRAJ_ROOT_DIR = '/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/trajectories'

def quat_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]: # w, x, y, z to rx, ry, rz
    w, x, y, z = q
    rx = np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2))
    ry = np.arcsin(2*(w*y-z*x))
    rz = np.arctan2(2*(w*z+x*y), 1-2*(y**2+z**2))
    return rx, ry, rz

# Load data
avgs = []
stds = []
lowests = []
highests = []

for trajs in sorted(os.listdir(TRAJ_ROOT_DIR))[:]:
    print(f'Processing {trajs}... ', end='')
    traj_path = os.path.join(TRAJ_ROOT_DIR, trajs, 'traj_est.pkl')

    # Skip if traj_est.pkl don't exist
    if not os.path.exists(traj_path):
        print(f'Trajectory does not exist, skipped')
        continue
    
    with open(traj_path, 'rb') as f:
        raw_traj = pickle.load(f)

    # p_ros = raw_traj['positions'] # 0.10942

    # From opengl to ros coordinate
    x_cg = raw_traj[:,0] # Right
    y_cg = raw_traj[:,1] # Down
    z_cg = raw_traj[:,2] # Forward
    x_ros = z_cg # Forward
    y_ros = -x_cg # Left
    z_ros = -y_cg # Up

    # Positions
    p_ros = np.stack((x_ros, y_ros), axis=1)

    # Distance
    dist = np.linalg.norm(np.diff(p_ros, axis=0), axis=1)

    # Statistics
    avg = np.mean(dist)
    std = np.std(dist)

    # Remove outliers
    dist = dist[dist < (avg + std)]
    dist = dist[dist > (avg - std)]

    # Statistics
    avg = np.mean(dist)
    std = np.std(dist)
    lowest = np.min(dist)
    highest = np.max(dist)
    avgs.append(avg)
    stds.append(std)
    lowests.append(lowest)
    highests.append(highest)
    print(f'avg: {avg:.5f}, std: {std:.5f}, low: {lowest:.5f}, high: {highest:.5f}')

    # Rescale
    scale = 0.10942 / avg
    p_ros *= scale 

    dist = np.linalg.norm(np.diff(p_ros, axis=0), axis=1)
    avg = np.mean(dist)
    std = np.std(dist)
    dist = dist[dist < (avg + std)]
    dist = dist[dist > (avg - std)]
    avg = np.mean(dist)

    print(f'avg: {avg:.5f}')

    # # Plot the distance
    # fig, ax = plt.subplots()
    # ax.set_title("Distance")
    # ax.set_xlabel('time')
    # ax.set_ylabel('distance')
    # ax.set_xlim(0, len(dist))
    # ax.grid()
    # ax.plot(dist, color='b', linewidth=1)
    # plt.show()
    # plt.close(fig)

# Plot the statistics
fig, ax = plt.subplots()
ax.set_title("Distance Statistics")
ax.set_xlabel('time')
ax.set_ylabel('distance')
ax.set_xlim(0, len(avgs))
ax.grid()
ax.plot(avgs, color='b', linewidth=1, label='avg')
# ax.plot(stds, color='g', linewidth=1, label='std')
# ax.plot(lowests, color='r', linewidth=1, label='lowest')
# ax.plot(highests, color='y', linewidth=1, label='highest')
ax.legend()
plt.show()

avg = np.mean(avgs)
std = np.std(avgs)

avgs = np.array(avgs)
avgs = avgs[avgs < (avg + std)]
avgs = avgs[avgs > (avg - std)]
avg = np.mean(avgs)

print(f'Overall avg: {avg:.5f}')

print('All done!')
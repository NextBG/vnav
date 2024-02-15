import os
import pickle
import matplotlib.pyplot as plt
import quaternion
import numpy as np
import shutil

IN_TRAJ_DIR = '/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/trajectories'
OUT_TRAJ_DIR = '/home/caoruixiang/datasets_mnt/vnav_datasets_jpg/nadawalk_tokyo/trajectories'

# move the intr_est.pkl, traj_est.pkl, traj_processed.pkl, traj_vis.png to OUT_TRAJ_DIR
for traj in sorted(os.listdir(IN_TRAJ_DIR))[:]:
    print(f'Copying {traj}... ', end='')
    traj_path = os.path.join(IN_TRAJ_DIR, traj)

    # shutil.move(os.path.join(traj_path, 'intr_est.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'intr_est.pkl'))
    # shutil.move(os.path.join(traj_path, 'traj_est.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_est.pkl'))
    # shutil.move(os.path.join(traj_path, 'traj_processed.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_processed.pkl'))
    # shutil.move(os.path.join(traj_path, 'traj_vis.png'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_vis.png'))

    # Move only if source exists
    if os.path.exists(os.path.join(traj_path, 'intr_est.pkl')):
        shutil.copy(os.path.join(traj_path, 'intr_est.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'intr_est.pkl'))

    if os.path.exists(os.path.join(traj_path, 'traj_est.pkl')):
        shutil.copy(os.path.join(traj_path, 'traj_est.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_est.pkl'))

    if os.path.exists(os.path.join(traj_path, 'traj_processed.pkl')):
        shutil.copy(os.path.join(traj_path, 'traj_processed.pkl'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_processed.pkl'))

    if os.path.exists(os.path.join(traj_path, 'traj_vis.png')):
        shutil.copy(os.path.join(traj_path, 'traj_vis.png'), os.path.join(OUT_TRAJ_DIR, traj, 'traj_vis.png'))


    print(f'Processed {traj}')
import cv2
import os
from tqdm import tqdm
import multiprocessing

# Desolve videos into frames
# IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/videos"
IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/tokyotownwalk_shinjukupark/videos"
OUT_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/tokyotownwalk_shinjukupark/temp"

def dissolve_video(video_file_name):
    # Settings
    in_path=IN_PATH
    out_path=OUT_PATH
    frame_interval = 3
    frames_per_traj=3000

    video_name = video_file_name.split('.')[0]
    print(f"Dissolving {video_name}, frames per trajectory: {frames_per_traj}")

    # Skip if video does not exist
    if not os.path.exists(os.path.join(in_path, video_name+".mp4")):
        print(f'Video does not exist, skipped')
        return

    # Skip if already exists
    frames_dir_path = os.path.join(out_path, f'{video_name}', "frames")
    if os.path.exists(frames_dir_path):
        print(f'Skipping {video_name}, already exists')
        return
    os.makedirs(frames_dir_path)

    vidcap = cv2.VideoCapture(os.path.join(in_path, video_name+".mp4"))


    start_padding = 1800 # Skip the first 1 minute
    end_padding = 1800 # Skip the last 1 minute
    all_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    stridded_frame_count = (all_frame_count-start_padding-end_padding) // frame_interval

    frame_to_dissolve = (stridded_frame_count//frames_per_traj) * frames_per_traj
    print(f"{video_name}, frames/traj {frames_per_traj}, Frame_to_dissolve {frame_to_dissolve}, All frame count {all_frame_count}, Stridded frame count {stridded_frame_count}")

    count = 0
    frame_number = 0

    out_frame_count = (all_frame_count-start_padding-end_padding)//(frame_interval*frame_to_dissolve)*(frame_interval*frame_to_dissolve)
    for _ in range(start_padding + out_frame_count):
        # Get frame
        if count % frame_interval == 0 and count >= start_padding:
            _, image = vidcap.read()
            # Save frame as JPG file
            cv2.imwrite(os.path.join(frames_dir_path, f"{frame_number:06d}.jpg"), image)
            frame_number += 1
        else:
            vidcap.grab()
        count += 1

    print(f'Dissolved {video_name}')

# Create trajectory root path if not exists
os.makedirs(OUT_PATH, exist_ok=True)

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

pool.map(dissolve_video, sorted(os.listdir(IN_PATH)))

print("Dissolve all done!")



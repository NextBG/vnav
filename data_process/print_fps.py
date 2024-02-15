import cv2
import os
from tqdm import tqdm
import multiprocessing

# Desolve videos into frames
# IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_tokyo/videos"
IN_PATH = "/home/caoruixiang/datasets_mnt/vnav_datasets/nadawalk_test/videos"
OUT_PATH = "/home/caoruixiang/datasets_mnt/test_datasets/nadawalk_test/temp"

for video in sorted(os.listdir(IN_PATH)):
    video_name = video.split('.')[0]
    
    # print the framerate of the video
    vidcap = cv2.VideoCapture(os.path.join(IN_PATH, video))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f'Video {video_name}, FPS {fps}')
    vidcap.release()
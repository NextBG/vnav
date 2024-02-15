import os
from pytube import Channel, Playlist, YouTube

# get all videos from the playlist
# playlist_url = 'https://www.youtube.com/playlist?list=PLBG9Vyt-7QLsSZ6sev-ERGn2RTd9bPnqx' # nadawalk tokyo
# playlist_url = 'https://www.youtube.com/playlist?list=PLBG9Vyt-7QLsdj0GteMhR2owOKLQYlma4' # nadawalk kanagawa
# playlist_url = 'https://www.youtube.com/playlist?list=PLBG9Vyt-7QLs6s_UG5OS4CDkrN6DnNKOA' # nadawalk rain
# playlist_url = 'https://www.youtube.com/playlist?list=PLqzMjQGUFh8cYaXDFqialufxP_qQD0tu1' # walkingtours shopandmall
# playlist_url = 'https://www.youtube.com/playlist?list=PLQZ2EdF3w6Tef0A9Ltt8F13dh4iZa1DLv' # VideoStreetViewJapan TokyoWalks
playlist_url = 'https://www.youtube.com/playlist?list=PLFzuCQTe6ZGOlRl4YhwG2bIv-qV63B1lM' # TokyoTownWalk ShinjukuPark


playlist = Playlist(playlist_url)
urls = playlist.video_urls

playlist_name = playlist.title
datasets_root = '/home/caoruixiang/datasets_mnt/vnav_datasets'
videos_download_path = os.path.join(datasets_root, playlist_name, 'videos')
if not os.path.exists(videos_download_path):
    os.makedirs(videos_download_path)

# Download all videos from the playlist in mp4
index = 0
for url in urls:
    retry_count = 0
    max_retries = 3
    download_success = False

    # Define the file path
    file_path = os.path.join(videos_download_path, "{}.mp4".format(str(index).zfill(6)))

    # skip if already downloaded
    if os.path.exists(file_path):
        print('Already downloaded, skip')
        index += 1
        continue

    while retry_count < max_retries and not download_success:
        try:
            yt = YouTube(url)
            print('Downloading video: ', yt.title, ' from url: ', url, ' index: ', index)

            yt.streams.filter(progressive=True, file_extension='mp4', res='360p', fps=30).first().download(videos_download_path, filename="{}.mp4".format(str(index).zfill(6)))
            print('Success')
            download_success = True
            index += 1
        except Exception as e:
            print('Failed:', e)
            if os.path.exists(file_path):
                os.remove(file_path)
            retry_count += 1
            print(f'Retrying... Attempt {retry_count} of {max_retries}')

    if not download_success:
        print('Download failed after maximum retries. Moving to next URL.')
        index += 1

import cv2
import os


def extract_frames(video_path, output_dir, target_images):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // target_images)
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_file, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Total frames saved from {video_path}: {saved_count}")


# 视频文件路径和输出目录
video_paths = ['D:/资料/组会/开题报告/参考资料（涉密）/农科院资料/白萝卜采收机/白萝卜采收机/白萝卜采收机设计图/C7694.mp4', 'D:/资料/组会/开题报告/参考资料（涉密）/农科院资料/白萝卜采收机/白萝卜采收机/白萝卜采收机设计图/WeChat_20231123160913.mp4']
output_dirs = ['D:/radish/data/images', 'D:/radish/data/images']
target_images = 500

for video_path, output_dir in zip(video_paths, output_dirs):
    extract_frames(video_path, output_dir, target_images)

import os
import shutil
import random

original_images_path = 'D:/BaiduNetdiskDownload/Self Driving Car.v3-fixed-small.yolov5pytorch/export/images'
original_labels_path = 'D:/BaiduNetdiskDownload/Self Driving Car.v3-fixed-small.yolov5pytorch/export/labels'

target_base_path = 'D:/yolov5/data'
target_images_path = os.path.join(target_base_path, 'Images')
target_labels_path = os.path.join(target_base_path, 'Labels')

# 创建必要的目录
for path in [target_images_path, target_labels_path]:
    for subdir in ['train', 'val']:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)

# 获取所有图片文件的基本名称
image_files = [os.path.splitext(f)[0] for f in os.listdir(original_images_path)
               if os.path.isfile(os.path.join(original_images_path, f))]

# 随机选择 5000 张图片
random.shuffle(image_files)
selected_images = image_files[:5000]  # 选择 5000 张图片

# 复制图片和标签到训练集和验证集
for i, image_base_name in enumerate(selected_images):
    image_file = image_base_name + '.jpg'  # JPG 文件
    label_file = image_base_name + '.txt'

    image_src = os.path.join(original_images_path, image_file)
    label_src = os.path.join(original_labels_path, label_file)

    if i < int(0.8 * len(selected_images)):  # 80% 划分到训练集
        target_image_dir = os.path.join(target_images_path, 'train')
        target_label_dir = os.path.join(target_labels_path, 'train')
    else:  # 20% 划分到验证集
        target_image_dir = os.path.join(target_images_path, 'val')
        target_label_dir = os.path.join(target_labels_path, 'val')

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(target_image_dir, image_file))
        shutil.copy(label_src, os.path.join(target_label_dir, label_file))

print("Selected 5000 images and their corresponding labels copied to train and val sets.")


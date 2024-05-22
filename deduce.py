import os
import json
import random
import shutil

# COCO数据集路径
coco_data_dir = 'D:/资料/组会/专业实践/yolov5/yolov5/datasets/train2017/'

# COCO标注文件路径
annotations_file = os.path.join(coco_data_dir, 'instances_train2017.json')

# 输出文件夹路径
output_folder = 'D:/资料/组会/专业实践/yolov5/yolov5/datasets/train1'
os.makedirs(output_folder, exist_ok=True)

# 读取COCO标注文件
with open(annotations_file, 'r') as f:
    coco_data = json.load(f)

# 找到带有人、车辆和交通信号灯的图片ID
selected_image_ids = {'person': [], 'car': [], 'traffic_light': []}
target_categories = {'person': 1, 'car': 3, 'traffic_light': 10}  # Category IDs for person, car, traffic light

for annotation in coco_data['annotations']:
    for category, category_id in target_categories.items():
        if annotation['category_id'] == category_id:
            selected_image_ids[category].append(annotation['image_id'])

# 计算每个类别应该选择的图片数量
total_images = 3200
images_per_category = {category: total_images // len(target_categories) for category in target_categories}
remaining_images = total_images % len(target_categories)

# 随机选择图片，确保每个类别数量相对均衡
selected_image_ids_final = []
for category, ids in selected_image_ids.items():
    random.shuffle(ids)
    selected_ids = ids[:images_per_category[category]]
    selected_image_ids_final.extend(selected_ids)

# 如果有剩余的图片数量，随机分配到各类别
if remaining_images > 0:
    categories = list(target_categories.keys())
    random.shuffle(categories)
    for category in categories[:remaining_images]:
        selected_image_ids_final.append(selected_image_ids[category][0])

# 复制选中的图片到输出文件夹（保留原格式和名称）
for image_info in coco_data['images']:
    if image_info['id'] in selected_image_ids_final:
        image_filename = image_info['file_name']
        image_path = os.path.join(coco_data_dir, image_filename)
        shutil.copy(image_path, output_folder)

print(f"Selected {len(selected_image_ids_final)} images have been copied to the specified folder.")

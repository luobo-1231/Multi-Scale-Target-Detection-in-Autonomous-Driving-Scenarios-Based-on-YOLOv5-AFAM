import os
import json

# 文件夹路径
folder_path = 'D:/yolov5/data/labels/train'  # 替换为你的文件夹路径

# 初始化类别计数器
label_counts = {}

# 遍历文件夹中的所有 JSON 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            # 假设JSON中的类别信息在 'category' 字段下
            categories = [obj['category'] for frame in json_data['frames'] for obj in frame['objects']]

            # 统计各类别标签的数量
            for label in categories:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

# 输出每个类别的标签数量
for label, count in label_counts.items():
    print(f'{label}标签数量: {count}')


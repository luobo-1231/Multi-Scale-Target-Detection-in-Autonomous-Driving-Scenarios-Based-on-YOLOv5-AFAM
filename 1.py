import os

# 文件夹路径
folder_path = 'D:/yolov5/data/label/train'  # 替换为你的文件夹路径

# 初始化类别计数器
label_counts = {}

# 遍历文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label = line.split(' ')[0]  # 获取标签的类别
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

# 输出每个类别的标签数量
for label, count in label_counts.items():
    print(f'{label}标签数量: {count}')

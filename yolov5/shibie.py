import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# 数据路径
data_dir = 'D:/yolov5/JPEG/train'
labels_dir = 'D:/yolov5/JPEG/labels'
classes_file = 'D:/yolov5/JPEG/classes.txt'

# 图像尺寸
image_size = (224, 224)
batch_size = 16

# 读取类别列表
with open(classes_file, 'r') as file:
    class_names = [line.strip() for line in file]

# 初始化空的列表来存储图像和标签数据
images = []
labels = []

# 遍历数据目录
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建图像文件路径
        img_path = os.path.join(data_dir, filename)

        # 读取图像并调整大小
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)

        # 构建标签文件路径
        label_filename = os.path.splitext(filename)[0] + '.xml'
        label_path = os.path.join(labels_dir, label_filename)

        # 解析XML标签文件
        tree = ET.parse(label_path)
        root = tree.getroot()

        # 从XML中提取标签信息（示例中假设标签是'friend'或'enemy'）
        label_data = None
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label in class_names:
                label_id = class_names.index(label)
                label_data = label_id

        if label_data is not None:
            # 将图像和标签添加到列表中
            images.append(img)
            labels.append(label_data)

# 将图像和标签转换为NumPy数组
images = np.array(images)
labels = np.array(labels)

# 进行数据标准化，确保像素值在[0, 1]范围内
images = images / 255.0

# 构建模型
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 提前停止训练，防止过拟合
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(
    images,
    labels,
    batch_size=batch_size,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# 保存模型
model.save('enemy_friend_model.h5')

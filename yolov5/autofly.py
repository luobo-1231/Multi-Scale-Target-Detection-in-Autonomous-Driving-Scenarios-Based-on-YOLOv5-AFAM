import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 数据目录
data_directory = 'D:/yolov5/JPEG/train'

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 读取图像文件和标签
image_paths = []
labels = []

for root, _, files in os.walk(data_directory):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_paths.append(os.path.join(root, file))
            # 解析标签信息
            xml_file = os.path.splitext(file)[0] + ".xml"  # 假设XML文件与图像文件同名
            xml_path = os.path.join(root, xml_file)
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root_element = tree.getroot()
                label = root_element.find("object").find("name").text
                if label == "enemy":
                    labels.append(1)  # 敌人
                elif label == "friend":
                    labels.append(0)  # 平民

# 随机打乱数据
data = list(zip(image_paths, labels))
np.random.shuffle(data)
image_paths, labels = zip(*data)

# 准备图像数据
def preprocess_image(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # 归一化像素值到0到1之间
    return image

images = [preprocess_image(image_path) for image_path in image_paths]

# 划分训练集和测试集
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = np.array(images[:split_index])
train_labels = np.array(labels[:split_index])
test_images = np.array(images[split_index:])
test_labels = np.array(labels[split_index:])

# 创建预训练的ResNet-50模型（不包括顶层分类器）
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定义分类器层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# 构建新模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(optimizer=Adam(lr=0.0001),  # 调整学习率
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型，使用数据增强
batch_size = 32
epochs = 50
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=(test_images, test_labels)
)

# 保存性能图为图像文件
plt.figure(figsize=(12, 4))

# 绘制准确性图
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# 绘制损失图
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 保存图像文件
performance_image_path = 'performance.png'
plt.savefig(performance_image_path)
plt.show()

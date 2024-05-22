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

# 使用摄像头进行物体识别
cap = cv2.VideoCapture(0)  # 使用默认摄像头

while True:
    ret, frame = cap.read()  # 读取一帧视频

    # 调整帧的大小
    frame = cv2.resize(frame, image_size)

    # 图像标准化
    frame = frame / 255.0

    # 使用模型进行预测
    predictions = model.predict(np.expand_dims(frame, axis=0))

    # 获取最大概率的类别
    predicted_class = np.argmax(predictions)

    # 根据类别列表获取类别名称
    class_name = class_names[predicted_class]

    # 在图像上绘制标签
    cv2.putText(frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 设置窗口大小
    window_width = 800
    window_height = 600
    cv2.namedWindow('Object Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Recognition', window_width, window_height)

    # 显示图像
    cv2.imshow('Object Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下'q'键退出
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

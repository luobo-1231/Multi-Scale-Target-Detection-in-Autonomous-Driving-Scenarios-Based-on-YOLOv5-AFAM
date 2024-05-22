
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# 载入训练好的模型
model = load_model('enemy_friend_model.h5')

# 使用默认摄像头进行物体识别
cap = cv2.VideoCapture(0)  # 使用默认摄像头

while True:
    ret, frame = cap.read()  # 读取一帧视频

    # 调整帧的大小
    frame = cv2.resize(frame, (224, 224))

    # 图像标准化
    frame = frame / 255.0

    # 使用模型进行预测
    predictions = model.predict(np.expand_dims(frame, axis=0))

    # 获取最大概率的类别
    predicted_class = np.argmax(predictions)

    # 根据类别列表获取类别名称
    class_names = ['friend', 'enemy']  # 替换为你的类别列表
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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 载入训练好的模型
model = load_model('enemy_friend_model.h5')

# 图像文件路径
test_image_path = 'D:/yolov5/JPEG/train/120.png'  # 替换为你的测试图像路径

# 图像尺寸
image_size = (224, 224)

# 读取类别列表
with open('D:/yolov5/JPEG/classes.txt', 'r') as file:
    class_names = [line.strip() for line in file]

# 载入图像并进行预处理
img = load_img(test_image_path, target_size=image_size)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # 图像标准化

# 使用模型进行预测
predictions = model.predict(img_array)

# 获取最大概率的类别
predicted_class = np.argmax(predictions)

# 根据类别列表获取类别名称
class_name = class_names[predicted_class]

print(f'The predicted class is: {class_name}')

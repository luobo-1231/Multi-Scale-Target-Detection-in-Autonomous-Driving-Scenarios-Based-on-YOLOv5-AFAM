from PIL import Image
import os

# 输入文件夹路径
input_folder = 'D:/yolov5/data/images/val'  # 修改为包含PNG图像的文件夹路径

# 输出文件夹路径
output_folder = 'D:/yolov5/data/images/=val/1'  # 修改为存储转换后JPG图像的文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有PNG图像
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 读取PNG图像
        img = Image.open(os.path.join(input_folder, filename))

        # 将PNG图像转换为JPG格式（保存质量可调整）
        img = img.convert("RGB")  # 转换为RGB格式（避免alpha通道）
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        img.save(os.path.join(output_folder, jpg_filename), quality=95)  # 调整保存质量

        print(f"Converted {filename} to {jpg_filename}")

print("Conversion completed.")

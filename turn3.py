from PIL import Image
import os

# 原始图片路径
original_images_path = 'D:/yolov5/data/images/val'  # 替换为你的 PNG 图片文件夹路径
# 目标图片路径
target_images_path = 'D:/yolov5/data/images/val/1'  # 替换为你想要保存 JPEG 图片的文件夹路径

# 如果目标文件夹不存在，则创建它
os.makedirs(target_images_path, exist_ok=True)

# 遍历原始图片文件夹中的所有文件
for filename in os.listdir(original_images_path):
    if filename.endswith('.png'):
        # 构建 PNG 图片的完整路径
        png_image_path = os.path.join(original_images_path, filename)

        # 打开 PNG 图片
        png_image = Image.open(png_image_path)

        # 将 PNG 图片转换为 JPEG 格式（质量设为95，可根据需要调整）
        jpeg_image_path = os.path.join(target_images_path, os.path.splitext(filename)[0] + '.jpg')
        png_image.convert('RGB').save(jpeg_image_path, 'JPEG', quality=95)

print("Conversion completed: PNG images converted to JPEG.")

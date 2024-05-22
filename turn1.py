import xml.etree.ElementTree as ET
import os

classes = ["car", "person", "traffic_light"]  # 你的类别列表

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# 以下代码需要根据你的文件夹路径进行修改
xml_dir = 'D:\yolov5\VOCdevkit\VOC2007\label'  # XML 文件所在的目录
txt_dir = 'D:\yolov5\VOCdevkit\VOC2007\labels'  # 输出的 TXT 文件目录

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

for file in os.listdir(xml_dir):
    if file.endswith('.xml'):
        xml_file = os.path.join(xml_dir, file)
        txt_file = os.path.join(txt_dir, os.path.splitext(file)[0] + '.txt')
        convert_annotation(xml_file, txt_file)

print("Conversion completed.")

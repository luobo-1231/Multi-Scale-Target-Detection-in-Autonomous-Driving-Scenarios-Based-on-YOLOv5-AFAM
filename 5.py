# xml_to_yolo_txt.py
# 此代码和VOC_KITTI文件夹同目录
import glob
import xml.etree.ElementTree as ET

class_names = ['Car', 'Pedestrian']
path = 'D:/BaiduNetdiskDownload/Annotations'

def single_xml_to_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    txt_file = xml_file.split('.')[0] + '.txt'  # 修改这里，保持与原始XML文件相同的名称
    with open(txt_file, 'w') as txt_file:
        for member in root.findall('object'):
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            class_name = member[0].text

            class_num = class_names.index(class_name)

            box_x_min = int(member[1][0].text)
            box_y_min = int(member[1][1].text)
            box_x_max = int(member[1][2].text)
            box_y_max = int(member[1][3].text)

            x_center = float(box_x_min + box_x_max) / (2 * picture_width)
            y_center = float(box_y_min + box_y_max) / (2 * picture_height)
            width = float(box_x_max - box_x_min) / picture_width
            height = float(box_y_max - box_y_min) / picture_height

            txt_file.write(str(class_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')

def dir_xml_to_txt(path):
    for xml_file in glob.glob(path + '/*.xml'):  # 添加斜杠以确保匹配xml文件
        single_xml_to_txt(xml_file)

if __name__ == '__main__':
    dir_xml_to_txt(path)

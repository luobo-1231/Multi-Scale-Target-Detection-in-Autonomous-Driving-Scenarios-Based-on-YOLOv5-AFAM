# kitti_txt_to_xml.py
# encoding:utf-8
# 根据一个给定的XML Schema，使用DOM树的形式从空白文件生成一个XML
from xml.dom.minidom import Document
import cv2
import os

def generate_xml(name, split_lines, img_size, class_ind):
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)
    img_name = name + '.png'
    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
    source = doc.createElement('source')
    annotation.appendChild(source)
    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)
    size = doc.createElement('size')
    annotation.appendChild(size)
    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)
    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_ind:
            obj = doc.createElement('object')
            annotation.appendChild(obj)
            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            obj.appendChild(title)
            bndbox = doc.createElement('bndbox')
            obj.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[5]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[6]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[7]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
    return doc.toprettyxml(indent='')

if __name__ == '__main__':
    class_ind = ('Pedestrian', 'Car', )
    cur_dir = os.getcwd()
    labels_dir = os.path.join(cur_dir, 'D:/BaiduNetdiskDownload/data_object_label_2/training/label_2')  # 标签文件目录
    annotations_dir = os.path.join(cur_dir, 'D:/BaiduNetdiskDownload/Annotations')  # 新建的Annotations目录

    # 创建Annotations目录
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    for parent, dirnames, filenames in os.walk(labels_dir):
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)
            with open(full_path) as f:
                split_lines = f.readlines()
                name = file_name[:-4]
                img_name = name + '.png'
                img_path = os.path.join('D:/BaiduNetdiskDownload/data_object_image_2/training/image_2', img_name)  # 图像文件目录，需要根据实际情况修改
                img_size = cv2.imread(img_path).shape
                annotation_path = os.path.join(annotations_dir, name + '.xml')
                print(f"Saving annotation file: {annotation_path}")
                with open(annotation_path, 'w') as annotation_file:
                    annotation_file.write(generate_xml(name, split_lines, img_size, class_ind))

    print('All txts have been converted into XMLs in Annotations directory')




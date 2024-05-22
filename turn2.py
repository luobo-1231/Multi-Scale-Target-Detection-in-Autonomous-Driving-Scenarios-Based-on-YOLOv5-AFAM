import os
import shutil
import random
from collections import defaultdict

# Paths to the original and target directories
original_images_path = 'D:/BaiduNetdiskDownload/data_object_image_2/training/image_2'
original_labels_path = 'D:/BaiduNetdiskDownload/Annotations'

target_base_path = 'D:/yolov5/data'
target_images_path = os.path.join(target_base_path, 'images')
target_labels_path = os.path.join(target_base_path, 'labels')

# Create train and val directories
for path in [target_images_path, target_labels_path]:
    for subdir in ['train', 'val']:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)

# Get all image files
image_files = [os.path.splitext(f)[0] for f in os.listdir(original_images_path)
               if os.path.isfile(os.path.join(original_images_path, f))]

# Dictionary to hold class-wise images
class_images = defaultdict(list)

# Populate class_images dictionary
for image_base_name in image_files:
    label_file = image_base_name + '.txt'
    label_src = os.path.join(original_labels_path, label_file)

    if os.path.exists(label_src):
        with open(label_src, 'r') as label_file:
            lines = label_file.readlines()
            for line in lines:
                class_name = line.split()[0]
                class_images[class_name].append(image_base_name)

# Debug: print the number of images per class
for class_name, images in class_images.items():
    print(f"Class {class_name}: {len(images)} images")

# Find the minimum number of samples across all classes
min_samples = min(len(images) for images in class_images.values())

# Select images ensuring balanced classes
selected_images = []
for images in class_images.values():
    selected_images.extend(random.sample(images, min_samples))

# Shuffle the selected images
random.shuffle(selected_images)

# Limit the total number of selected images to 5000
selected_images = selected_images[:5000]

# Debug: print the total number of selected images
print(f"Total selected images: {len(selected_images)}")

# Split the selected images into train and val sets (80% train, 20% val)
train_images = selected_images[:4000]
val_images = selected_images[4000:]

# Debug: print the number of images in train and val sets
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

# Copy the selected images and labels to the respective train and val directories
for image_base_name in train_images:
    image_file = image_base_name + '.png'
    label_file = image_base_name + '.txt'

    image_src = os.path.join(original_images_path, image_file)
    label_src = os.path.join(original_labels_path, label_file)

    target_image_dir = os.path.join(target_images_path, 'train')
    target_label_dir = os.path.join(target_labels_path, 'train')

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(target_image_dir, image_file))
        shutil.copy(label_src, os.path.join(target_label_dir, label_file))
    else:
        print(f"Missing file: {image_base_name}")

for image_base_name in val_images:
    image_file = image_base_name + '.png'
    label_file = image_base_name + '.txt'

    image_src = os.path.join(original_images_path, image_file)
    label_src = os.path.join(original_labels_path, label_file)

    target_image_dir = os.path.join(target_images_path, 'val')
    target_label_dir = os.path.join(target_labels_path, 'val')

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(target_image_dir, image_file))
        shutil.copy(label_src, os.path.join(target_label_dir, label_file))
    else:
        print(f"Missing file: {image_base_name}")

print("Selected 5000 images and their corresponding labels copied to train and val sets based on balanced classes.")

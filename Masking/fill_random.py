import os
import shutil
import random
import numpy as np
from PIL import Image

def is_black_block(block):
    return np.all(block == 0)

def fill_random_color(block):
    random_color = np.random.randint(0, 256, block.shape, dtype=np.uint8)
    return random_color

def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    h, w, c = img_array.shape
    block_size = int(h / 8)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_array[i+1:i+block_size-1, j+1:j+block_size-1]
            if is_black_block(block):
                img_array[i+1:i+block_size-1, j+1:j+block_size-1] = fill_random_color(block)
    
    new_img = Image.fromarray(img_array)
    new_img.save(output_path)

def copy_and_process_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('*.png', '*.jpg', '*.jpeg'))
    
    for root, _, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, relative_path)
        
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_root, file)
                process_image(src_path, dst_path)

root_dir = '/home/haoyang/data/data/MdaCD/Dataset/'
Datasets = ['CDD', 'LEVIRCD', 'LEVIRCDPLUS', 'SYSUCD', 'WHUCD']
dirs = ['train', 'val', 'test']
penalties = ['0.1', '0.01']
for dataset in Datasets:
    for dir in dirs:
        for penalty in penalties:
            for t in ['1', '2']:
                source = root_dir + dataset + '/' + dir + '/penalty' + penalty + '_mask' + t
                destination = root_dir + dataset + '/' + dir + '/random' + penalty + '_mask' + t
                print(f'{source} -> {destination}')
                copy_and_process_images(source, destination)

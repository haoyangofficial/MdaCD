import os
import numpy as np
from PIL import Image
from collections import deque

# 扫描目录中的所有.png文件
def fill_empty_batches_in_images(directory):
    # 获取目录下所有png文件
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    # 遍历每张图片
    for file in files:
        file_path = os.path.join(directory, file)
        img = Image.open(file_path)
        img_array = np.array(img)

        # 假设每张图片是256x256且被分成8x8块
        block_size = 32  # 256 / 8
        rows, cols = 8, 8  # 8x8的小块

        # 遍历每个块并检查是否是空的（全黑）
        filled_img_array = img_array.copy()

        # 保存所有非空块的位置
        non_empty_blocks = []

        for row in range(rows):
            for col in range(cols):
                block = img_array[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size]
                
                if np.any(block != 0):  # 如果不是空块
                    non_empty_blocks.append((row, col, block))

        # 使用BFS填充空块
        queue = deque(non_empty_blocks)  # 队列中存储的是非空块的位置和对应的块
        visited = set()  # 记录已经处理过的块

        while queue:
            row, col, block = queue.popleft()

            # 遍历四个邻域方向
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + direction[0], col + direction[1]
                if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                    new_block = img_array[new_row*block_size:(new_row+1)*block_size, new_col*block_size:(new_col+1)*block_size]
                    # 如果是空块，填充
                    if np.all(new_block == 0):
                        filled_img_array[new_row*block_size:(new_row+1)*block_size, new_col*block_size:(new_col+1)*block_size] = block
                        queue.append((new_row, new_col, block))  # 把这个块加入队列
                    visited.add((new_row, new_col))

        # 保存修改后的图片
        filled_img = Image.fromarray(filled_img_array)
        filled_img.save(file_path)

if __name__ == "__main__":
    directory = 'path_to_your_directory'  # 设置目录路径
    fill_empty_batches_in_images(directory)
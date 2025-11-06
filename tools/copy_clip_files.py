import os
import shutil
import sys

def main(dataset, mask):
    for folder in ['train', 'val' ,'test']:
        for i in [1, 2]:
            source_file = f"/home/haoyang/data/data/MdaCD/Dataset/clip_files/{dataset}/{folder}/{mask}{i}_clipcls_56_vit16.json"
            destination_dir = f"/home/haoyang/data/data/MdaCD/Dataset/{dataset}/{folder}/time{i}_clipcls_56_vit16.json"
            
            if os.path.exists(source_file):
                shutil.copy(source_file, destination_dir)
                print(f"Copied {source_file} to {destination_dir}")
            else:
                print(f"Source file {source_file} does not exist.")
                return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python copy_clip_files.py <dataset> <mask>")
    else:
        dataset, mask = sys.argv[1], sys.argv[2]
        main(dataset, mask)
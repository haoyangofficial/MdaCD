import os

dataset_path = "/home/haoyang/data/data/MdaCD/Dataset/SYSUCD"
output_path = dataset_path

if not os.path.exists(output_path):
    os.makedirs(output_path)

for folder_name in ['train', 'val', 'test']:
    txt_path = os.path.join(output_path, f"{folder_name}.txt")

    with open(txt_path, "w") as f:
        image_folderA = os.path.join(dataset_path, folder_name, "time1")
        image_folderB =  os.path.join(dataset_path, folder_name, "time2")
        label_folder = os.path.join(dataset_path, folder_name, "label")

        image_filesA = sorted(os.listdir(image_folderA))
        image_filesB = sorted(os.listdir(image_folderB))
        label_files = sorted(os.listdir(label_folder))

        if len(image_filesA) != len(label_files):
            raise ValueError(f"File Numbers Mismatch: {folder_name}")

        for image_fileA, image_fileB, label_file in zip(image_filesA, image_filesB, label_files):
            image_pathA = os.path.join(image_folderA, image_fileA)
            image_pathB = os.path.join(image_folderB, image_fileB)
            label_path = os.path.join(label_folder, label_file)
            f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

print('Done')
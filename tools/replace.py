import os

def replace_text_in_files(root_dir, old_text, new_text):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                updated_content = content.replace(old_text, new_text)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"Updated: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

root_directory = "/home/haoyang/data/data/MdaCD/Dataset/clip_files/SYSUCD"
replace_text_in_files(root_directory, "/home/haoyang/v2/SYSUCD", "/home/haoyang/data/data/MdaCD/Dataset/SYSUCD")
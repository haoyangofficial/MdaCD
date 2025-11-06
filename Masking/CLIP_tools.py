import os
import json
import torch
import clip
import numpy as np
from PIL import Image

def read_class_names(path):
    """Read class names from a file."""
    with open(path, 'r') as f:
        class_names = [line.strip().split(', ')[0] for line in f]
    return class_names

def predict_image_class(image_path, model, preprocess, text, device):
    """Predict class probabilities for a given image."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs[0]

def CLIP_forward(src_path='./', 
                 img_split=['temp1', 'temp2'], 
                 model_name='ViT-B/16', 
                 class_names_path='rscls.txt', 
                 device='cpu'):
    """
    Process images from the dataset and generate predictions.

    Args:
        src_path (str): Path to the source dataset.
        split (list of str): Dataset split(s) to process (e.g., ['train', 'val']).
        img_split (list of str): Image split(s) to process (e.g., ['time1', 'time2']).
        model_name (str): CLIP model to load (e.g., 'ViT-B/16').
        class_names_path (str): Path to the class names file.
        device (str): Device to use for inference (e.g., 'cuda:0').
        tag (str): A tag to append to the output file name.

    Returns:
        None. Writes predictions to files.
    """
    # Load the model and preprocess pipeline
    model, preprocess = clip.load(model_name, device=device)
    class_names = read_class_names(class_names_path)
    text = clip.tokenize(class_names).to(device)

    # Loop through image split
    for isp in img_split:
        image_folder_path = os.path.join(src_path, isp)
        results = []

        # Iterate over the image files in the folder
        for filename in os.listdir(image_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_folder_path, filename)
                probs = predict_image_class(image_path, model, preprocess, text, device)
                sorted_probs = sorted(zip(class_names, probs.astype(np.float32)), key=lambda x: x[1], reverse=True)
                result = {"image_path": image_path}
                for cn, p in sorted_probs:
                    result[cn] = "{:.4f}".format(p)
                results.append(result)

        # Save predictions to a JSON file
        output_file = os.path.join(src_path, f'{isp}/{isp}.json')
        with open(output_file, mode='w') as predictions_file:
            json.dump(results, predictions_file, indent=4, ensure_ascii=False)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

components = [
    "ground track field", "farmland", "bare land", "fertile land", "golf course", 
    "solar panel", "prairie", "dense residential", "single-family residential", "square", 
    "oil tank", "storage tanks", "intersection", "parking lot", "meadow", "basketball court", 
    "mine", "wetland", "commercial area", "cotton field", "church", "runway", "park", 
    "industrial area", "campus", "interchange", "building", "tennis court", "stadium", 
    "chaparral", "cars", "airport", "railway", "freeway", "pond", "impermeable surface", 
    "shrubbery", "river", "island", "tree", "mountain", "road", "highway", "forest", 
    "container", "ship", "desert", "lake", "snow land", "cabin", "bridge", "terrace", 
    "airplane", "sea", "harbor", "beach"
]

def json_difference(file_path1, file_path2):
    data1 = load_json(file_path1)
    data2 = load_json(file_path2)

    total_squared_diff = 0

    for component in components:
        val1 = float(data1[0].get(component, 0))
        val2 = float(data2[0].get(component, 0))
        total_squared_diff += (val1 - val2) ** 2

    return total_squared_diff

# Example usage of the function
if __name__ == "__main__":
    # These would be set in another script or environment
    src_path = 'CDdata/WHUCD'
    img_split = ['time1', 'time2']
    model_name = 'ViT-B/16'
    class_names_path = 'rscls.txt'
    device = 'cuda:0'
    tag = '56_vit16'

    CLIP_forward(src_path, img_split, model_name, class_names_path, device)
import os
import shutil
import numpy as np
from PIL import Image

def add_random_noise(image, noise_factor=0.3):
    img_array = np.array(image)
    noise = np.random.normal(0, 1, img_array.shape) * 255 * noise_factor
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def create_noisy_dataset(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for folder in os.listdir(source_dir):
        source_folder = os.path.join(source_dir, folder)
        target_folder = os.path.join(target_dir, folder)
        
        if os.path.isdir(source_folder):
            os.makedirs(target_folder, exist_ok=True)
            
            for img_file in os.listdir(source_folder):
                if img_file.endswith('.png'):
                    source_img_path = os.path.join(source_folder, img_file)
                    target_img_path = os.path.join(target_folder, img_file)
                    
                    img = Image.open(source_img_path).convert('RGB')
                    noisy_img = add_random_noise(img)
                    noisy_img.save(target_img_path)

if __name__ == "__main__":
    source_directory = "dataset"
    target_directory = "aug_dataset"
    create_noisy_dataset(source_directory, target_directory)

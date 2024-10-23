import os
from PIL import Image
import numpy as np

def add_noise(image, noise_type='gaussian', factor=0.1):
    img_array = np.array(image)
    
    if noise_type == 'gaussian':
        mean = 0
        std = factor * 255
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_img = img_array + noise
    elif noise_type == 'salt_and_pepper':
        prob = factor
        rnd = np.random.rand(*img_array.shape[:2])
        noisy_img = img_array.copy()
        noisy_img[rnd < prob/2] = 0
        noisy_img[rnd > 1 - prob/2] = 255
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'salt_and_pepper'.")
    
    return Image.fromarray(np.uint8(np.clip(noisy_img, 0, 255)))

def add_distortion(image, distortion_type='blur', factor=1):
    if distortion_type == 'blur':
        return image.filter(Image.GaussianBlur(radius=factor))
    elif distortion_type == 'rotate':
        return image.rotate(factor)
    else:
        raise ValueError("Unsupported distortion type. Use 'blur' or 'rotate'.")

def process_image(image_path, operation='noise', op_type='gaussian', factor=0.1):
    img = Image.open(image_path)
    
    if operation == 'noise':
        processed_img = add_noise(img, op_type, factor)
    elif operation == 'distortion':
        processed_img = add_distortion(img, op_type, factor)
    else:
        raise ValueError("Unsupported operation. Use 'noise' or 'distortion'.")
    
    # Generate output filename
    file_name, file_extension = os.path.splitext(image_path)
    output_path = f"{file_name}_{operation}_{op_type}{file_extension}"
    
    processed_img.save(output_path)
    print(f"Processed image saved as {output_path}")

# Example usage
image_path = 'test_png.png'  # The image file in the current directory

# Add Gaussian noise
process_image(image_path, operation='noise', op_type='gaussian', factor=0.1)

# Uncomment to use other effects:
# Add Salt and Pepper noise
# process_image(image_path, operation='noise', op_type='salt_and_pepper', factor=0.05)

# Add Blur distortion
# process_image(image_path, operation='distortion', op_type='blur', factor=2)

# Add Rotation distortion
# process_image(image_path, operation='distortion', op
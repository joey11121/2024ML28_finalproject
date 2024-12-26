import os
import shutil

# Define the paths for folders A, B, and the output folder
folder_a = "test_output"  # Replace with the actual path to folder A
folder_b = "output_pdf"  # Replace with the actual path to folder B
output_folder = "dataset"  # Replace with the desired output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get lists of all .semantic and .png files
semantic_files = [f for f in os.listdir(folder_a) if f.endswith(".semantic")]
png_files = [f for f in os.listdir(folder_b) if f.endswith(".png")]

# Match files based on the common name (excluding extensions)
for semantic_file in semantic_files:
    base_name = os.path.splitext(semantic_file)[0]
    corresponding_png = base_name + ".png"
    
    if corresponding_png in png_files:
        # Create a subfolder with the base name
        subfolder_path = os.path.join(output_folder, base_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Copy the .semantic and .png files into the subfolder
        shutil.copy(os.path.join(folder_a, semantic_file), subfolder_path)
        shutil.copy(os.path.join(folder_b, corresponding_png), subfolder_path)
        print(f"successfully creat {subfolder_path}")

print("Files have been organized into subfolders.")

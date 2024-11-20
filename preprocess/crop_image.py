import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import os

def crop_music_score(input_path, output_dir, poppler_path, target_height=128, page_counter=1):
    """
    Convert PDF to images, crop each page to a specified height while preserving the musical content.
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Convert PDF to images
    try:
        pages = convert_from_path(
            input_path, 
            dpi=80,  # Higher DPI for better quality
            poppler_path=poppler_path
        )
    except Exception as e:
        raise ValueError(f"Error converting PDF to images: {e}")

    for i, page in enumerate(pages):
        # Convert PIL Image to numpy array and BGR format
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        if image is None:
            print(f"Warning: Could not convert page {i+1}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding for binarization
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find non-zero points (actual content)
        non_zero = cv2.findNonZero(adaptive_thresh)
        if non_zero is None:
            print(f"Warning: No content found in page {i+1}")
            continue

        # Get bounds of the content
        x, y, w, h = cv2.boundingRect(non_zero)

        # Add padding around the content
        padding = int(h * 0.2)  # 20% padding
        top = max(0, y - padding)
        bottom = min(image.shape[0], y + h + padding)

        # Crop the image
        cropped = image[top:bottom, :]

        # Resize to target height while maintaining aspect ratio
        aspect_ratio = (cropped.shape[1] / cropped.shape[0])
        new_width = int(target_height * aspect_ratio)
        resized = cv2.resize(cropped, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Save the result with a unique page number
        output_path = os.path.join(output_dir, f"page_{page_counter}.png")
        cv2.imwrite(output_path, resized)
        print(f"Processed page {i+1} from {input_path}, saved to {output_path}")

        # Increment the page counter
        page_counter += 1

    return page_counter



def main():
    # Define root directories
    base_dir = os.path.abspath(os.getcwd())
    input_dir = os.path.join(base_dir, "output_pdf")
    output_dir = os.path.join(base_dir, "output_png")
    poppler_path = r"C:\\Program Files\\poppler-24.08.0\\Library\\bin"

    # Validate input directory existence
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all subdirectories in input_dir
    for sub_dir in os.listdir(input_dir):
        sub_dir_path = os.path.join(input_dir, sub_dir)

        # Check if the current path is a directory
        if not os.path.isdir(sub_dir_path):
            continue

        # Check for "pdf_parts" folder inside the subdirectory
        pdf_parts_dir = os.path.join(sub_dir_path, "pdf_parts")
        if not os.path.exists(pdf_parts_dir):
            print(f"No 'pdf_parts' folder found in {sub_dir_path}, skipping.")
            continue

        # Create a corresponding output subdirectory in "output_png"
        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        # Process all PDF files in the "pdf_parts" folder
        pdf_files = [f for f in os.listdir(pdf_parts_dir) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {pdf_parts_dir}, skipping.")
            continue

        print(f"Processing PDFs in {pdf_parts_dir}...")
        page_counter = 1
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_parts_dir, pdf_file)
            page_counter = crop_music_score(pdf_path, output_sub_dir, poppler_path, page_counter=page_counter)
    
    print("Processing complete.")


if __name__ == "__main__":
    main()

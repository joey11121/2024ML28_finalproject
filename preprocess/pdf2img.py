import os
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

def pdf_to_png(pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Specify the path to poppler
    # You have to install poppler first before execution
    # Download poppler: https://github.com/oschwartz10612/poppler-windows/releases/
    poppler_path = r"C:\\poppler-24.08.0\\Library\\bin"  # Adjust this path as needed

    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, poppler_path=poppler_path)

        # Save each image as PNG
        for i, image in enumerate(images):
            image.save(os.path.join(output_folder, f'page_{i+1}.png'), 'PNG')
        
        print(f"PDF converted to PNG images in {output_folder}")
    except PDFInfoNotInstalledError:
        print("Error: Poppler is not installed or not found in the specified path.")
        print("Please make sure Poppler is installed and the path is correct.")

# Example usage
pdf_file = 'lab2_report.pdf'
output_dir = 'test_png'

pdf_to_png(pdf_file, output_dir)
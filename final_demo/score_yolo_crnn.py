import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
import argparse
import os
from typing import Dict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokenizer(tokenizer_path: str) -> Dict:
    """Load tokenizer from saved JSON file"""
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    return tokenizer_data

def decode_prediction(output: torch.Tensor, tokenizer_data: Dict) -> str:
    """Decode model output using CTC decoding and tokenizer"""
    # Get best path
    best_path = torch.argmax(output, dim=2).squeeze(0)
    
    # Convert indices to symbols
    id_to_symbol = {int(k): v for k, v in tokenizer_data['id_to_symbol'].items()}
    
    # CTC decoding (removing duplicates and blanks)
    previous = None
    decoded = []
    for idx in best_path:
        idx = idx.item()
        if idx != 0 and idx != previous:  # 0 is blank token
            decoded.append(id_to_symbol[idx])
        previous = idx
    
    return '\t'.join(decoded)

def predict_single_image(image_path: str, model: torch.nn.Module, tokenizer_data: Dict, device: str) -> str:
    """Predict symbols from a single image"""
    # Prepare image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    # Load and resize image
    image = Image.open(image_path)
    target_height = 128
    aspect_ratio = image.size[0] / image.size[1]
    target_width = int(target_height * aspect_ratio)
    image = image.resize((target_width, target_height))
    image = transform(image).unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        
    # Decode prediction
    prediction = decode_prediction(output, tokenizer_data)
    return prediction

def process_directory(input_dir: str, output_dir: str, model: torch.nn.Module, 
                     tokenizer_data: Dict, device: str):
    """Process all images in the input directory and save results to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in Path(input_dir).glob('*.jpg')]  # Adjust extension if needed
    total_files = len(image_files)
    logger.info(f"Found {total_files} images to process in {input_dir}")
    
    # Process each image
    for idx, image_path in enumerate(sorted(image_files), 1):
        try:
            # Get prediction
            prediction = predict_single_image(str(image_path), model, tokenizer_data, device)
            
            # Create output path
            output_path = Path(output_dir) / f"{image_path.stem}.semantic"
            
            # Save prediction
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prediction)
            
            logger.info(f"Processed {image_path.name} ({idx}/{total_files})")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict music symbols from cropped stave images')
    parser.add_argument('--input_dir', default='detection_results/cropped_staves',
                        help='Directory containing the cropped stave images')
    parser.add_argument('--output_dir', default='stave_semantic_results',
                        help='Directory to save the semantic results')
    parser.add_argument('--model_dir', default='./checkpoints/semantic/run_20241125_125147/models',
                        help='Directory containing the model and tokenizer')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Verify input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    # Set paths
    model_path = f"{args.model_dir}/best_model_epoch_75.pt"
    tokenizer_path = f"{args.model_dir}/tokenizer.json"
    
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    logger.info(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer_data = load_tokenizer(tokenizer_path)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = CRNN(len(tokenizer_data['symbol_to_id'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Successfully loaded model")
    
    # Process all images in directory
    process_directory(args.input_dir, args.output_dir, model, tokenizer_data, device)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    from train_crnn_semantic import CRNN
    main()

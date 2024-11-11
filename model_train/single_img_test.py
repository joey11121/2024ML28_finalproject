import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import logging
from typing import Tuple, List
import os

# Import necessary classes from training script
from train_crnn_torch2 import CRNN, MusicTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleImageTester:
    def __init__(self, model_path: str, data_dir: str, device: str = 'cuda'):
        self.device = device
        self.data_dir = data_dir
        
        # Initialize tokenizer
        self.tokenizer = MusicTokenizer()
        self.tokenizer.fit(data_dir)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    
    def _load_model(self, model_path: str) -> CRNN:
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CRNN(self.tokenizer.vocabulary_size).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded model from epoch {checkpoint['epoch']} "
                   f"with validation loss {checkpoint['val_loss']:.4f}")
        return model
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image"""
        # Load image
        image = Image.open(image_path)
        
        # Resize to fixed height while maintaining aspect ratio
        target_height = 128
        aspect_ratio = image.size[0] / image.size[1]
        target_width = int(target_height * aspect_ratio)
        image = image.resize((target_width, target_height))
        
        # Apply transforms
        image = self.transform(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    def decode_prediction(self, output: torch.Tensor) -> str:
        """Convert model output to symbol sequence"""
        # output shape: [1, T, C]
        pred_indices = output[0].argmax(dim=1)  # [T]
        
        # Remove repeated symbols and blanks
        filtered_seq = []
        prev_symbol = -1
        for symbol in pred_indices:
            symbol = symbol.item()
            if symbol != prev_symbol and symbol != 0:  # 0 is blank token
                filtered_seq.append(symbol)
            prev_symbol = symbol
            
        return self.tokenizer.decode(filtered_seq)
    
    def read_true_label(self, image_path: str) -> str:
        """Read the true label from the corresponding .agnostic file"""
        base_path = os.path.splitext(image_path)[0]
        label_path = f"{base_path}.agnostic"
        
        with open(label_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def test_single_image(self, image_path: str) -> Tuple[str, str, bool]:
        """Test model on a single image and compare with true label"""
        # Process image
        image = self.process_image(image_path)
        image = image.to(self.device)
        
        # Get true label
        true_label = self.read_true_label(image_path)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            prediction = self.decode_prediction(output)
        
        # Compare
        is_correct = prediction == true_label
        
        return prediction, true_label, is_correct

def main():
    # Configuration
    model_path = './checkpoints/best_model.pt'
    data_dir = '../primus'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create tester
    tester = SingleImageTester(model_path, data_dir, device)
    
    # Test a single image
    # Replace with your image path
    image_path = "../primus/000100146-1_1_1/000100146-1_1_1.png"
    
    prediction, true_label, is_correct = tester.test_single_image(image_path)
    
    # Print results
    logger.info(f"\nResults for {image_path}:")
    logger.info(f"\nPrediction:\n{prediction}")
    logger.info(f"\nTrue Label:\n{true_label}")
    logger.info(f"\nCorrect: {is_correct}")
    
    if not is_correct:
        # Split into symbols for easier comparison
        pred_symbols = prediction.split('\t')
        true_symbols = true_label.split('\t')
        
        logger.info("\nDetailed comparison:")
        for i, (pred, true) in enumerate(zip(pred_symbols, true_symbols)):
            if pred != true:
                logger.info(f"Position {i}: Predicted '{pred}' vs True '{true}'")

if __name__ == '__main__':
    main()

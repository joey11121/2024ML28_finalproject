import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import logging
from typing import Tuple, List
import os
import glob
import time

# Import necessary classes from training script
from train_crnn_agnostic import CRNN, MusicTokenizer

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
        # Error_rate = self.calculate_edit_distance(output, prediction)
        is_correct = prediction == true_label
        
        return prediction, true_label, is_correct


    def calculate_edit_distance(self, output_list: list, prediction_list: list) -> Tuple[float, List[str]]:
        m, n = len(prediction_list), len(output_list)
        
        # Initialize dp array and operations tracker
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        operations = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
            if i > 0:
                operations[i][0] = operations[i-1][0] + [f"Delete '{prediction_list[i-1]}' at position {i-1}"]
        
        for j in range(n + 1):
            dp[0][j] = j
            if j > 0:
                operations[0][j] = operations[0][j-1] + [f"Insert '{output_list[j-1]}' at position {j-1}"]
        
        # Fill dp table and track operations
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if prediction_list[i-1] == output_list[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    operations[i][j] = operations[i-1][j-1]
                else:
                    # Find minimum of three operations
                    delete_cost = dp[i-1][j] + 1
                    insert_cost = dp[i][j-1] + 1
                    substitute_cost = dp[i-1][j-1] + 1
                    
                    min_cost = min(delete_cost, insert_cost, substitute_cost)
                    dp[i][j] = min_cost
                    
                    # Track operation performed
                    if min_cost == delete_cost:
                        operations[i][j] = operations[i-1][j] + [
                            f"Delete '{prediction_list[i-1]}' at position {i-1}"
                        ]
                    elif min_cost == insert_cost:
                        operations[i][j] = operations[i][j-1] + [
                            f"Insert '{output_list[j-1]}' at position {j-1}"
                        ]
                    else:  # substitute
                        operations[i][j] = operations[i-1][j-1] + [
                            f"Replace '{prediction_list[i-1]}' with '{output_list[j-1]}' at position {i-1}"
                        ]
        
        # Calculate error rate
        max_length = max(m, n)
        error_rate = dp[m][n] / max_length if max_length > 0 else 0
        
        return error_rate, operations[m][n]
    def test_multiple_images(self, num_images: int = 8700) -> float:
        """Test last num_images images and calculate average SER"""
        # Open log file with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f'test_results_{timestamp}.txt'
        
        def log_print(message):
            """Helper function to print and write to file"""
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        
        total_edit_distance = 0
        total_symbols = 0
        correct_count = 0
        total_count = 0
        
        # Get list of all image files
        all_image_files = sorted(glob.glob(os.path.join(self.data_dir, "*/*.png")))
        total_files = len(all_image_files)
        start_idx = total_files - num_images
        test_images = all_image_files[start_idx:]
        
        # Write initial information
        log_print(f"\nTest Results - {timestamp}")
        log_print("="*50)
        log_print(f"Total files found: {total_files}")
        log_print(f"Testing last {num_images} images (index {start_idx} to {total_files-1})")
        log_print(f"Processing {len(test_images)} images...\n")
        
        for idx, image_path in enumerate(test_images):
            # Process image
            image = self.process_image(image_path)
            image = image.to(self.device)
            
            # Get true label
            true_label = self.read_true_label(image_path)
            true_symbols = true_label.split('\t')
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image)
                prediction = self.decode_prediction(output)
                pred_symbols = prediction.split('\t')
            
            # Calculate edit distance
            edit_dist, operations = self.calculate_edit_distance(true_symbols, pred_symbols)
            
            # Update statistics
            sequence_length = len(true_symbols)
            total_edit_distance += edit_dist * sequence_length
            total_symbols += sequence_length
            
            if prediction == true_label:
                correct_count += 1
            total_count += 1
            
            # Log progress periodically
            if (idx + 1) % 100 == 0:
                current_ser = (total_edit_distance / total_symbols) * 100
                current_accuracy = (correct_count / total_count) * 100
                log_print(f"Processed {idx + 1}/{len(test_images)} images")
                log_print(f"Current SER: {current_ser:.2f}%")
                log_print(f"Current Sequence Accuracy: {current_accuracy:.2f}%\n")
                
                # If there were errors in the last sample, log them
                if prediction != true_label:
                    log_print(f"Sample error analysis for {image_path}:")
                    log_print(f"Edit operations needed:")
                    for op in operations:
                        log_print(f"  {op}")
                    log_print("\n")
        
        # Calculate final metrics
        average_ser = (total_edit_distance / total_symbols) * 100
        sequence_accuracy = (correct_count / total_count) * 100
        
        # Log final results
        log_print("\nFinal Results:")
        log_print("="*50)
        log_print(f"Average Symbol Error Rate: {average_ser:.2f}%")
        log_print(f"Sequence Accuracy: {sequence_accuracy:.2f}%")
        log_print(f"Total sequences processed: {total_count}")
        log_print(f"Total symbols processed: {total_symbols}")
        log_print(f"Correctly predicted sequences: {correct_count}")
        log_print("\nResults saved to: {log_file}")
        
        return average_ser



def main():
    # Configuration
    model_path = './checkpoints/agnostic/best_model.pt'
    data_dir = '../primus'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create tester
    tester = SingleImageTester(model_path, data_dir, device)
    
    # Test last 8700 images
    average_ser = tester.test_multiple_images(num_images=8700)

if __name__ == '__main__':
    main()


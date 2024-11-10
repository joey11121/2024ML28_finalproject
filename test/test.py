import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
from typing import List, Tuple
import editdistance
from tqdm import tqdm

# Import necessary classes from training script
from train_crnn_torch import CRNN, MusicTokenizer, PriMuSDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OMRTester:
    def __init__(self, checkpoint_path: str, data_dir: str, vocab_dir: str = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_dir = data_dir
        
        # Initialize tokenizer and fit on training data directory (or provided vocab directory)
        self.tokenizer = MusicTokenizer()
        self.tokenizer.fit(vocab_dir or data_dir)  # Use vocab_dir if provided, otherwise data_dir
        
        # Load model with the correct vocabulary size
        self.model = CRNN(self.tokenizer.vocabulary_size).to(self.device)
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
    def decode_prediction(self, output: torch.Tensor) -> List[str]:
        """Convert model output to symbol sequences"""
        # output shape: (T, B, C)
        pred = output.permute(1, 0, 2)  # (B, T, C)
        pred = torch.argmax(pred, dim=2)  # (B, T)
        
        batch_results = []
        for sequence in pred:
            # Merge repeated symbols and remove blanks
            merged = []
            previous = -1
            for symbol in sequence:
                symbol = symbol.item()
                if symbol != previous and symbol != 0:  # 0 is blank token
                    merged.append(symbol)
                previous = symbol
            
            # Convert indices to symbols
            decoded = self.tokenizer.decode(merged)
            batch_results.append(decoded)
            
        return batch_results
    
    @torch.no_grad()
    def evaluate(self, batch_size: int = 16) -> Tuple[float, float]:
        """
        Evaluate model on test data
        Returns:
            sequence_accuracy: percentage of perfectly predicted sequences
            symbol_error_rate: average edit distance between predicted and true sequences
        """
        # Create test dataset and dataloader
        test_dataset = PriMuSDataset(self.data_dir, self.tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        self.model.eval()
        
        total_sequences = 0
        correct_sequences = 0
        total_edit_distance = 0
        total_length = 0
        
        for images, targets, input_lengths, target_lengths in tqdm(test_loader):
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)  # (B, T, C)
            
            # Decode predictions
            predicted_sequences = self.decode_prediction(outputs)
            
            # Convert target indices to symbols
            target_sequences = [
                self.tokenizer.decode(target[:length].tolist())
                for target, length in zip(targets, target_lengths)
            ]
            
            # Calculate metrics
            for pred, target in zip(predicted_sequences, target_sequences):
                total_sequences += 1
                if pred == target:
                    correct_sequences += 1
                    
                edit_dist = editdistance.eval(pred.split('\t'), target.split('\t'))
                total_edit_distance += edit_dist
                total_length += len(target.split('\t'))
        
        # Calculate metrics safely
        sequence_accuracy = (correct_sequences / total_sequences * 100) if total_sequences > 0 else 0.0
        symbol_error_rate = (total_edit_distance / total_length * 100) if total_length > 0 else 0.0
        
        return sequence_accuracy, symbol_error_rate

    def predict_single(self, image_path: str) -> str:
        """Predict symbols for a single image"""
        # Create dataset with single image
        dataset = PriMuSDataset(os.path.dirname(image_path), self.tokenizer)
        image = dataset._load_and_process_image(image_path)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            predicted = self.decode_prediction(output)[0]
        
        return predicted

def main():
    # Example usage
    checkpoint_path = "checkpoint_epoch_100.pt"  # Replace with your checkpoint path
    test_dir = "../primus_test"  # Replace with your test data directory
    train_dir = "../primus"  # Your training data directory (with complete vocabulary)
    
    # Initialize tester
    tester = OMRTester(
        checkpoint_path=checkpoint_path,
        data_dir = test_dir,
        vocab_dir=train_dir  # Use training directory to build complete vocabulary
    )
    
    # Evaluate model
    sequence_accuracy, symbol_error_rate = tester.evaluate()
    logger.info(f"Sequence Accuracy: {sequence_accuracy:.2f}%")
    logger.info(f"Symbol Error Rate: {symbol_error_rate:.2f}%")
    
    # Example of single image prediction
    #image_path = "../primus_test/example/example.png"  # Replace with your image path
    #if os.path.exists(image_path):
        #prediction = tester.predict_single(image_path)
        #logger.info(f"Predicted symbols: {prediction}")

if __name__ == "__main__":
    main()

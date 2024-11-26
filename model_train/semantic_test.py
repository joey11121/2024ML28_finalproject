import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import editdistance
import logging
import os
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import necessary classes from training script
from train_crnn_semantic import CRNN, PriMuSDataset, MusicTokenizer, collate_fn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str) -> Tuple[CRNN, MusicTokenizer]:
    """Load the trained model and tokenizer"""
    # Load tokenizer
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    tokenizer = MusicTokenizer()
    tokenizer.symbols = set(tokenizer_data['symbols'])
    tokenizer.symbol_to_id = tokenizer_data['symbol_to_id']
    tokenizer.id_to_symbol = {int(k): v for k, v in tokenizer_data['id_to_symbol'].items()}
    
    # Initialize and load model
    model = CRNN(tokenizer.vocabulary_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def decode_predictions(outputs: torch.Tensor, input_lengths: torch.Tensor, tokenizer: MusicTokenizer) -> List[str]:
    """Decode model outputs into symbol sequences"""
    predictions = []
    batch_size = outputs.size(1)
    
    # For each item in batch
    for b in range(batch_size):
        # Get best path
        output = outputs[:input_lengths[b], b]
        indices = torch.argmax(output, dim=1)
        
        # Collapse repeated symbols and remove blanks
        collapsed = []
        previous = -1
        for idx in indices:
            if idx != 0 and idx != previous:  # 0 is blank token
                collapsed.append(idx.item())
            previous = idx
        
        # Convert to symbols
        sequence = tokenizer.decode(collapsed)
        predictions.append(sequence)
    
    return predictions

def calculate_metrics(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """Calculate Symbol Error Rate and Accuracy"""
    total_edits = 0
    total_symbols = 0
    correct_sequences = 0
    
    for pred, target in zip(predictions, targets):
        pred_symbols = pred.split('\t')
        target_symbols = target.split('\t')
        
        # Calculate edit distance
        edits = editdistance.eval(pred_symbols, target_symbols)
        total_edits += edits
        total_symbols += len(target_symbols)
        
        # Check for perfect match
        if pred == target:
            correct_sequences += 1
    
    # Calculate metrics
    ser = (total_edits / total_symbols) * 100 if total_symbols > 0 else 100
    accuracy = (correct_sequences / len(predictions)) * 100 if predictions else 0
    
    return ser, accuracy

def evaluate_model(
    model: CRNN,
    test_loader: DataLoader,
    tokenizer: MusicTokenizer,
    device: str
) -> Tuple[float, float]:
    """Evaluate model on test dataset"""
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, targets, input_lengths, target_lengths = batch
            images = images.to(device)
            
            # Get model outputs
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            
            # Decode predictions
            predictions = decode_predictions(outputs.cpu(), input_lengths, tokenizer)
            
            # Decode targets
            target_sequences = []
            for target, length in zip(targets, target_lengths):
                target_seq = tokenizer.decode(target[:length].tolist())
                target_sequences.append(target_seq)
            
            all_predictions.extend(predictions)
            all_targets.extend(target_sequences)
    
    # Calculate metrics
    ser, accuracy = calculate_metrics(all_predictions, all_targets)
    
    return ser, accuracy

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = "./checkpoints/semantic"
    batch_size = 32
    splits_file = "dataset_splits_20241125_125149.txt"
    
    # Read test file paths from the splits file
    test_files = []
    reading_test = False
    
    with open(splits_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "TEST SET" in line:
                reading_test = True
                continue
            if reading_test:
                if line.strip() == "":  # Empty line indicates end of test set
                    break
                if not line.startswith('--'):  # Skip separator lines
                    test_files.append(line.strip())
    
    logger.info(f"Loaded {len(test_files)} test files from splits file")
    
    # Find the latest model checkpoint
    runs = os.listdir(checkpoint_dir)
    latest_run = max(runs)
    models_dir = os.path.join(checkpoint_dir, latest_run, "models")
    model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model")]
    latest_model = "best_model_epoch_181.pt"
    
    # Paths to model and tokenizer
    model_path = os.path.join(models_dir, latest_model)
    tokenizer_path = os.path.join(models_dir, "tokenizer.json")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device)
    
    # Create test dataset and loader directly from file paths
    test_dataset = PriMuSDataset(test_files, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    logger.info(f"Created test dataset with {len(test_dataset)} samples")
    
    # Evaluate model
    ser, accuracy = evaluate_model(model, test_loader, tokenizer, device)
    
    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"Symbol Error Rate (SER): {ser:.2f}%")
    logger.info(f"Sequence Accuracy: {accuracy:.2f}%")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
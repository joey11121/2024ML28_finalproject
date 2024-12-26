import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
import glob
import random
from typing import List, Dict, Tuple
import numpy as np
import time
from datetime import datetime, timedelta
import json
import matplotlib
matplotlib.use('Agg')  # Set this before importing pyplot
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Double ensure we're using Agg backend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        super(CRNN, self).__init__()
        
        # CNN layers with dropout
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)
        )
        
        # RNN layers with dropout
        self.rnn = nn.Sequential(
            nn.LSTM(2048, 256, bidirectional=True, batch_first=True, dropout=dropout_rate),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        conv = self.cnn(x)
        
        # Prepare for RNN
        batch_size, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.reshape(batch_size, width, channels * height)
        
        # RNN sequence learning
        rnn_input = conv
        for i, lstm in enumerate(self.rnn):
            rnn_output, _ = lstm(rnn_input)
            rnn_input = rnn_output
            
        # Classification
        output = self.classifier(rnn_output)
        return nn.functional.log_softmax(output, dim=2)

class MusicTokenizer:
    def __init__(self):
        self.symbols = set()
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        
    def fit(self, data_dir: str) -> None:
        """Build vocabulary from agnostic files in PrIMuS directory structure"""
        agnostic_files = glob.glob(os.path.join(data_dir, "*/*.agnostic"))
        logger.info(f"Found {len(agnostic_files)} agnostic files")
        
        # Collect unique symbols
        for file_path in agnostic_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    symbols = content.split('\t')
                    self.symbols.update(symbols)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        # Create mappings (add blank token at index 0)
        self.symbols = sorted(list(self.symbols))
        self.symbol_to_id = {s: i+1 for i, s in enumerate(self.symbols)}
        self.symbol_to_id['<blank>'] = 0
        self.id_to_symbol = {i+1: s for i, s in enumerate(self.symbols)}
        self.id_to_symbol[0] = '<blank>'
        
        logger.info(f"Vocabulary size: {len(self.symbols) + 1}")
        
    def encode(self, text: str) -> List[int]:
        """Convert tab-separated symbol sequence to integer indices"""
        symbols = text.strip().split('\t')
        return [self.symbol_to_id[s] for s in symbols if s]
    
    def decode(self, indices: List[int]) -> str:
        """Convert integer indices back to symbol sequence"""
        return '\t'.join([self.id_to_symbol[i] for i in indices if i != 0])
    
    @property
    def vocabulary_size(self) -> int:
        return len(self.symbol_to_id)

class PriMuSDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: MusicTokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        
        # Find all PNG files
        self.image_files = glob.glob(os.path.join(data_dir, "*/*.png"))
        
        # Randomly select 28600 samples if we have more
        if len(self.image_files) > 28600:
            random.seed(42)  # For reproducibility
            self.image_files = random.sample(self.image_files, 28600)
            logger.info(f"Randomly selected 28600 samples from {len(self.image_files)} total samples")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path)
        
        # Resize to fixed height while maintaining aspect ratio
        target_height = 128
        aspect_ratio = image.size[0] / image.size[1]
        target_width = int(target_height * aspect_ratio)
        image = image.resize((target_width, target_height))
        
        if self.transform:
            image = self.transform(image)
            
        # Load label
        base_path = os.path.splitext(img_path)[0]
        label_path = f"{base_path}.agnostic"
        
        with open(label_path, 'r', encoding='utf-8') as f:
            label_text = f.read().strip()
            label = torch.tensor(self.tokenizer.encode(label_text), dtype=torch.long)
            
        return image, label

def create_data_loaders(data_dir: str, tokenizer: MusicTokenizer, batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with the specified split sizes
    """
    # Create complete dataset
    dataset = PriMuSDataset(data_dir, tokenizer)
    
    # Calculate split sizes
    train_size = 20000
    val_size = 8600
    assert train_size + val_size == 28600, "Split sizes must sum to 28600"
    
    # Create train/val splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for variable width images and label lengths"""
    # Sort batch by image width (descending)
    batch.sort(key=lambda x: x[0].size(2), reverse=True)
    images, labels = zip(*batch)
    
    # Get max widths and lengths
    max_width = max(img.size(2) for img in images)
    max_length = max(label.size(0) for label in labels)
    
    # Pad images and labels
    padded_images = torch.zeros(len(images), 1, 128, max_width)
    padded_labels = torch.zeros(len(labels), max_length).long()
    input_lengths = []
    label_lengths = []
    
    for i, (image, label) in enumerate(zip(images, labels)):
        width = image.size(2)
        length = label.size(0)
        
        padded_images[i, :, :, :width] = image
        padded_labels[i, :length] = label
        input_lengths.append(width // 16)  # After CNN
        label_lengths.append(length)
    
    return padded_images, padded_labels, torch.tensor(input_lengths), torch.tensor(label_lengths)


class TrainingMonitor:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': [],
            'epoch_time': [],
            'total_time': []
        }
        self.start_time = time.time()
        
    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float, epoch_time: float):
        """Update training history and plot progress"""
        total_time = time.time() - self.start_time
        
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        self.history['total_time'].append(total_time)
        
        # Save history first
        self._save_history()
        
        # Then create plot
        try:
            self._plot_progress()
        except Exception as e:
            logger.warning(f"Failed to create plot: {e}")
        
        # Log time information
        self._log_time_info(epoch, epoch_time, total_time)
    
    def _plot_progress(self):
        """Create and save training progress plots"""
        # Clear any existing figures
        plt.close('all')
        
        # Create new figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot losses
        ax1.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(self.history['epoch'], self.history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        # Plot time per epoch
        ax3.plot(self.history['epoch'], self.history['epoch_time'])
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Time per Epoch')
        ax3.grid(True)
        
        # Save and close
        plt.tight_layout()
        try:
            fig.savefig('training_progress.png')
        finally:
            plt.close(fig)
    
    def _save_history(self):
        """Save training history to JSON file"""
        history_dict = {
            key: [float(val) if isinstance(val, (np.float32, np.float64)) else val 
                  for val in values]
            for key, values in self.history.items()
        }
        try:
            with open('training_history.json', 'w') as f:
                json.dump(history_dict, f)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def _log_time_info(self, epoch: int, epoch_time: float, total_time: float):
        """Log time information"""
        avg_epoch_time = sum(self.history['epoch_time']) / len(self.history['epoch_time'])
        estimated_remaining = avg_epoch_time * (200 - epoch)  # Assuming max 200 epochs
        
        logger.info(f'Time for epoch {epoch}: {timedelta(seconds=int(epoch_time))}')
        logger.info(f'Total training time: {timedelta(seconds=int(total_time))}')
        logger.info(f'Average time per epoch: {timedelta(seconds=int(avg_epoch_time))}')
        logger.info(f'Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}')


def train_model(
    data_dir: str, 
    num_epochs: int = 200, 
    batch_size: int = 64,
    device: str = 'cuda',
    patience: int = 20
):
    # Start timing total training
    total_start_time = time.time()
    
    # Initialize tokenizer and monitor
    tokenizer = MusicTokenizer()
    tokenizer.fit(data_dir)
    monitor = TrainingMonitor()
    
    # Log initial setup time
    setup_time = time.time() - total_start_time
    logger.info(f'Setup time: {timedelta(seconds=int(setup_time))}')
    
    # Create data loaders with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    train_loader, val_loader = create_data_loaders(
        data_dir, 
        tokenizer, 
        batch_size
    )
    
    # Initialize model and training components
    model = CRNN(tokenizer.vocabulary_size, dropout_rate=0.3).to(device)
    base_lr = 0.0001 * (64/16)**0.5
    optimizer = optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=7,
        verbose=True,
        min_lr=1e-6
    )
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        batch_times = []
        
        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(train_loader):
            batch_start_time = time.time()
            
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if batch_idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                    f'Train Loss: {loss.item():.4f}, '
                    f'LR: {current_lr:.6f}, '
                    f'Batch Time: {timedelta(seconds=int(batch_time))}'
                )
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update monitor
        monitor.update(epoch, avg_train_loss, avg_val_loss, current_lr, epoch_time)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'training_time': time.time() - total_start_time
            }, 'best_model.pt')
            logger.info(f'Saved new best model with validation loss: {avg_val_loss:.4f}')
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Log final timing information
    total_training_time = time.time() - total_start_time
    logger.info(f'Total training time: {timedelta(seconds=int(total_training_time))}')
    logger.info(f'Average time per epoch: {timedelta(seconds=int(total_training_time/(epoch+1)))}')
    logger.info(f'Average time per batch: {timedelta(seconds=int(avg_batch_time))}')

if __name__ == "__main__":
    data_dir = "../dataset"
    train_model(data_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

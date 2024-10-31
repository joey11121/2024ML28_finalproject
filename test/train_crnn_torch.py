
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
import glob
from typing import List, Dict, Tuple
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        
        # RNN layers
        self.rnn = nn.Sequential(
            nn.LSTM(2048, 256, bidirectional=True, batch_first=True),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction (B, 1, H, W) -> (B, 256, H/16, W/16)
        conv = self.cnn(x)
        
        # Prepare for RNN (B, 256, H/16, W/16) -> (B, W/16, 256*H/16)
        batch_size, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (B, W/16, 256, H/16)
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

def train_model(data_dir: str, num_epochs: int = 100, batch_size: int = 16, device: str = 'cuda'):
    # Initialize tokenizer
    tokenizer = MusicTokenizer()
    tokenizer.fit(data_dir)
    
    # Create dataset and dataloader
    dataset = PriMuSDataset(data_dir, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model and training components
    model = CRNN(tokenizer.vocabulary_size).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C) format for CTC
            
            # Calculate loss
            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    data_dir = "../primus"
    train_model(data_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

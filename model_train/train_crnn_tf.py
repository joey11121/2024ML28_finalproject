import tensorflow as tf
from keras import layers, Model
import os
import logging
import numpy as np
from datetime import datetime
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRNN(Model):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional layers(CNN)
        self.cnn = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        
        # Reshaping layer
        self.reshape = layers.Reshape((-1, 2048))  # 256 * 8
        
        # Bidirectional LSTM layers(rnn)
        self.rnn1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))
        self.rnn2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))
        
        # Output layer
        self.classifier = layers.Dense(num_classes)  # No +1 needed as blank is already in vocabulary
        
    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.reshape(x)
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.classifier(x)
        return tf.nn.softmax(x)

class MusicTokenizer:
    def __init__(self):
        self.symbols = set()
        self.symbol_to_id = {}
        self.id_to_symbol = {}
        
    def fit(self, data_dir):
        """Build vocabulary from agnostic files in PrIMuS directory structure"""
        # Find all agnostic files recursively
        agnostic_files = glob.glob(os.path.join(data_dir, "*/*.agnostic"))
        logger.info(f"Found {len(agnostic_files)} agnostic files")
        
        # Collect all unique symbols
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
        
        logger.info(f"Vocabulary size: {len(self.symbols) + 1}")  # +1 for blank
        
    def encode(self, text):
        """Convert tab-separated symbol sequence to integer indices"""
        symbols = text.strip().split('\t')
        return [self.symbol_to_id[s] for s in symbols if s]
    
    def decode(self, indices):
        """Convert integer indices back to symbol sequence"""
        return '\t'.join([self.id_to_symbol[i] for i in indices if i != 0])  # Skip blank token
    
    @property
    def vocabulary_size(self):
        return len(self.symbol_to_id)

def create_primus_dataset(data_dir, tokenizer, batch_size=16):
    """
    Create tf.data.Dataset from PrIMuS directory structure:
    primus/
        image_name1/
            image_name1.png
            image_name1.agnostic
        image_name2/
            image_name2.png
            image_name2.agnostic
        ...
    """
    def load_and_preprocess_image(file_path):
        # Read image
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img, channels=1)
        
        # Resize to fixed height while maintaining aspect ratio
        target_height = 128
        aspect_ratio = tf.shape(img)[1] / tf.shape(img)[0]
        target_width = tf.cast(tf.cast(target_height, tf.float32) * aspect_ratio, tf.int32)
        img = tf.image.resize(img, [target_height, target_width])
        
        # Normalize
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def load_label(file_path):
        # Get directory and base name from image path
        file_dir = tf.strings.regex_replace(file_path, '/[^/]+$', '')
        base_name = tf.strings.regex_replace(file_dir, '.+/', '')
        
        # Construct path to agnostic file
        label_path = tf.strings.join([file_dir, '/', base_name, '.agnostic'])
        
        # Read and process label
        label = tf.io.read_file(label_path)
        label = tf.strings.strip(label)
        
        # Convert to Python string for processing
        label_str = label.numpy().decode('utf-8')
        
        # Encode using tokenizer
        indices = tokenizer.encode(label_str)
        return tf.convert_to_tensor(indices, dtype=tf.int64)

    def process_path(file_path):
        image = load_and_preprocess_image(file_path)
        label = load_label(file_path)
        return image, label

    # Find all PNG files recursively
    pattern = os.path.join(data_dir, "*/*.png")
    image_files = tf.data.Dataset.list_files(pattern, shuffle=True)
    
    # Create dataset
    dataset = image_files.map(
        process_path,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch with padding
    padded_shapes = (
        [128, None, 1],  # Image shape (height is fixed, width is variable)
        [None]           # Label shape (variable length)
    )
    padding_values = (
        tf.constant(0, dtype=tf.float32),  # Image padding
        tf.constant(0, dtype=tf.int64)     # Label padding
    )
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, name='ctc_loss'):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        
        # Prepare input length (time steps after CNN)
        input_length = tf.fill([batch_size], tf.shape(y_pred)[1])
        
        # Calculate actual label lengths (excluding padding)
        label_length = tf.reduce_sum(tf.cast(y_true != 0, tf.int32), axis=1)
        
        # CTC loss requires these shapes:
        # y_true: (batch_size, max_string_length)
        # y_pred: (batch_size, max_time_steps, num_classes)
        # input_length: (batch_size, 1)
        # label_length: (batch_size, 1)
        
        return self.loss_fn(y_true, y_pred, input_length, label_length)
    
def verify_dataset(data_dir):
    """Utility function to verify the dataset structure and content"""
    logger.info("Verifying dataset structure...")
    
    # Check directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Count subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logger.info(f"Found {len(subdirs)} sample directories")
    
    # Check random samples
    import random
    sample_dirs = random.sample(subdirs, min(5, len(subdirs)))
    for dir_name in sample_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        png_file = os.path.join(dir_path, f"{dir_name}.png")
        agnostic_file = os.path.join(dir_path, f"{dir_name}.agnostic")
        
        logger.info(f"\nChecking {dir_name}:")
        logger.info(f"PNG exists: {os.path.exists(png_file)}")
        logger.info(f"Agnostic exists: {os.path.exists(agnostic_file)}")
        
        if os.path.exists(agnostic_file):
            with open(agnostic_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Sample content: {content[:100]}...")

    return True

def train_model(data_dir):
    # Initialize tokenizer and build vocabulary
    tokenizer = MusicTokenizer()
    tokenizer.fit(data_dir)
    
    # Create model and training components
    model = CRNN(tokenizer.vocabulary_size)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0) #ADAM
    loss_fn = CTCLoss()
    
    # Create dataset
    train_dataset = create_primus_dataset(data_dir, tokenizer, batch_size=16)
    
    # Rest of the training code remains the same...
    # (Keep your existing training loop code)

if __name__ == "__main__":
    data_dir = "../primus"
    train_model(data_dir)
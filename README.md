# Sheet Music Scanner Using CRNN & YOLO

## Introduction
A deep learning-based sheet music scanning and recognition system that converts sheet music images into playable music files.
This project  uses YOLO to detect the staff lines of each sheet music and CRNN to detect the information on the sheet music. 
## Project Structure
```
sheet-music-scanner/
├── preprocess/          # Dataset generation scripts
├── model_train/         # CRNN model training files
├── postprocess/         # Music conversion utilities
└── yolo_train/         # YOLO model training scripts
```

## Preprocessing
The preprocess directory contains scripts for dataset generation and labeling. Here is the procedure for creating your own dataset. 

### Data Collection
Create two folders for storing downloaded sheet music:
pdf/: PDF format sheet music from Musescore
mxl/: MXL format sheet music from Musescore

### Data Processing Pipeline
* split_measure.py: Splits sheet music into individual measures in .musicxml format
* xml2pdf.py: Converts split measures from .musicxml to .pdf format
* crop_image.py: Processes .pdf files to create standardized 128px height .png images
* semantic.py: Generates labels for each image using split musicxml files, outputs .semantic files
You wiil get all of the .png files and the label files, and you have to put each of them into the same folder. Each folder should contain a single image and its corresponding label file. Finally, remember to create a dataset folder and move all of the folders with image and the label file to the dataset folder. 

## YOLO Model Training
The `model_train` directory includes CRNN (Convolutional Recurrent Neural Network) implementation:
- Model architecture details
- Training procedures
- Evaluation metrics
- Pretrained models (if available)

### Postprocessing
The `postprocess` directory contains utilities for converting model outputs to music:
- Label to music conversion process
- Output format specifications
- Audio generation details

### YOLO Training
The `yolo_train` directory contains object detection model implementation:
- YOLO model specifications
- Training configuration
- Detection targets and classes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sheet-music-scanner.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```python
# Example command for preprocessing
python preprocess/prepare_data.py --input_dir /path/to/sheets --output_dir /path/to/dataset
```

### Model Training
```python
# Example command for training the CRNN model
python model_train/train.py --config config/crnn_config.yaml
```

### Music Generation
```python
# Example command for converting sheet music to audio
python postprocess/generate_music.py --input image.jpg --output music.midi
```

## Results
- Include sample results
- Performance metrics
- Example outputs

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Other dependencies (list them here)

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Choose appropriate license]

## Citation
If you use this project in your research, please cite:
```
@article{your_reference,
  title={Sheet Music Scanner},
  author={Your Name},
  year={2024}
}
```

## Contact
- Your Name
- Email
- Project Link: [https://github.com/yourusername/sheet-music-scanner](https://github.com/yourusername/sheet-music-scanner)

## Acknowledgments
- List any acknowledgments
- Credit external resources
- Mention inspirations

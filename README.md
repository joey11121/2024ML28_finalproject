# Sheet Music Scanner Using CRNN & YOLO

## Introduction
A deep learning-based sheet music scanning and recognition system that converts sheet music images into playable music files.
This project  uses YOLO to detect the staff lines of each sheet music and CRNN to detect the information on the sheet music. 

## Environment Setup
### Installation

```bash
# Clone the repository
git clone https://github.com/joey11121/sheet-music-scanner.git
```
### WSL & PyTorch Setup
We conducted our experiment mostly on WSL Ubuntu20.04. We used PyTorch. Here is a note for the setup of WSL and PyTorch. 
https://www.notion.so/12c40750f6e180789d30d0e918050818
### Package Dependency
The project requires two separate Conda environments for CRNN and YOLO models. For CRNN model training and data preprocessing, create an environment with Python 3.10 and install dependencies from `crnn_requirements.txt`. For YOLO model training, create a separate environment with Python 3.8 and install dependencies from `yolo_requirements.txt`. Use the following commands:

```bash
# CRNN Environment
conda create -n crnn_torch python=3.10
conda activate crnn_torch
pip install -r crnn_requirements.txt

# YOLO Environment  
conda create -n yolo_env python=3.8
conda activate yolo_env
pip install -r yolo_requirements.txt
```
## Project Structure
```
sheet-music-scanner/
├── preprocess/          # Dataset generation scripts
├── model_train/         # CRNN model training files
├── postprocess/         # Music conversion utilities
└── yolo_train/         # YOLO model training scripts
```

## Data Preprocessing
The preprocess directory contains scripts for crnn dataset generation and labeling. Here is the procedure for creating your own dataset. 

### Data Collection
Create two folders for storing downloaded sheet music:
pdf/: PDF format sheet music from Musescore
mxl/: MXL format sheet music from Musescore

### Data Processing Pipeline
* split_measure.py: Splits sheet music into individual measures in .musicxml format
* xml2pdf.py: Converts split measures from .musicxml to .pdf format
* crop_image.py: Processes .pdf files to create standardized 128px height .png images
* semantic.py: Generates labels for each image using split musicxml files, outputs .semantic files
* package.py: Organizes the dataset by creating folders where each folder contains one image file along with its corresponding label file
* addnoise.py: Add the noise to the dataset 

You will get all of the .png files and the label files, and you have to put each of them into the same folder. Each folder should contain a single image and its corresponding label file. Finally, remember to create a dataset folder and move all of the folders with image and the label file to the dataset folder. You can run the addnoise.py to add the noise on the image dataset. 
To access our example dataset, you can follow the link here.
https://drive.google.com/file/d/1RUyaMTYw3pbHBUAPmGhC7juB5oL9XsM0/view?usp=drive_link


## YOLO Model Training

The `yolo_train` directory contains YOLO model implementation for staff detection. Below is the directory structure
```
yolo_train/
├── data.yaml       # Dataset configuration
├── Staff_detector.py       # StaffDetection Class
├── yolo_train.py         # Dataset configuration
├── yolo_test.py          # Model evaluation
└── yolo_crop.py          # Bounding Box Evaluation
```
#### Setup
1. Prepare dataset in YOLO format. Our dataset is created using Roboflow, a website tool for YOLO annotation.
Link for our YOLO dataset: https://app.roboflow.com/final-project-51vom/final-project-8fk6m/models
Link for our YOLO dataset in Google drive: https://drive.google.com/drive/folders/1-Ix5zgGv9lMFO_YDPh75hWNG5DOYiHCm?usp=sharing
You can create your own YOLO dataset in this website as well 
2. After you annnotate, please include `data.yaml` configuration and the dataset in the directory:
```yaml
path: /path/to/dataset  # Dataset root directory
train: train/images     # Train images directory
val: val/images         # Validation images directory
test: test/images       # Test images directory

names:
  0: staff_line         # Class names
```

The dataset should be placed under the yolo_train directory. 

#### Training
```bash
python yolo_train.py
```

#### Testing
```bash
# Test on single image
python yolo_crop.py --image_path /path/to/image.jpg

# Evaluate on test set
python yolo_test.py
```




## CRNN Model Training
The `model_train` directory includes CRNN (Convolutional Recurrent Neural Network) implementation. Below is the directory structure:
```
model_train/
├── train_crnn_semantic.py # Train the crnn model with semantic labeling as we have done. 
├── train_crnn_agnostic.py # Train the crnn model with agnostic labeling as we have done. 
├── test_agnostic.py # Train the crnn model with agnostic labeling as we have done. 
├── semantic_test.py # Evaluate the model by the semantic labeling scheme with SER. 
├── checkpoints # The folder for saving model checkpoints
├── training_progress # Save the training records

```
Run the train_crnn_semantic.py with your own dataset and parameters by the following shell command:
```bash
# Set environment variables
export DATA_PATH=/path/to/dataset

# Run training with custom parameters
python train_crnn.py \
    --data_dir $DATA_PATH \
    --num_epochs 200 \
    --batch_size 48 \
    --patience 25 \
    --device cuda
```
The file will split the dataset into training, valid, and test. The data list will be saved in the dataset_splits_{datetime}.txt. It helps user to identify the test dataset in the evaluation stage. To evaluate the CRNN model after training: 
```bash
# Remeber to change the file name of the dataset_split_{datetime}.txt, model checkpoint, and the directory of your testdataset. 
python semantic_test.py
```

#### Parameters
* `--data_dir`: Path to dataset directory
* `--num_epochs`: Number of training epochs (default: 200)
* `--batch_size`: Batch size for training (default: 48)
* `--patience`: Early stopping patience (default: 25)
* `--device`: Training device ('cuda' or 'cpu')





## Postprocessing
The `postprocess` directory contains utilities for converting model outputs to music. 
If you want to convert the label to the music file(.mei). You can run the following command. 

```bash
python semantic2mei.py [--input_dir DIR] [--output_dir DIR] [--bpm TEMPO] [--num_parts PARTS]
```

Default values:
- input_dir: 'stave_semantic_results', the result of each staff line generated by CRNN
- output_dir: 'midi_results', for storing .mei
- bpm: 120.0
- num_parts: 1, set the number of voice part 

Example usage:
```bash
python semantic2mei.py --bpm 140 --num_parts 4
```

## User Usage Simulation
The user may want to use the image to transform it to music. The directory final_demo contains all of the files to achieve the task. 
1. use the yolo_crop.py to crop the image
2. Use the score_yolo_crnn.py to get the label. It will iterate through the cropped imagse of the input image and get all the label from all of the stafflines from the image. 
3. use semantic2mei.py to get the music

Example usage:
```bash
python yolo_crop.py --image [--input_image PATH]
python score_yolo_crnn.py
python semantic2mei.py [--input_dir DIR] [--output_dir DIR] [--bpm TEMPO] [--num_parts PARTS]
```

## Contact
- Liang-Yu, Cheng
- joey60209@gmail.com
- Project Link: [https://github.com/joey11121/2024ML28_finalproject.git](https://github.com/joey11121/2024ML28_finalproject.git)


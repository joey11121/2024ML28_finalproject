from ultralytics import YOLO
import cv2
import torch

class StaffDetector:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8s.pt')
            
    def detect(self, image):
        """
        Detect musical staff lines in an image
        Returns: List of bounding boxes with confidence scores
        """
        results = self.model(image, conf=0.35)[0]
        staff_boxes = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            staff_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        # Sort boxes by y-coordinate (top to bottom)
        staff_boxes.sort(key=lambda x: x['bbox'][1])
        
        return staff_boxes

def train_staff_detector(data_yaml_path: str = "data.yaml", epochs: int = 100):
    """Train YOLOv8 model for musical staff detection"""
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=512,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=20,  # Early stopping
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        project='staff_detection',
        name='train'
    )
    
    return results

import yaml
import os
from staff_detector import StaffDetector
from ultralytics.utils.metrics import bbox_iou
import numpy as np
import cv2
from datetime import datetime
import torch

def visualize_detections(image_path, staff_boxes, output_dir):
    """Draw boxes on image and save as PNG"""
    # Read image
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    # Draw each detection
    for staff in staff_boxes:
        bbox = staff['bbox']
        conf = staff['confidence']
        
        # Draw rectangle
        cv2.rectangle(
            vis_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2
        )
        
        # Add confidence score
        cv2.putText(
            vis_image,
            f"{conf:.2f}",
            (int(bbox[0]), int(bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{base_name}_detected_{timestamp}.png")
    
    # Save image
    cv2.imwrite(output_path, vis_image)
    print(f"Saved visualization to: {output_path}")
    
    return output_path

def evaluate_detection(detector, test_dir, output_dir):
    """Evaluate detector performance and save visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics storage
    all_ious = []
    confidences = []
    detection_counts = []
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} test images")
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Get detections
        staff_boxes = detector.detect(image)
        detection_counts.append(len(staff_boxes))
        
        # Save visualization
        visualize_detections(img_path, staff_boxes, output_dir)
        
        # Collect confidences
        for staff in staff_boxes:
            confidences.append(staff['confidence'])
        
        # Check for ground truth labels
        label_dir = os.path.join(os.path.dirname(test_dir), 'labels')
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert normalized coordinates to absolute
                    img_height, img_width = image.shape[:2]
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    
                    gt_box = [x1, y1, x2, y2]
                    
                    # Calculate IoU with each detection
                    for staff in staff_boxes:
                        pred_box = staff['bbox']
                        iou = bbox_iou(
                            torch.tensor([pred_box]),
                            torch.tensor([gt_box])
                        ).item()
                        all_ious.append(iou)
    
    # Calculate metrics
    avg_iou = np.mean(all_ious) if all_ious else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    avg_detections = np.mean(detection_counts)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number of test images: {len(image_files)}")
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Detections per Image: {avg_detections:.1f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"Test Images: {len(image_files)}\n")
        f.write(f"Average IoU: {avg_iou:.3f}\n")
        f.write(f"Average Confidence: {avg_confidence:.3f}\n")
        f.write(f"Average Detections per Image: {avg_detections:.1f}\n")
    
    return {
        'avg_iou': avg_iou,
        'avg_confidence': avg_confidence,
        'avg_detections': avg_detections,
        'num_images': len(image_files)
    }

if __name__ == "__main__":
    # Load model
    model_path = 'staff_detection/train2/weights/best.pt'
    detector = StaffDetector(model_path)
    
    # Set directories
    test_dir = os.path.join(os.path.dirname(os.path.abspath('data.yaml')), 'test', 'images')
    output_dir = "test_visualizations"
    
    print(f"Looking for test images in: {test_dir}")
    print(f"Saving visualizations to: {output_dir}")
    
    # Run evaluation
    try:
        metrics = evaluate_detection(detector, test_dir, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the test directory exists and contains images.")
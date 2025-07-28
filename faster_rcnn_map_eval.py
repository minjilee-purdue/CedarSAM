import json
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import os
from collections import defaultdict
import matplotlib.pyplot as plt

class mAPCalculator:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        Boxes format: [x1, y1, x2, y2]
        """
        # Convert from [x1, y1, x2, y2] format if needed
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        intersect_x1 = max(x1_min, x2_min)
        intersect_y1 = max(y1_min, y2_min)
        intersect_x2 = min(x1_max, x2_max)
        intersect_y2 = min(y1_max, y2_max)
        
        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return 0.0
            
        intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        
        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def load_ground_truth(self, json_file_path):
        """
        Load ground truth annotations from JSON file.
        For ERC detection, all annotations are treated as class 1 (ERC trees).
        """
        with open(json_file_path, 'r') as f:
            gt_data = json.load(f)
        
        # Convert to standardized format - all objects are ERC trees (class 1)
        ground_truth = {}
        total_objects = 0
        
        for image_name, annotations in gt_data.items():
            boxes = []
            for ann in annotations:
                # Convert [x1, y1, x2, y2] format
                bbox = ann['bbox']
                boxes.append({
                    'bbox': bbox,
                    'class_id': 1,  # All objects are ERC trees (class 1, background is 0)
                    'original_layer': ann['layer'],  # Keep original layer info for reference
                    'hex_color': ann['hex_color']
                })
                total_objects += 1
            ground_truth[image_name] = boxes
        
        print(f"Loaded ground truth: {len(ground_truth)} images with {total_objects} ERC trees total")
        return ground_truth
    
    def run_inference(self, model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Run inference on a single image using your trained Faster R-CNN model.
        """
        model.eval()
        model = model.to(device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep_indices = scores >= self.confidence_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    def calculate_ap_for_class(self, predictions, ground_truths, class_id):
        """
        Calculate Average Precision for a specific class.
        """
        # Collect all predictions and ground truths for this class
        all_predictions = []
        all_ground_truths = []
        
        for image_name in predictions.keys():
            # Get predictions for this class
            pred = predictions[image_name]
            class_mask = pred['labels'] == class_id
            if np.any(class_mask):
                for i, box in enumerate(pred['boxes'][class_mask]):
                    all_predictions.append({
                        'image': image_name,
                        'bbox': box,
                        'score': pred['scores'][class_mask][i],
                        'matched': False
                    })
            
            # Get ground truths for this class
            if image_name in ground_truths:
                gt = ground_truths[image_name]
                for ann in gt:
                    if ann['class_id'] == class_id:
                        all_ground_truths.append({
                            'image': image_name,
                            'bbox': ann['bbox'],
                            'matched': False
                        })
        
        if len(all_predictions) == 0:
            print(f"No predictions found for class {class_id}")
            return 0.0
        
        if len(all_ground_truths) == 0:
            print(f"No ground truths found for class {class_id}")
            return 0.0
        
        print(f"Class {class_id}: {len(all_predictions)} predictions, {len(all_ground_truths)} ground truths")
        
        # Sort predictions by confidence score (descending)
        all_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate precision and recall
        true_positives = np.zeros(len(all_predictions))
        false_positives = np.zeros(len(all_predictions))
        
        # Group ground truths by image for efficient lookup
        gt_by_image = defaultdict(list)
        for gt in all_ground_truths:
            gt_by_image[gt['image']].append(gt)
        
        for i, pred in enumerate(all_predictions):
            # Find the best matching ground truth box
            best_iou = 0
            best_gt = None
            
            for gt in gt_by_image[pred['image']]:
                if not gt['matched']:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            
            # Check if it's a true positive
            if best_iou >= self.iou_threshold and best_gt is not None:
                true_positives[i] = 1
                best_gt['matched'] = True
            else:
                false_positives[i] = 1
        
        # Calculate cumulative precision and recall
        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-10)
        recall = cumulative_tp / (len(all_ground_truths) + 1e-10)
        
        # Calculate Average Precision using the 11-point interpolation method
        ap = self.calculate_average_precision(precision, recall)
        
        return ap
    
    def calculate_average_precision(self, precision, recall):
        """
        Calculate Average Precision using 11-point interpolation.
        """
        # Add points at recall 0 and 1
        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Calculate AP using 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        ap = 0
        
        for threshold in recall_thresholds:
            # Find the maximum precision for recall >= threshold
            indices = recall >= threshold
            if np.any(indices):
                ap += np.max(precision[indices])
        
        return ap / 11
    
    def calculate_map_multiple_iou(self, model, ground_truth_file, images_directory, 
                                   iou_thresholds=None):
        """
        Calculate mAP across multiple IoU thresholds (COCO-style evaluation).
        """
        if iou_thresholds is None:
            # COCO evaluation: IoU from 0.5 to 0.95 with step 0.05
            iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
        
        # Load ground truth
        ground_truths = self.load_ground_truth(ground_truth_file)
        
        # Run inference on all images (only once)
        predictions = {}
        image_files = [f for f in os.listdir(images_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Running inference on {len(image_files)} images...")
        processed_count = 0
        
        for image_file in image_files:
            image_name = os.path.splitext(image_file)[0]
            if image_name in ground_truths:  # Only process images with ground truth
                image_path = os.path.join(images_directory, image_file)
                try:
                    pred = self.run_inference(model, image_path)
                    predictions[image_name] = pred
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        print(f"Successfully processed {processed_count} images")
        
        if not predictions:
            print("No predictions generated. Check your image paths and model.")
            return 0.0, {}, {}, predictions
        
        # Calculate AP for each IoU threshold
        ap_results = {}
        print(f"\nCalculating AP for ERC trees across {len(iou_thresholds)} IoU thresholds...")
        
        for iou_thresh in iou_thresholds:
            # Temporarily set the IoU threshold
            original_iou = self.iou_threshold
            self.iou_threshold = iou_thresh
            
            ap = self.calculate_ap_for_class(predictions, ground_truths, 1)
            ap_results[iou_thresh] = ap
            
            print(f"mAP@{iou_thresh:.2f}: {ap:.4f}")
            
            # Restore original IoU threshold
            self.iou_threshold = original_iou
        
        # Calculate overall mAP (average across all IoU thresholds)
        overall_map = np.mean(list(ap_results.values()))
        
        return overall_map, ap_results, {1: overall_map}, predictions
    
    def calculate_map(self, model, ground_truth_file, images_directory):
        """
        Calculate mean Average Precision (mAP) for all classes at IoU=0.5.
        """
        # Load ground truth
        ground_truths = self.load_ground_truth(ground_truth_file)
        
        # Run inference on all images
        predictions = {}
        image_files = [f for f in os.listdir(images_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Running inference on {len(image_files)} images...")
        processed_count = 0
        
        for image_file in image_files:
            image_name = os.path.splitext(image_file)[0]
            if image_name in ground_truths:  # Only process images with ground truth
                image_path = os.path.join(images_directory, image_file)
                try:
                    pred = self.run_inference(model, image_path)
                    predictions[image_name] = pred
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        print(f"Successfully processed {processed_count} images")
        
        if not predictions:
            print("No predictions generated. Check your image paths and model.")
            return 0.0, {}, predictions
        
        # For binary classification (ERC detection), we only calculate AP for class 1 (ERC trees)
        # Class 0 is background and not evaluated
        class_aps = {}
        
        print(f"\nCalculating AP for ERC trees (class 1)...")
        ap = self.calculate_ap_for_class(predictions, ground_truths, 1)
        class_aps[1] = ap
        print(f"ERC Trees (class 1): AP = {ap:.4f}")
        
        # For binary detection, mAP is just the AP of the positive class
        map_score = ap
        
        return map_score, class_aps, predictions

def load_model_with_correct_classes(model_path):
    """
    Load the model and automatically detect the number of classes from the saved weights.
    """
    # Load the saved state dict to inspect the model structure
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract number of classes from the classifier weight shape
    cls_score_weight = checkpoint['roi_heads.box_predictor.cls_score.weight']
    num_classes = cls_score_weight.shape[0]  # This includes background class
    
    print(f"Detected {num_classes} classes in the saved model (including background)")
    
    # Create model with correct number of classes
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # Modify the classifier head to match the saved model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Load the weights
    model.load_state_dict(checkpoint)
    
    return model, num_classes

# Usage example
def main():
    # Load your trained model with automatic class detection
    model_path = '/home/minjilee/Desktop/cnn_july22/r-cnn/faster_rcnn_erc_epoch10.pth'
    model, num_classes = load_model_with_correct_classes(model_path)
    
    print(f"Loaded model with {num_classes} classes (0=background, 1=ERC)")
    
    # Initialize mAP calculator
    map_calculator = mAPCalculator(iou_threshold=0.5, confidence_threshold=0.3)
    
    # Calculate mAP across multiple IoU thresholds (COCO-style)
    ground_truth_file = '/home/minjilee/Desktop/cnn_july22/r-cnn/extracted_bboxes.json'
    images_directory = '/home/minjilee/Desktop/combined_0624/jpg'
    
    print("="*60)
    print("COCO-STYLE EVALUATION (mAP@0.5:0.95)")
    print("="*60)
    
    overall_map, ap_per_iou, class_aps, predictions = map_calculator.calculate_map_multiple_iou(
        model, ground_truth_file, images_directory
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"mAP@0.5:0.95 (COCO-style): {overall_map:.4f}")
    print(f"mAP@0.5: {ap_per_iou[0.5]:.4f}")
    
    # Find the closest value to 0.75 if exact match doesn't exist
    closest_75 = min(ap_per_iou.keys(), key=lambda x: abs(x - 0.75))
    print(f"mAP@0.75: {ap_per_iou[closest_75]:.4f}")
    print(f"{'='*60}")
    
    # Additional statistics
    total_predictions = sum(len(pred['boxes']) for pred in predictions.values())
    total_ground_truths = sum(len(gt) for gt in map_calculator.load_ground_truth(ground_truth_file).values())
    
    print(f"\nDetection Statistics:")
    print(f"Total images processed: {len(predictions)}")
    print(f"Total predictions made: {total_predictions}")
    print(f"Total ground truth objects: {total_ground_truths}")
    print(f"Average predictions per image: {total_predictions/len(predictions):.2f}")
    print(f"Average ground truth objects per image: {total_ground_truths/len(predictions):.2f}")
    
    # Create comprehensive visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: mAP across different IoU thresholds
    iou_thresholds = list(ap_per_iou.keys())
    ap_values = list(ap_per_iou.values())
    
    ax1.plot(iou_thresholds, ap_values, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('IoU Threshold')
    ax1.set_ylabel('Average Precision (AP)')
    ax1.set_title('AP vs IoU Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Highlight key thresholds
    key_thresholds = [0.5, closest_75, 0.9]
    for iou in key_thresholds:
        if iou in ap_per_iou:
            ax1.axvline(x=iou, color='red', linestyle='--', alpha=0.5)
            ax1.text(iou, ap_per_iou[iou] + 0.02, f'{ap_per_iou[iou]:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Summary bar chart
    summary_metrics = {
        'mAP@0.5': ap_per_iou[0.5],
        'mAP@0.75': ap_per_iou[closest_75],
        'mAP@0.5:0.95': overall_map
    }
    
    bars = ax2.bar(summary_metrics.keys(), summary_metrics.values(), 
                   color=['green', 'orange', 'blue'], alpha=0.7)
    ax2.set_ylabel('Average Precision')
    ax2.set_title('ERC Detection Performance Summary')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, summary_metrics.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results table
    print(f"\n{'='*50}")
    print("DETAILED RESULTS BY IoU THRESHOLD:")
    print(f"{'='*50}")
    print(f"{'IoU Threshold':<15} {'AP':<10}")
    print(f"{'-'*25}")
    for iou_thresh, ap in ap_per_iou.items():
        print(f"{iou_thresh:<15.2f} {ap:<10.4f}")
    
    return overall_map, ap_per_iou, predictions

if __name__ == "__main__":
    main()
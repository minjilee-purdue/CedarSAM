# fp_fn_analysis.py
import os
import json
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image

# Load model directly from .pth
model_path = "/home/minjilee/Desktop/cnn_july22/r-cnn/faster_rcnn_erc.pth"
def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Helper function to draw bounding boxes
def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Analysis function for a single image
def analyze_and_visualize(gt_json, model, image_path, output_path, iou_thresh=0.5):
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)

    if isinstance(gt_data, dict) and 'objects' in gt_data:
        gt_boxes = torch.tensor([obj['bbox'] for obj in gt_data['objects']])
    else:
        gt_boxes = torch.empty((0, 4))

    # Get prediction from model
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)[0]

    pred_boxes = output['boxes']
    pred_scores = output['scores']
    keep = pred_scores >= 0.5
    pred_boxes = pred_boxes[keep]

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        iou = torch.zeros((len(gt_boxes), len(pred_boxes)))
    else:
        iou = box_iou(gt_boxes, pred_boxes)

    matched_gt = set()
    matched_pred = set()
    for i in range(iou.size(0)):
        for j in range(iou.size(1)):
            if iou[i, j] >= iou_thresh:
                matched_gt.add(i)
                matched_pred.add(j)

    tp_boxes = [pred_boxes[j].tolist() for j in matched_pred]
    fp_boxes = [pred_boxes[j].tolist() for j in range(len(pred_boxes)) if j not in matched_pred]
    fn_boxes = [gt_boxes[i].tolist() for i in range(len(gt_boxes)) if i not in matched_gt]

    # Draw and save image
    img = cv2.imread(image_path)
    if img is not None:
        draw_boxes(img, tp_boxes, (0, 255, 0), 'TP')
        draw_boxes(img, fp_boxes, (0, 0, 255), 'FP')
        draw_boxes(img, fn_boxes, (255, 0, 0), 'FN')
        cv2.imwrite(output_path, img)

    TP, FP, FN = len(tp_boxes), len(fp_boxes), len(fn_boxes)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'Image Name': os.path.basename(image_path),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-score': round(f1, 4),
    }

if __name__ == "__main__":
    gt_dir = "/home/minjilee/Desktop/combined_0624/location"
    img_dir = "/home/minjilee/Desktop/combined_0624/jpg"
    out_dir = "/home/minjilee/Desktop/cnn_july22/r-cnn/fpfn_results"
    os.makedirs(out_dir, exist_ok=True)

    model = get_model(num_classes=2)

    summary_data = []
    for fname in tqdm(os.listdir(gt_dir)):
        if not fname.endswith(".json"):
            continue

        base_name = os.path.splitext(fname)[0].replace("_location", "")
        img_name = base_name + ".jpg"

        gt_json_path = os.path.join(gt_dir, fname)
        image_path = os.path.join(img_dir, img_name)
        out_path = os.path.join(out_dir, f"fpfn_{img_name}")

        if not os.path.exists(image_path):
            print(f"Missing image: {img_name}")
            continue

        try:
            result = analyze_and_visualize(gt_json_path, model, image_path, out_path)
        except Exception as e:
            print(f"Failed on {fname}: {e}")
            result = {
                'Image Name': img_name,
                'TP': 0,
                'FP': 0,
                'FN': 0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-score': 0.0,
            }

        summary_data.append(result)

    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(out_dir, "/home/minjilee/Desktop/cnn_july22/r-cnn/fpfn_summary.csv"), index=False)
    print("Saved summary to fpfn_summary.csv")
# faster-rcnn.py

import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou

# ---------------------- #
# 1. Custom JSON Dataset
# ---------------------- #
class ERCDataset_JSON(Dataset):
    def __init__(self, img_dir, json_dir, transforms=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        file_id = os.path.splitext(img_filename)[0]
        json_path = os.path.join(self.json_dir, f"{file_id}_location.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        boxes = []
        labels = []
        for obj in data.get("objects", []):
            bbox = obj["bbox"]
            boxes.append(bbox)
            labels.append(1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_files)

# ---------------------- #
# 2. Model
# ---------------------- #
def get_model(num_classes=2):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    return FasterRCNN(backbone, num_classes=num_classes)

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------- #
# 3. Train One Epoch
# ---------------------- #
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

# ---------------------- #
# 4. Validation
# ---------------------- #
def validate(model, dataloader, device, iou_thresh=0.5):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(device)
                pred_boxes = output["boxes"][output["scores"] > 0.5]
                if len(pred_boxes) == 0:
                    total_fn += len(gt_boxes)
                    continue
                ious = box_iou(pred_boxes, gt_boxes)
                max_iou, _ = ious.max(dim=1)
                tp = (max_iou > iou_thresh).sum().item()
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp
                total_tp += tp
                total_fp += fp
                total_fn += fn
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return f1

# ---------------------- #
# 5. Visualization
# ---------------------- #
def visualize_predictions(model, dataset, device, save_dir="vis_results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for i in range(min(5, len(dataset))):
        img, _ = dataset[i]
        with torch.no_grad():
            pred = model([img.to(device)])[0]
        img_vis = F.to_pil_image(img)
        draw = ImageDraw.Draw(img_vis)
        for box, score in zip(pred["boxes"], pred["scores"]):
            if score > 0.5:
                draw.rectangle(box.tolist(), outline="red", width=3)
                draw.text((box[0], box[1]), f"{score:.2f}", fill="red")
        img_vis.save(os.path.join(save_dir, f"pred_{i}.jpg"))

# ---------------------- #
# 6. Plot Loss and F1
# ---------------------- #
def plot_metrics(train_losses, val_f1s):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.title("Training Loss & Validation F1")
    plt.savefig("metrics_plot.png")
    plt.close()

# ---------------------- #
# 7. IoU Histogram
# ---------------------- #
def confusion_matrix_iou(model, dataloader, device):
    import seaborn as sns
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(device)
                pred_boxes = output["boxes"][output["scores"] > 0.5]
                if len(pred_boxes) and len(gt_boxes):
                    ious = box_iou(pred_boxes, gt_boxes)
                    iou_scores.extend(ious.cpu().numpy().flatten())
    plt.figure(figsize=(6,4))
    sns.histplot(iou_scores, bins=20, kde=True)
    plt.title("IoU Score Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.savefig("iou_distribution.png")
    plt.close()
    print("Saved IoU histogram to iou_distribution.png")

# ---------------------- #
# 8. Save Predictions as JSON
# ---------------------- #
def save_predictions_to_json(model, dataset, device, output_path="predictions.json"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, _ = dataset[i]
            output = model([image.to(device)])[0]
            boxes = output['boxes'].cpu().numpy().tolist()
            scores = output['scores'].cpu().numpy().tolist()
            predictions.append({
                "image_id": i,
                "boxes": boxes,
                "scores": scores
            })
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {output_path}")

# ---------------------- #
# 9. Single Image Inference
# ---------------------- #
def inference_on_image(model_path, image_path, device):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model([image_tensor])[0]
    draw = ImageDraw.Draw(image)
    for box, score in zip(output["boxes"], output["scores"]):
        if score > 0.5:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")
    image.save("inference_result.jpg")
    print("Saved inference result to inference_result.jpg")

# ---------------------- #
# 10. Main
# ---------------------- #
def main():
    img_dir = "/home/minjilee/Desktop/combined_0624/jpg"
    json_dir = "/home/minjilee/Desktop/combined_0624/location"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ERCDataset_JSON(img_dir, json_dir, transforms=F.to_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    train_losses, val_f1s = [], []
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        loss = train_one_epoch(model, train_loader, optimizer, device)
        f1 = validate(model, val_loader, device)
        train_losses.append(loss)
        val_f1s.append(f1)

    torch.save(model.state_dict(), "faster_rcnn_erc.pth")
    plot_metrics(train_losses, val_f1s)
    visualize_predictions(model, test_ds, device)
    confusion_matrix_iou(model, val_loader, device)
    save_predictions_to_json(model, test_ds, device)
    inference_on_image("faster_rcnn_erc.pth", os.path.join(img_dir, dataset.image_files[0]), device)

if __name__ == "__main__":
    main()




'''
Epoch 1/20
Train Loss: 0.5963
Precision: 0.0000, Recall: 0.0000, F1: 0.0000

Epoch 2/20
Train Loss: 0.6524
Precision: 0.4455, Recall: 0.6875, F1: 0.5406

Epoch 3/20
Train Loss: 0.6045
Precision: 0.5256, Recall: 0.8894, F1: 0.6607

Epoch 4/20
Train Loss: 0.6093
Precision: 0.3860, Recall: 0.7163, F1: 0.5017

Epoch 5/20
Train Loss: 0.5631
Precision: 0.3157, Recall: 1.0048, F1: 0.4805

Epoch 6/20
Train Loss: 0.5397
Precision: 0.4500, Recall: 0.9519, F1: 0.6111

Epoch 7/20
Train Loss: 0.5098
Precision: 0.4442, Recall: 0.9952, F1: 0.6142

Epoch 8/20
Train Loss: 0.4614
Precision: 0.5215, Recall: 0.9327, F1: 0.6690

Epoch 9/20
Train Loss: 0.4571
Precision: 0.4289, Recall: 0.9279, F1: 0.5866

Epoch 10/20
Train Loss: 0.4161
Precision: 0.6080, Recall: 0.8798, F1: 0.7191

Epoch 11/20
Train Loss: 0.3900
Precision: 0.5116, Recall: 0.9567, F1: 0.6667

Epoch 12/20
Train Loss: 0.3842
Precision: 0.5269, Recall: 0.9423, F1: 0.6759

Epoch 13/20
Train Loss: 0.3336
Precision: 0.5857, Recall: 0.9038, F1: 0.7108

Epoch 14/20
Train Loss: 0.3030
Precision: 0.6816, Recall: 0.8750, F1: 0.7663

Epoch 15/20
Train Loss: 0.3141
Precision: 0.6271, Recall: 0.9135, F1: 0.7436

Epoch 16/20
Train Loss: 0.2879
Precision: 0.7628, Recall: 0.7885, F1: 0.7754

Epoch 17/20
Train Loss: 0.2883
Precision: 0.7068, Recall: 0.8462, F1: 0.7702

Epoch 18/20
Train Loss: 0.2539
Precision: 0.6960, Recall: 0.8365, F1: 0.7598

Epoch 19/20
Train Loss: 0.2369
Precision: 0.7778, Recall: 0.8077, F1: 0.7925

Epoch 20/20
Train Loss: 0.2188
Precision: 0.6705, Recall: 0.8510, F1: 0.7500
Traceback (most recent call last):


'''
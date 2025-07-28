# faster_rcnn_main.py


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import json
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import datetime

# ---- Dataset Definition ----
class CustomRCNNDataset(Dataset):
    def __init__(self, img_dir, json_dir):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.image_ids = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id + ".jpg")
        json_path = os.path.join(self.json_dir, image_id + "_location.json")

        image = default_loader(img_path)

        with open(json_path, "r") as f:
            data = json.load(f)
        boxes = [obj["bbox"] for obj in data["objects"]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((len(boxes),), dtype=torch.int64)  # All labeled as ERC

        image = F.to_tensor(image)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

# ---- Save Model ----
def save_model(model, path="faster_rcnn_erc_epoch20.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

# ---- Training ----
def train(model, optimizer, dataloader, device, num_epochs=20):
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return loss_history

# ---- Loss Plot ----
def plot_loss(loss_history):
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("metrics_plot.png")
    plt.close()

# ---- COCO-style Evaluation (placeholder if COCOeval not fully applicable) ----
def evaluate_map(model, dataloader, device):
    model.eval()
    results = []
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ERC"}]}
    ann_id = 1

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, (output, target) in enumerate(zip(outputs, targets)):
                image_id = int(target["image_id"].item())
                coco_gt["images"].append({"id": image_id})
                for j, box in enumerate(target["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    width, height = x2 - x1, y2 - y1
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    ann_id += 1

                for box, score in zip(output["boxes"], output["scores"]):
                    x1, y1, x2, y2 = box.tolist()
                    width, height = x2 - x1, y2 - y1
                    results.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, width, height],
                        "score": score.item()
                    })

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = f"predictions_{timestamp}.json"
    with open("coco_gt.json", "w") as f:
        json.dump(coco_gt, f)
    with open(pred_path, "w") as f:
        json.dump(results, f)
    print(f"📦 Saved predictions to {pred_path}")

    coco_gt_obj = COCO("coco_gt.json")
    coco_dt = coco_gt_obj.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt_obj, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# ---- Main ----
def main():
    img_dir = "/home/minjilee/Desktop/combined_0624/jpg"
    json_dir = "/home/minjilee/Desktop/cnn_july22/r-cnn/bbox"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomRCNNDataset(img_dir, json_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # model = fasterrcnn_resnet50_fpn(pretrained=True)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # ERC and background
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    
    
    '''
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, 2)  # [background, ERC]
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Starting Training...")
    loss_history = train(model, optimizer, dataloader, device, num_epochs=20)

    save_model(model, "faster_rcnn_erc.pth")
    plot_loss(loss_history)

    print("\n📊 Evaluating mAP using COCOeval...")
    evaluate_map(model, dataloader, device)

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
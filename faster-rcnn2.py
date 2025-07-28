#faster-rcnn2.py 
# visualize...all the utilities


import os
import torch
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from faster_rcnn_main import (
    ERCDataset_JSON,
    get_model,
    collate_fn,
    plot_metrics,
    visualize_predictions,
    confusion_matrix_iou,
    save_predictions_to_json,
    inference_on_image
)

def run_postprocessing():
    img_dir = "/home/minjilee/Desktop/combined_0624/jpg"
    json_dir = "/home/minjilee/Desktop/combined_0624/location"
    model_path = "/home/minjilee/Desktop/cnn_july22/r-cnn/faster_rcnn_erc.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = ERCDataset_JSON(img_dir, json_dir, transforms=F.to_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load trained model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Run all postprocessing tasks
    visualize_predictions(model, test_ds, device)
    confusion_matrix_iou(model, val_loader, device)
    save_predictions_to_json(model, test_ds, device)
    inference_on_image(model_path, os.path.join(img_dir, dataset.image_files[0]), device)

if __name__ == "__main__":
    run_postprocessing()






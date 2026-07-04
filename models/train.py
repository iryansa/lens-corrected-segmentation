import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import mlflow

class CardDataset(Dataset):
    def __init__(self, split_dir, resize_shape=(512, 512), augment=False):
        self.img_dir = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.img_names = sorted(os.listdir(self.img_dir))
        self.resize_shape = resize_shape
        self.augment = augment

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image & mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize for memory efficiency & training speed
        img = cv2.resize(img, self.resize_shape, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.resize_shape, interpolation=cv2.INTER_NEAREST)

        # Normalize mask to binary values
        mask = (mask > 127).astype(np.float32)

        # Apply basic online augmentations if training
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            # Random slight brightness adjustments
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Convert to tensor formats
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor

def compute_metrics(preds_logits, targets):
    # Apply sigmoid threshold
    preds = (preds_logits > 0.0).float()
    
    # Compute intersection and union
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    precision = (intersection + 1e-6) / (preds.sum() + 1e-6)
    recall = (intersection + 1e-6) / (targets.sum() + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    
    return iou, precision, recall, f1

def main():
    # 1. Hyperparameters
    epochs = 15
    batch_size = 4
    lr = 1e-4
    resize_shape = (512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Datasets & Dataloaders
    train_dataset = CardDataset("dataset/train", resize_shape=resize_shape, augment=True)
    val_dataset = CardDataset("dataset/val", resize_shape=resize_shape, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model initialization
    try:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        print("Model initialized with pre-trained ResNet-34 ImageNet weights.")
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights ({e}). Initializing with random weights.")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
    model = model.to(device)

    # 4. Optimizer and Loss Functions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")

    # 5. MLflow Tracking Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PlayingCardSegmentation")

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_model.pth")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "image_size": f"{resize_shape[0]}x{resize_shape[1]}",
            "encoder": "resnet34",
            "device": str(device)
        })

        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_iou = 0.0
            
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                
                loss = bce_loss_fn(outputs, masks) + dice_loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                iou, _, _, _ = compute_metrics(outputs, masks)
                train_iou += iou.item() * imgs.size(0)
                
            train_loss /= len(train_loader.dataset)
            train_iou /= len(train_loader.dataset)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_f1 = 0.0
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    
                    loss = bce_loss_fn(outputs, masks) + dice_loss_fn(outputs, masks)
                    val_loss += loss.item() * imgs.size(0)
                    
                    iou, precision, recall, f1 = compute_metrics(outputs, masks)
                    val_iou += iou.item() * imgs.size(0)
                    val_precision += precision.item() * imgs.size(0)
                    val_recall += recall.item() * imgs.size(0)
                    val_f1 += f1.item() * imgs.size(0)
                    
            val_loss /= len(val_loader.dataset)
            val_iou /= len(val_loader.dataset)
            val_precision /= len(val_loader.dataset)
            val_recall /= len(val_loader.dataset)
            val_f1 /= len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }, step=epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best model to {best_model_path} with Val Loss: {val_loss:.4f}")
                mlflow.log_artifact(best_model_path, artifact_path="model_checkpoints")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

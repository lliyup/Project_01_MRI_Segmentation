import os
import sys
import json
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from model.unet3d import UNet3D
from losses.dice_loss import DiceLoss
from utils.metrics import dice_score, iou_score, logits_to_prediction
from datasets.heart_patch_dataset import HeartPatchDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nnunet_fold0_split(split_path="results/nnunet_fold0_split.json"):
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Cannot find split file: {split_path}. "
            f"Please run scripts/read_nnunet_split.py first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_cases = [case + ".nii.gz" for case in split["train_cases"]]
    val_cases = [case + ".nii.gz" for case in split["val_cases"]]

    return train_cases, val_cases


def train_one_epoch(model, dataloader, dice_loss_fn, ce_loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        dice_loss = dice_loss_fn(logits, masks)
        ce_loss = ce_loss_fn(logits, masks)
        loss = dice_loss + ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits_to_prediction(logits)
            dice = dice_score(preds, masks, class_id=1)
            iou = iou_score(preds, masks, class_id=1)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }


@torch.no_grad()
def validate(model, dataloader, dice_loss_fn, ce_loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        dice_loss = dice_loss_fn(logits, masks)
        ce_loss = ce_loss_fn(logits, masks)
        loss = dice_loss + ce_loss

        preds = logits_to_prediction(logits)
        dice = dice_score(preds, masks, class_id=1)
        iou = iou_score(preds, masks, class_id=1)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = "data/raw/Task02_Heart"

    train_cases, val_cases = load_nnunet_fold0_split(
        split_path="results/nnunet_fold0_split.json"
    )

    print("=" * 50)
    print("Train Self-implemented 3D U-Net with nnU-Net Fold 0 Split")
    print("=" * 50)
    print("Device:", device)
    print("Train cases:", len(train_cases), train_cases)
    print("Val cases:", len(val_cases), val_cases)

    train_dataset = HeartPatchDataset(
        dataset_dir=dataset_dir,
        case_names=train_cases,
        patch_size=(64, 96, 96),
        sample_per_case=4,
        positive_rate=0.8,
    )

    val_dataset = HeartPatchDataset(
        dataset_dir=dataset_dir,
        case_names=val_cases,
        patch_size=(64, 96, 96),
        sample_per_case=2,
        positive_rate=1.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet3D(
        in_channels=1,
        num_classes=2,
        base_channels=8,
    ).to(device)

    dice_loss_fn = DiceLoss(include_background=False)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    best_val_dice = -1.0
    history = []

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            dice_loss_fn=dice_loss_fn,
            ce_loss_fn=ce_loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            dice_loss_fn=dice_loss_fn,
            ce_loss_fn=ce_loss_fn,
            device=device,
        )

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        })

        history_df = pd.DataFrame(history)
        history_df.to_csv(
            "results/heart_training_log_nnunet_split.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            save_path = f"checkpoints/best_heart_unet3d_nnunet_split_epoch{epoch:02d}.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "base_channels": 8,
                    "patch_size": (64, 96, 96),
                    "best_val_dice": best_val_dice,
                    "train_cases": train_cases,
                    "val_cases": val_cases,
                    "split_source": "nnU-Net fold 0",
                },
                save_path,
            )

            print(f"New best model saved to: {os.path.abspath(save_path)}")

    print("=" * 50)
    print("Training finished.")
    print("Best Val Dice:", best_val_dice)
    print("Training log saved to: results/heart_training_log_nnunet_split.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
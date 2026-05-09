import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 根据你刚才训练输出整理的日志
    logs = [
        [1, 1.1676, 0.5651, 0.4464, 0.8771, 0.8324, 0.7143],
        [2, 0.9400, 0.6468, 0.5579, 0.7252, 0.7946, 0.6607],
        [3, 0.7392, 0.7274, 0.6288, 0.5118, 0.8907, 0.8042],
        [4, 0.6001, 0.7462, 0.6700, 0.4075, 0.8876, 0.7988],
        [5, 0.4802, 0.8001, 0.7153, 0.3046, 0.9146, 0.8432],
        [6, 0.5835, 0.6516, 0.5976, 0.3247, 0.8788, 0.7857],
        [7, 0.4280, 0.7528, 0.6834, 0.2908, 0.8779, 0.7840],
        [8, 0.3822, 0.7681, 0.6859, 0.2216, 0.9048, 0.8270],
        [9, 0.3280, 0.8073, 0.7348, 0.1875, 0.9231, 0.8576],
        [10, 0.4294, 0.7918, 0.7362, 0.1861, 0.9183, 0.8493],
        [11, 0.3253, 0.7935, 0.7271, 0.1972, 0.9066, 0.8297],
        [12, 0.3307, 0.8268, 0.7728, 0.1631, 0.9225, 0.8564],
        [13, 0.3449, 0.8039, 0.7561, 0.1401, 0.9332, 0.8753],
        [14, 0.2830, 0.8456, 0.7933, 0.1613, 0.9168, 0.8473],
        [15, 0.3155, 0.8153, 0.7643, 0.1548, 0.9221, 0.8567],
        [16, 0.2798, 0.8254, 0.7581, 0.1729, 0.9056, 0.8280],
        [17, 0.2447, 0.8279, 0.7637, 0.1279, 0.9324, 0.8737],
        [18, 0.2970, 0.8494, 0.8003, 0.1424, 0.9245, 0.8603],
        [19, 0.2700, 0.8286, 0.7755, 0.1117, 0.9412, 0.8892],
        [20, 0.2240, 0.8743, 0.8211, 0.1787, 0.8994, 0.8176],
    ]

    columns = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
    ]

    df = pd.DataFrame(logs, columns=columns)

    csv_path = os.path.join("results", "heart_training_log.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 50)
    print("Training Log Summary")
    print("=" * 50)
    print(df)
    print()
    print("Best Val Dice:")
    best_row = df.loc[df["val_dice"].idxmax()]
    print(best_row)
    print("CSV saved to:", csv_path)

    # Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Heart MRI 3D U-Net Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join("figures", "heart_loss_curve.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # Dice 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_dice"], marker="o", label="Train Dice")
    plt.plot(df["epoch"], df["val_dice"], marker="o", label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Heart MRI 3D U-Net Dice Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    dice_path = os.path.join("figures", "heart_dice_curve.png")
    plt.savefig(dice_path, dpi=200)
    plt.close()

    # IoU 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_iou"], marker="o", label="Train IoU")
    plt.plot(df["epoch"], df["val_iou"], marker="o", label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Heart MRI 3D U-Net IoU Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    iou_path = os.path.join("figures", "heart_iou_curve.png")
    plt.savefig(iou_path, dpi=200)
    plt.close()

    print("Loss curve saved to:", loss_path)
    print("Dice curve saved to:", dice_path)
    print("IoU curve saved to:", iou_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
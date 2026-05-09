import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from model.unet3d import UNet3D
from datasets.heart_patch_dataset import normalize_mri


def get_bbox(mask):
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return {
        "z": (int(z_min), int(z_max)),
        "y": (int(y_min), int(y_max)),
        "x": (int(x_min), int(x_max)),
        "center": (
            int((z_min + z_max) / 2),
            int((y_min + y_max) / 2),
            int((x_min + x_max) / 2),
        ),
    }


def get_start_positions(size, patch_size, stride):
    if size <= patch_size:
        return [0]

    positions = list(range(0, size - patch_size + 1, stride))

    if positions[-1] != size - patch_size:
        positions.append(size - patch_size)

    return positions


@torch.no_grad()
def sliding_window_prob(
    model,
    image,
    patch_size=(64, 96, 96),
    stride=(32, 48, 48),
    device="cuda",
):
    """
    image: [D, H, W]
    返回类别 1 概率图: [D, H, W]
    """

    model.eval()

    d, h, w = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    prob_sum = np.zeros((d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)

    z_starts = get_start_positions(d, pd, sd)
    y_starts = get_start_positions(h, ph, sh)
    x_starts = get_start_positions(w, pw, sw)

    total = len(z_starts) * len(y_starts) * len(x_starts)
    count = 0

    print("Sliding window patches:", total)

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = image[z:z + pd, y:y + ph, x:x + pw]

                patch_tensor = torch.from_numpy(patch).float()
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(device)

                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1)

                prob = probs[0, 1].cpu().numpy()

                prob_sum[z:z + pd, y:y + ph, x:x + pw] += prob
                count_map[z:z + pd, y:y + ph, x:x + pw] += 1

                count += 1
                if count % 20 == 0 or count == total:
                    print(f"Processed {count}/{total}")

    prob_map = prob_sum / np.maximum(count_map, 1e-8)
    return prob_map


def dice_iou_numpy(pred, target, smooth=1e-5):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    dice = (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )
    iou = (intersection + smooth) / (union + smooth)

    return float(dice), float(iou), int(intersection)


def threshold_sweep(prob_map, label):
    print()
    print("=" * 50)
    print("Threshold Sweep")
    print("=" * 50)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    for th in thresholds:
        pred = (prob_map >= th).astype(np.uint8)
        dice, iou, inter = dice_iou_numpy(pred, label)

        print(
            f"Threshold {th:.2f} | "
            f"Dice: {dice:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Pred voxels: {int(pred.sum())} | "
            f"Intersection: {inter}"
        )


def save_debug_slice(image, label, prob_map, threshold, save_path):
    """
    保存真实左心房最大切片上的：
    原图 / 真实 mask / 概率图 / 阈值预测
    """

    areas = label.sum(axis=(1, 2))
    slice_idx = int(np.argmax(areas))

    image_slice = image[slice_idx]
    label_slice = label[slice_idx]
    prob_slice = prob_map[slice_idx]
    pred_slice = (prob_slice >= threshold).astype(np.uint8)

    p1 = np.percentile(image_slice, 1)
    p99 = np.percentile(image_slice, 99)
    image_slice = np.clip(image_slice, p1, p99)
    image_slice = (image_slice - image_slice.min()) / (
        image_slice.max() - image_slice.min() + 1e-8
    )

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"MRI Slice {slice_idx}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(label_slice, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(prob_slice, cmap="hot", vmin=0, vmax=1)
    plt.title("Probability Map")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pred_slice, cmap="gray")
    plt.title(f"Prediction th={threshold}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Debug slice saved to:", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(
        "checkpoints",
        "best_heart_unet3d_epoch19.pth",
    )

    case_name = "la_003.nii.gz"

    image_path = os.path.join(
        "data", "raw", "Task02_Heart", "imagesTr", case_name
    )
    label_path = os.path.join(
        "data", "raw", "Task02_Heart", "labelsTr", case_name
    )

    print("=" * 50)
    print("Debug Whole-volume Inference")
    print("=" * 50)
    print("Device:", device)
    print("Case:", case_name)
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet3D(
        in_channels=1,
        num_classes=2,
        base_channels=16,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    label = (label > 0).astype(np.uint8)

    print("Original image shape:", image.shape)
    print("Original label shape:", label.shape)

    # HWD -> DHW
    image_dhw = np.transpose(image, (2, 0, 1))
    label_dhw = np.transpose(label, (2, 0, 1))

    image_dhw = normalize_mri(image_dhw)

    print("DHW image shape:", image_dhw.shape)
    print("DHW label shape:", label_dhw.shape)

    prob_map = sliding_window_prob(
        model=model,
        image=image_dhw,
        patch_size=(64, 96, 96),
        stride=(32, 48, 48),
        device=device,
    )

    pred_05 = (prob_map >= 0.5).astype(np.uint8)

    dice, iou, inter = dice_iou_numpy(pred_05, label_dhw)

    print()
    print("=" * 50)
    print("Basic Result at threshold=0.5")
    print("=" * 50)
    print("Dice:", dice)
    print("IoU:", iou)
    print("Intersection voxels:", inter)
    print("Pred positive voxels:", int(pred_05.sum()))
    print("GT positive voxels:", int(label_dhw.sum()))

    print()
    print("=" * 50)
    print("Bounding Boxes")
    print("=" * 50)
    print("GT bbox:", get_bbox(label_dhw))
    print("Pred bbox:", get_bbox(pred_05))

    gt_prob = prob_map[label_dhw > 0]
    bg_prob = prob_map[label_dhw == 0]

    print()
    print("=" * 50)
    print("Probability Statistics")
    print("=" * 50)
    print("Global prob min/max/mean:", float(prob_map.min()), float(prob_map.max()), float(prob_map.mean()))
    print("GT region prob mean/max:", float(gt_prob.mean()), float(gt_prob.max()))
    print("GT region prob p95:", float(np.percentile(gt_prob, 95)))
    print("BG region prob mean/max:", float(bg_prob.mean()), float(bg_prob.max()))
    print("BG region prob p99:", float(np.percentile(bg_prob, 99)))

    threshold_sweep(prob_map, label_dhw)

    os.makedirs("figures", exist_ok=True)
    save_debug_slice(
        image=image_dhw,
        label=label_dhw,
        prob_map=prob_map,
        threshold=0.5,
        save_path=os.path.join("figures", "debug_volume_probability_slice.png"),
    )

    print("=" * 50)
    print("Debug finished.")
    print("=" * 50)


if __name__ == "__main__":
    main()
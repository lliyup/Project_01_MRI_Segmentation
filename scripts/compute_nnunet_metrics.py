import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage


def dice_iou_numpy(pred, target, smooth=1e-5):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    dice = (2.0 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )
    iou = (intersection + smooth) / (union + smooth)

    return float(dice), float(iou)


def get_surface(mask):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = mask ^ eroded

    return surface


def hd95(pred, target, spacing):
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return np.inf

    pred_surface = get_surface(pred)
    target_surface = get_surface(target)

    dt_target = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing,
    )

    dt_pred = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    pred_to_target = dt_target[pred_surface]
    target_to_pred = dt_pred[target_surface]

    all_distances = np.concatenate([pred_to_target, target_to_pred])

    return float(np.percentile(all_distances, 95))


def load_mask_hwd(path):
    nii = nib.load(path)
    arr = nii.get_fdata()
    arr = (arr > 0).astype(np.uint8)
    spacing_hwd = nii.header.get_zooms()[:3]
    return arr, spacing_hwd


def main():
    label_dir = os.path.join(
        "data",
        "raw",
        "Task02_Heart",
        "labelsTr",
    )

    nnunet_val_dir = r"D:\nnunet_work\nnUNet_results\Dataset002_Heart\nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres\fold_0\validation"

    if not os.path.exists(nnunet_val_dir):
        raise FileNotFoundError(
            f"Cannot find nnU-Net validation folder: {nnunet_val_dir}"
        )

    case_files = sorted([
        f for f in os.listdir(nnunet_val_dir)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])

    if len(case_files) == 0:
        raise RuntimeError("No .nii.gz prediction files found in nnU-Net validation folder.")

    results = []

    print("=" * 50)
    print("Compute nnU-Net Validation Metrics")
    print("=" * 50)
    print("Prediction dir:", nnunet_val_dir)
    print("Cases:", case_files)

    for case_name in case_files:
        pred_path = os.path.join(nnunet_val_dir, case_name)
        label_path = os.path.join(label_dir, case_name)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Cannot find label: {label_path}")

        pred, _ = load_mask_hwd(pred_path)
        label, spacing_hwd = load_mask_hwd(label_path)

        dice, iou = dice_iou_numpy(pred, label)
        hd95_mm = hd95(pred, label, spacing_hwd)

        print(
            f"{case_name} | Dice: {dice:.4f} | IoU: {iou:.4f} | HD95: {hd95_mm:.4f} mm"
        )

        results.append({
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "hd95_mm": hd95_mm,
            "pred_voxels": int(pred.sum()),
            "gt_voxels": int(label.sum()),
        })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "nnunet_50epoch_validation_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(df)
    print()
    print("Mean Dice:", df["dice"].mean())
    print("Mean IoU:", df["iou"].mean())
    print("Mean HD95 mm:", df["hd95_mm"].mean())
    print("CSV saved to:", csv_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
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
    """
    提取 3D mask 表面体素。

    mask: [D, H, W]
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = mask ^ eroded

    return surface


def hd95(pred, target, spacing):
    """
    计算 95% Hausdorff Distance。

    pred:   [D, H, W]
    target: [D, H, W]
    spacing: (spacing_d, spacing_h, spacing_w)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return np.inf

    pred_surface = get_surface(pred)
    target_surface = get_surface(target)

    # distance_transform_edt 计算的是到最近 0 的距离
    # 所以这里对 surface 取反，得到每个点到 target surface 的距离
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


def load_label_and_spacing(label_path):
    nii = nib.load(label_path)
    arr = nii.get_fdata()
    arr = (arr > 0).astype(np.uint8)

    # 原始 shape 是 [H, W, D]
    # 转成 [D, H, W]
    arr_dhw = np.transpose(arr, (2, 0, 1))

    # 原始 zooms 是 (spacing_h, spacing_w, spacing_d)
    zooms = nii.header.get_zooms()[:3]
    spacing_dhw = (float(zooms[2]), float(zooms[0]), float(zooms[1]))

    return arr_dhw, spacing_dhw


def load_prediction(pred_path):
    nii = nib.load(pred_path)
    arr = nii.get_fdata()
    arr = (arr > 0).astype(np.uint8)

    # 原始保存为 [H, W, D]
    # 转成 [D, H, W]
    arr_dhw = np.transpose(arr, (2, 0, 1))

    return arr_dhw


def main():
    dataset_dir = os.path.join("data", "raw", "Task02_Heart")
    label_dir = os.path.join(dataset_dir, "labelsTr")

    pred_dir = os.path.join("results", "predictions", "volume_eval")

    val_cases = [
        "la_014.nii.gz",
        "la_016.nii.gz",
        "la_003.nii.gz",
        "la_007.nii.gz",
    ]

    results = []

    print("=" * 50)
    print("Compute Whole-volume Dice / IoU / HD95")
    print("=" * 50)

    for case_name in val_cases:
        print()
        print("Case:", case_name)

        label_path = os.path.join(label_dir, case_name)

        raw_pred_path = os.path.join(
            
            pred_dir,
            case_name.replace(".nii.gz", "_raw_pred.nii.gz"),
        )

        lcc_pred_path = os.path.join(
            pred_dir,
            case_name.replace(".nii.gz", "_lcc_pred.nii.gz"),
        )

        if not os.path.exists(raw_pred_path):
            raise FileNotFoundError(f"Raw prediction not found: {raw_pred_path}")

        if not os.path.exists(lcc_pred_path):
            raise FileNotFoundError(f"LCC prediction not found: {lcc_pred_path}")

        label, spacing = load_label_and_spacing(label_path)
        raw_pred = load_prediction(raw_pred_path)
        lcc_pred = load_prediction(lcc_pred_path)

        raw_dice, raw_iou = dice_iou_numpy(raw_pred, label)
        lcc_dice, lcc_iou = dice_iou_numpy(lcc_pred, label)

        raw_hd95 = hd95(raw_pred, label, spacing)
        lcc_hd95 = hd95(lcc_pred, label, spacing)

        print(
            f"Raw  | Dice: {raw_dice:.4f} | IoU: {raw_iou:.4f} | HD95: {raw_hd95:.4f} mm"
        )
        print(
            f"LCC  | Dice: {lcc_dice:.4f} | IoU: {lcc_iou:.4f} | HD95: {lcc_hd95:.4f} mm"
        )

        results.append({
            "case": case_name,
            "raw_dice": raw_dice,
            "raw_iou": raw_iou,
            "raw_hd95_mm": raw_hd95,
            "lcc_dice": lcc_dice,
            "lcc_iou": lcc_iou,
            "lcc_hd95_mm": lcc_hd95,
            "spacing_d": spacing[0],
            "spacing_h": spacing[1],
            "spacing_w": spacing[2],
        })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "heart_volume_metrics_with_hd95.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(df[[
        "case",
        "raw_dice",
        "raw_iou",
        "raw_hd95_mm",
        "lcc_dice",
        "lcc_iou",
        "lcc_hd95_mm",
    ]])

    print()
    print("Mean Raw Dice:", df["raw_dice"].mean())
    print("Mean Raw IoU:", df["raw_iou"].mean())
    print("Mean Raw HD95 mm:", df["raw_hd95_mm"].mean())

    print("Mean LCC Dice:", df["lcc_dice"].mean())
    print("Mean LCC IoU:", df["lcc_iou"].mean())
    print("Mean LCC HD95 mm:", df["lcc_hd95_mm"].mean())

    print("CSV saved to:", csv_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
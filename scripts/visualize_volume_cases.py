import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt


def normalize_slice(image_slice):
    p1 = np.percentile(image_slice, 1)
    p99 = np.percentile(image_slice, 99)

    image_slice = np.clip(image_slice, p1, p99)
    image_slice = (image_slice - image_slice.min()) / (
        image_slice.max() - image_slice.min() + 1e-8
    )

    return image_slice


def create_error_map(gt, pred):
    """
    0 = background
    1 = TP
    2 = FN
    3 = FP
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = gt & pred
    fn = gt & (~pred)
    fp = (~gt) & pred

    error_map = np.zeros_like(gt, dtype=np.uint8)
    error_map[tp] = 1
    error_map[fn] = 2
    error_map[fp] = 3

    return error_map


def load_hwd(path):
    nii = nib.load(path)
    arr = nii.get_fdata()
    return arr


def choose_largest_gt_slice(label_hwd):
    """
    原始数据是 [H, W, D]，沿 D 方向选 GT 面积最大的切片。
    """
    label = (label_hwd > 0).astype(np.uint8)
    areas = label.sum(axis=(0, 1))
    slice_idx = int(np.argmax(areas))
    return slice_idx


def visualize_case(case_name, metrics_row, save_path):
    image_path = os.path.join(
        "data", "raw", "Task02_Heart", "imagesTr", case_name
    )
    label_path = os.path.join(
        "data", "raw", "Task02_Heart", "labelsTr", case_name
    )

    pred_dir = os.path.join("results", "predictions", "volume_eval")

    raw_pred_path = os.path.join(
        pred_dir,
        case_name.replace(".nii.gz", "_raw_pred.nii.gz"),
    )

    lcc_pred_path = os.path.join(
        pred_dir,
        case_name.replace(".nii.gz", "_lcc_pred.nii.gz"),
    )

    image = load_hwd(image_path)
    label = (load_hwd(label_path) > 0).astype(np.uint8)
    raw_pred = (load_hwd(raw_pred_path) > 0).astype(np.uint8)
    lcc_pred = (load_hwd(lcc_pred_path) > 0).astype(np.uint8)

    slice_idx = choose_largest_gt_slice(label)

    image_slice = normalize_slice(image[:, :, slice_idx])
    label_slice = label[:, :, slice_idx]
    raw_slice = raw_pred[:, :, slice_idx]
    lcc_slice = lcc_pred[:, :, slice_idx]

    raw_error = create_error_map(label_slice, raw_slice)
    lcc_error = create_error_map(label_slice, lcc_slice)

    plt.figure(figsize=(20, 8))

    plt.subplot(2, 5, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"MRI Slice {slice_idx}")
    plt.axis("off")

    plt.subplot(2, 5, 2)
    plt.imshow(label_slice, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 5, 3)
    plt.imshow(raw_slice, cmap="gray")
    plt.title("Raw Prediction")
    plt.axis("off")

    plt.subplot(2, 5, 4)
    plt.imshow(lcc_slice, cmap="gray")
    plt.title("LCC Prediction")
    plt.axis("off")

    plt.subplot(2, 5, 5)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(lcc_slice, alpha=0.35)
    plt.title("LCC Overlay")
    plt.axis("off")

    plt.subplot(2, 5, 6)
    plt.imshow(image_slice, cmap="gray")
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(2, 5, 7)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(label_slice, alpha=0.35)
    plt.title("GT Overlay")
    plt.axis("off")

    plt.subplot(2, 5, 8)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(raw_error, alpha=0.65, vmin=0, vmax=3)
    plt.title("Raw Error Map")
    plt.axis("off")

    plt.subplot(2, 5, 9)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(lcc_error, alpha=0.65, vmin=0, vmax=3)
    plt.title("LCC Error Map")
    plt.axis("off")

    plt.subplot(2, 5, 10)
    text = (
        f"Case: {case_name}\n"
        f"Raw Dice: {metrics_row['raw_dice']:.4f}\n"
        f"Raw IoU: {metrics_row['raw_iou']:.4f}\n"
        f"Raw HD95: {metrics_row['raw_hd95_mm']:.2f} mm\n\n"
        f"LCC Dice: {metrics_row['lcc_dice']:.4f}\n"
        f"LCC IoU: {metrics_row['lcc_iou']:.4f}\n"
        f"LCC HD95: {metrics_row['lcc_hd95_mm']:.2f} mm"
    )

    plt.text(0.05, 0.5, text, fontsize=12, va="center")
    plt.axis("off")

    plt.suptitle(f"Whole-volume Prediction Visualization: {case_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)


def main():
    metrics_path = os.path.join("results", "heart_volume_metrics_with_hd95.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Cannot find results/heart_volume_metrics_with_hd95.csv. "
            "Please run compute_heart_volume_hd95.py first."
        )

    df = pd.read_csv(metrics_path)

    os.makedirs("figures", exist_ok=True)

    # 最好病例：LCC Dice 最高
    best_row = df.loc[df["lcc_dice"].idxmax()]
    best_case = best_row["case"]

    # 最差病例：LCC Dice 最低
    worst_row = df.loc[df["lcc_dice"].idxmin()]
    worst_case = worst_row["case"]

    # 后处理提升最大病例
    df["dice_improvement"] = df["lcc_dice"] - df["raw_dice"]
    improved_row = df.loc[df["dice_improvement"].idxmax()]
    improved_case = improved_row["case"]

    print("=" * 50)
    print("Volume Case Visualization")
    print("=" * 50)
    print("Best LCC case:", best_case)
    print("Worst LCC case:", worst_case)
    print("Most improved case:", improved_case)

    visualize_case(
        best_case,
        best_row,
        os.path.join("figures", f"{best_case}_best_volume_visualization.png"),
    )

    visualize_case(
        worst_case,
        worst_row,
        os.path.join("figures", f"{worst_case}_worst_volume_visualization.png"),
    )

    visualize_case(
        improved_case,
        improved_row,
        os.path.join("figures", f"{improved_case}_most_improved_visualization.png"),
    )

    print("=" * 50)
    print("Visualization finished.")
    print("=" * 50)


if __name__ == "__main__":
    main()
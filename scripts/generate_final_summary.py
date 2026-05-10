import os
import pandas as pd


def load_training_log():
    path = os.path.join("results", "heart_training_log.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Cannot find results/heart_training_log.csv. "
            "Please make sure the patch-level training log exists."
        )

    df = pd.read_csv(path)
    return df


def load_volume_metrics():
    path = os.path.join("results", "heart_volume_metrics_with_hd95.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Cannot find results/heart_volume_metrics_with_hd95.csv. "
            "Please run scripts/compute_heart_volume_hd95.py first."
        )

    df = pd.read_csv(path)
    return df


def main():
    os.makedirs("results", exist_ok=True)

    train_df = load_training_log()
    volume_df = load_volume_metrics()

    # patch-level best result
    best_patch_row = train_df.loc[train_df["val_dice"].idxmax()]

    best_patch_epoch = int(best_patch_row["epoch"])
    best_patch_val_loss = float(best_patch_row["val_loss"])
    best_patch_val_dice = float(best_patch_row["val_dice"])
    best_patch_val_iou = float(best_patch_row["val_iou"])

    # whole-volume mean metrics
    mean_raw_dice = float(volume_df["raw_dice"].mean())
    mean_raw_iou = float(volume_df["raw_iou"].mean())
    mean_raw_hd95 = float(volume_df["raw_hd95_mm"].mean())

    mean_lcc_dice = float(volume_df["lcc_dice"].mean())
    mean_lcc_iou = float(volume_df["lcc_iou"].mean())
    mean_lcc_hd95 = float(volume_df["lcc_hd95_mm"].mean())

    # best / worst / most improved
    best_case_row = volume_df.loc[volume_df["lcc_dice"].idxmax()]
    worst_case_row = volume_df.loc[volume_df["lcc_dice"].idxmin()]

    volume_df = volume_df.copy()
    volume_df["dice_improvement"] = volume_df["lcc_dice"] - volume_df["raw_dice"]
    volume_df["hd95_reduction"] = volume_df["raw_hd95_mm"] - volume_df["lcc_hd95_mm"]

    most_improved_row = volume_df.loc[volume_df["dice_improvement"].idxmax()]
    most_hd95_reduced_row = volume_df.loc[volume_df["hd95_reduction"].idxmax()]

    # summary table
    summary_items = [
        {
            "section": "Patch-level validation",
            "metric": "Best epoch",
            "value": best_patch_epoch,
            "note": "Best epoch selected by validation Dice."
        },
        {
            "section": "Patch-level validation",
            "metric": "Best Val Loss",
            "value": best_patch_val_loss,
            "note": "Patch-level validation loss."
        },
        {
            "section": "Patch-level validation",
            "metric": "Best Val Dice",
            "value": best_patch_val_dice,
            "note": "Patch-level validation Dice."
        },
        {
            "section": "Patch-level validation",
            "metric": "Best Val IoU",
            "value": best_patch_val_iou,
            "note": "Patch-level validation IoU."
        },
        {
            "section": "Whole-volume raw prediction",
            "metric": "Mean Raw Dice",
            "value": mean_raw_dice,
            "note": "Mean Dice before post-processing."
        },
        {
            "section": "Whole-volume raw prediction",
            "metric": "Mean Raw IoU",
            "value": mean_raw_iou,
            "note": "Mean IoU before post-processing."
        },
        {
            "section": "Whole-volume raw prediction",
            "metric": "Mean Raw HD95 mm",
            "value": mean_raw_hd95,
            "note": "Mean HD95 before post-processing."
        },
        {
            "section": "Whole-volume LCC post-processing",
            "metric": "Mean LCC Dice",
            "value": mean_lcc_dice,
            "note": "Mean Dice after largest connected component post-processing."
        },
        {
            "section": "Whole-volume LCC post-processing",
            "metric": "Mean LCC IoU",
            "value": mean_lcc_iou,
            "note": "Mean IoU after largest connected component post-processing."
        },
        {
            "section": "Whole-volume LCC post-processing",
            "metric": "Mean LCC HD95 mm",
            "value": mean_lcc_hd95,
            "note": "Mean HD95 after largest connected component post-processing."
        },
        {
            "section": "Case analysis",
            "metric": "Best LCC case",
            "value": best_case_row["case"],
            "note": f"LCC Dice={best_case_row['lcc_dice']:.4f}, LCC HD95={best_case_row['lcc_hd95_mm']:.2f} mm."
        },
        {
            "section": "Case analysis",
            "metric": "Worst LCC case",
            "value": worst_case_row["case"],
            "note": f"LCC Dice={worst_case_row['lcc_dice']:.4f}, LCC HD95={worst_case_row['lcc_hd95_mm']:.2f} mm."
        },
        {
            "section": "Case analysis",
            "metric": "Most improved case by Dice",
            "value": most_improved_row["case"],
            "note": f"Raw Dice={most_improved_row['raw_dice']:.4f} -> LCC Dice={most_improved_row['lcc_dice']:.4f}."
        },
        {
            "section": "Case analysis",
            "metric": "Most improved case by HD95",
            "value": most_hd95_reduced_row["case"],
            "note": f"Raw HD95={most_hd95_reduced_row['raw_hd95_mm']:.2f} mm -> LCC HD95={most_hd95_reduced_row['lcc_hd95_mm']:.2f} mm."
        },
    ]

    summary_df = pd.DataFrame(summary_items)

    csv_path = os.path.join("results", "final_experiment_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    txt_path = os.path.join("results", "final_experiment_summary.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Final Experiment Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Project: Self-implemented 3D U-Net for Heart MRI Left Atrium Segmentation\n")
        f.write("Dataset: MSD Task02 Heart\n")
        f.write("Task: Whole-volume left atrium segmentation from 3D cardiac MRI\n\n")

        f.write("-" * 80 + "\n")
        f.write("1. Patch-level validation result\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best epoch: {best_patch_epoch}\n")
        f.write(f"Best Val Loss: {best_patch_val_loss:.4f}\n")
        f.write(f"Best Val Dice: {best_patch_val_dice:.4f}\n")
        f.write(f"Best Val IoU: {best_patch_val_iou:.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("2. Whole-volume evaluation result\n")
        f.write("-" * 80 + "\n")
        f.write("Raw prediction:\n")
        f.write(f"  Mean Raw Dice: {mean_raw_dice:.4f}\n")
        f.write(f"  Mean Raw IoU: {mean_raw_iou:.4f}\n")
        f.write(f"  Mean Raw HD95: {mean_raw_hd95:.2f} mm\n\n")

        f.write("After largest connected component post-processing:\n")
        f.write(f"  Mean LCC Dice: {mean_lcc_dice:.4f}\n")
        f.write(f"  Mean LCC IoU: {mean_lcc_iou:.4f}\n")
        f.write(f"  Mean LCC HD95: {mean_lcc_hd95:.2f} mm\n\n")

        f.write("-" * 80 + "\n")
        f.write("3. Case-level analysis\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Best LCC case: {best_case_row['case']} "
            f"(Dice={best_case_row['lcc_dice']:.4f}, "
            f"IoU={best_case_row['lcc_iou']:.4f}, "
            f"HD95={best_case_row['lcc_hd95_mm']:.2f} mm)\n"
        )
        f.write(
            f"Worst LCC case: {worst_case_row['case']} "
            f"(Dice={worst_case_row['lcc_dice']:.4f}, "
            f"IoU={worst_case_row['lcc_iou']:.4f}, "
            f"HD95={worst_case_row['lcc_hd95_mm']:.2f} mm)\n"
        )
        f.write(
            f"Most improved case by Dice: {most_improved_row['case']} "
            f"(Raw Dice={most_improved_row['raw_dice']:.4f} -> "
            f"LCC Dice={most_improved_row['lcc_dice']:.4f})\n"
        )
        f.write(
            f"Most improved case by HD95: {most_hd95_reduced_row['case']} "
            f"(Raw HD95={most_hd95_reduced_row['raw_hd95_mm']:.2f} mm -> "
            f"LCC HD95={most_hd95_reduced_row['lcc_hd95_mm']:.2f} mm)\n\n"
        )

        f.write("-" * 80 + "\n")
        f.write("4. Main conclusion\n")
        f.write("-" * 80 + "\n")
        f.write(
            "The self-implemented 3D U-Net achieved strong patch-level validation performance "
            f"with Best Val Dice={best_patch_val_dice:.4f} and Best Val IoU={best_patch_val_iou:.4f}. "
            "However, patch-level performance was higher than whole-volume performance, indicating "
            "that patch-level validation is optimistic and cannot fully represent full-case segmentation ability.\n\n"
        )
        f.write(
            "In whole-volume sliding window inference, raw predictions showed distant false positives, "
            f"leading to Mean Raw Dice={mean_raw_dice:.4f} and Mean Raw HD95={mean_raw_hd95:.2f} mm. "
            "After applying largest connected component post-processing, false positives were reduced, "
            f"improving Mean Dice to {mean_lcc_dice:.4f} and reducing Mean HD95 to {mean_lcc_hd95:.2f} mm.\n\n"
        )
        f.write(
            "This suggests that the model can segment the main left atrium structure, but still suffers "
            "from false positives and boundary errors. Future improvements should include stronger "
            "background patch sampling, whole-volume training strategy, nnU-Net/MONAI baseline comparison, "
            "more data augmentation, and boundary-aware loss functions.\n"
        )

    print("=" * 50)
    print("Final Summary Generated")
    print("=" * 50)
    print("CSV saved to:", csv_path)
    print("TXT saved to:", txt_path)
    print()
    print(summary_df)
    print("=" * 50)


if __name__ == "__main__":
    main()
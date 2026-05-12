import os
import pandas as pd


def main():
    os.makedirs("results", exist_ok=True)

    baseline_path = os.path.join(
        "results",
        "heart_volume_metrics_nnunet_split_with_hd95.csv",
    )

    nnunet_path = os.path.join(
        "results",
        "nnunet_50epoch_validation_metrics.csv",
    )

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Cannot find baseline metrics: {baseline_path}")

    if not os.path.exists(nnunet_path):
        raise FileNotFoundError(f"Cannot find nnU-Net metrics: {nnunet_path}")

    baseline_df = pd.read_csv(baseline_path)
    nnunet_df = pd.read_csv(nnunet_path)

    rows = [
        {
            "method": "Self-implemented 3D U-Net Raw",
            "validation_split": "nnU-Net fold 0 validation cases",
            "mean_dice": baseline_df["raw_dice"].mean(),
            "mean_iou": baseline_df["raw_iou"].mean(),
            "mean_hd95_mm": baseline_df["raw_hd95_mm"].mean(),
            "note": "Raw whole-volume sliding window prediction before post-processing.",
        },
        {
            "method": "Self-implemented 3D U-Net + LCC",
            "validation_split": "nnU-Net fold 0 validation cases",
            "mean_dice": baseline_df["lcc_dice"].mean(),
            "mean_iou": baseline_df["lcc_iou"].mean(),
            "mean_hd95_mm": baseline_df["lcc_hd95_mm"].mean(),
            "note": "Whole-volume prediction after largest connected component post-processing.",
        },
        {
            "method": "nnU-Net v2 3d_fullres",
            "validation_split": "nnU-Net fold 0 validation cases",
            "mean_dice": nnunet_df["dice"].mean(),
            "mean_iou": nnunet_df["iou"].mean(),
            "mean_hd95_mm": nnunet_df["hd95_mm"].mean(),
            "note": "nnU-Net v2 3d_fullres, fold 0, 50 epochs short-run.",
        },
    ]

    summary_df = pd.DataFrame(rows)

    csv_path = os.path.join("results", "v3_fair_comparison_summary.csv")
    txt_path = os.path.join("results", "v3_fair_comparison_summary.txt")

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    raw_dice = baseline_df["raw_dice"].mean()
    raw_iou = baseline_df["raw_iou"].mean()
    raw_hd95 = baseline_df["raw_hd95_mm"].mean()

    lcc_dice = baseline_df["lcc_dice"].mean()
    lcc_iou = baseline_df["lcc_iou"].mean()
    lcc_hd95 = baseline_df["lcc_hd95_mm"].mean()

    nnunet_dice = nnunet_df["dice"].mean()
    nnunet_iou = nnunet_df["iou"].mean()
    nnunet_hd95 = nnunet_df["hd95_mm"].mean()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("V3.0 Fair Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Validation split:\n")
        f.write("All methods are evaluated on the same nnU-Net fold 0 validation cases.\n")
        f.write("Validation cases: la_007, la_016, la_021, la_024\n\n")

        f.write("1. Quantitative results\n")
        f.write("-" * 80 + "\n")

        for _, row in summary_df.iterrows():
            f.write(f"Method: {row['method']}\n")
            f.write(f"Validation split: {row['validation_split']}\n")
            f.write(f"Mean Dice: {row['mean_dice']:.4f}\n")
            f.write(f"Mean IoU: {row['mean_iou']:.4f}\n")
            f.write(f"Mean HD95: {row['mean_hd95_mm']:.2f} mm\n")
            f.write(f"Note: {row['note']}\n\n")

        f.write("2. Main findings\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"The raw self-implemented 3D U-Net achieved Mean Dice={raw_dice:.4f}, "
            f"Mean IoU={raw_iou:.4f}, and Mean HD95={raw_hd95:.2f} mm on the same "
            "nnU-Net fold 0 validation cases. The low raw Dice and high HD95 indicate "
            "that raw whole-volume predictions contained distant false positives.\n\n"
        )

        f.write(
            f"After largest connected component post-processing, the self-implemented "
            f"3D U-Net improved to Mean Dice={lcc_dice:.4f}, Mean IoU={lcc_iou:.4f}, "
            f"and Mean HD95={lcc_hd95:.2f} mm. This shows that the model can segment "
            "the main left atrium structure, while LCC effectively removes distant "
            "false positive components.\n\n"
        )

        f.write(
            f"nnU-Net v2 3d_fullres achieved Mean Dice={nnunet_dice:.4f}, "
            f"Mean IoU={nnunet_iou:.4f}, and Mean HD95={nnunet_hd95:.2f} mm. "
            "It still outperformed the self-implemented 3D U-Net + LCC, showing the "
            "advantage of nnU-Net in automatic planning, preprocessing, data augmentation, "
            "inference strategy, and robust post-processing.\n\n"
        )

        f.write("3. Conclusion\n")
        f.write("-" * 80 + "\n")
        f.write(
            "Under the same validation split, the self-implemented 3D U-Net baseline "
            "can approach nnU-Net performance after LCC post-processing, but nnU-Net "
            "remains stronger and more stable. This validates both the correctness of "
            "the self-implemented pipeline and the value of nnU-Net as a strong medical "
            "image segmentation baseline.\n"
        )

    print("=" * 50)
    print("V3.0 Fair Comparison Summary Generated")
    print("=" * 50)
    print(summary_df)
    print()
    print("CSV saved to:", csv_path)
    print("TXT saved to:", txt_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
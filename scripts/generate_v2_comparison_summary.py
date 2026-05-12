import os
import pandas as pd


def main():
    os.makedirs("results", exist_ok=True)

    baseline_path = os.path.join("results", "heart_volume_metrics_with_hd95.csv")
    nnunet_path = os.path.join("results", "nnunet_50epoch_validation_metrics.csv")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Cannot find baseline metrics: {baseline_path}")

    if not os.path.exists(nnunet_path):
        raise FileNotFoundError(f"Cannot find nnU-Net metrics: {nnunet_path}")

    baseline_df = pd.read_csv(baseline_path)
    nnunet_df = pd.read_csv(nnunet_path)

    rows = [
        {
            "method": "Self-implemented 3D U-Net Raw",
            "setting": "whole-volume sliding window",
            "mean_dice": baseline_df["raw_dice"].mean(),
            "mean_iou": baseline_df["raw_iou"].mean(),
            "mean_hd95_mm": baseline_df["raw_hd95_mm"].mean(),
            "note": "Raw prediction before post-processing."
        },
        {
            "method": "Self-implemented 3D U-Net + LCC",
            "setting": "whole-volume sliding window + largest connected component",
            "mean_dice": baseline_df["lcc_dice"].mean(),
            "mean_iou": baseline_df["lcc_iou"].mean(),
            "mean_hd95_mm": baseline_df["lcc_hd95_mm"].mean(),
            "note": "Post-processed prediction using largest connected component."
        },
        {
            "method": "nnU-Net v2 3d_fullres",
            "setting": "fold 0, 50 epochs short-run",
            "mean_dice": nnunet_df["dice"].mean(),
            "mean_iou": nnunet_df["iou"].mean(),
            "mean_hd95_mm": nnunet_df["hd95_mm"].mean(),
            "note": "nnU-Net strong baseline. Validation split differs from the self-implemented baseline."
        },
    ]

    summary_df = pd.DataFrame(rows)

    csv_path = os.path.join("results", "v2_comparison_summary.csv")
    txt_path = os.path.join("results", "v2_comparison_summary.txt")

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("V2.0 Comparison Summary: Self-implemented 3D U-Net vs nnU-Net\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Quantitative comparison\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"Method: {row['method']}\n")
            f.write(f"Setting: {row['setting']}\n")
            f.write(f"Mean Dice: {row['mean_dice']:.4f}\n")
            f.write(f"Mean IoU: {row['mean_iou']:.4f}\n")
            f.write(f"Mean HD95: {row['mean_hd95_mm']:.2f} mm\n")
            f.write(f"Note: {row['note']}\n\n")

        f.write("2. Main observation\n")
        f.write("-" * 80 + "\n")
        f.write(
            "The self-implemented 3D U-Net successfully established a complete "
            "medical image segmentation baseline, including patch-level training, "
            "whole-volume sliding window inference, Dice/IoU/HD95 evaluation, and "
            "largest connected component post-processing. LCC post-processing improved "
            "the whole-volume Dice from "
            f"{baseline_df['raw_dice'].mean():.4f} to {baseline_df['lcc_dice'].mean():.4f}, "
            "and reduced HD95 from "
            f"{baseline_df['raw_hd95_mm'].mean():.2f} mm to {baseline_df['lcc_hd95_mm'].mean():.2f} mm.\n\n"
        )

        f.write(
            "The nnU-Net v2 3d_fullres 50-epoch short-run achieved substantially better "
            "validation performance, with Mean Dice "
            f"{nnunet_df['dice'].mean():.4f}, Mean IoU {nnunet_df['iou'].mean():.4f}, "
            f"and Mean HD95 {nnunet_df['hd95_mm'].mean():.2f} mm. "
            "This demonstrates the advantage of nnU-Net as a strong medical image segmentation baseline, "
            "benefiting from automatic planning, preprocessing, augmentation, inference, and post-processing.\n\n"
        )

        f.write("3. Important limitation\n")
        f.write("-" * 80 + "\n")
        f.write(
            "The validation cases used by the self-implemented 3D U-Net and nnU-Net are not fully identical. "
            "Therefore, the current table should be interpreted as a method-level reference comparison rather "
            "than a strictly controlled fair comparison. A stricter comparison should use the same training and "
            "validation split for both methods.\n"
        )

    print("=" * 50)
    print("V2.0 Comparison Summary Generated")
    print("=" * 50)
    print(summary_df)
    print()
    print("CSV saved to:", csv_path)
    print("TXT saved to:", txt_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
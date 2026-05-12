import os
import sys
import glob
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import ndimage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from model.unet3d import UNet3D
from datasets.heart_patch_dataset import normalize_mri


def find_best_checkpoint():
    pattern = os.path.join(
        "checkpoints",
        "best_heart_unet3d_nnunet_split_epoch11.pth",
    )
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            "Cannot find checkpoints/best_heart_unet3d_nnunet_split_epoch*.pth. "
            "Please run train_heart_3dunet_nnunet_split.py first."
        )

    return files[-1]


def load_nnunet_fold0_val_cases(split_path="results/nnunet_fold0_split.json"):
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Cannot find split file: {split_path}. "
            f"Please run scripts/read_nnunet_split.py first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_cases = [case + ".nii.gz" for case in split["val_cases"]]
    return val_cases


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

    print(f"Sliding window patches: {total}")

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
                count_map[z:z + pd, y:y + ph, x:x + pw] += 1.0

                count += 1
                if count % 50 == 0 or count == total:
                    print(f"Processed {count}/{total}")

    prob_map = prob_sum / np.maximum(count_map, 1e-8)
    return prob_map


def keep_largest_connected_component(mask):
    mask = mask.astype(np.uint8)

    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask

    component_sizes = ndimage.sum(
        mask,
        labeled,
        index=range(1, num_features + 1),
    )

    largest_component_id = int(np.argmax(component_sizes)) + 1
    largest_mask = (labeled == largest_component_id).astype(np.uint8)

    return largest_mask


def dice_iou_numpy(pred, target, smooth=1e-5):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    dice = (2.0 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )
    iou = (intersection + smooth) / (union + smooth)

    return float(dice), float(iou), int(intersection)


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


def get_bbox(mask):
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return {
        "z_min": int(z_min),
        "z_max": int(z_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_min": int(x_min),
        "x_max": int(x_max),
    }


def evaluate_case(
    model,
    case_name,
    image_dir,
    label_dir,
    output_dir,
    device,
    threshold=0.5,
):
    image_path = os.path.join(image_dir, case_name)
    label_path = os.path.join(label_dir, case_name)

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    label = (label > 0).astype(np.uint8)

    zooms = label_nii.header.get_zooms()[:3]
    spacing_dhw = (float(zooms[2]), float(zooms[0]), float(zooms[1]))

    # HWD -> DHW
    image_dhw = np.transpose(image, (2, 0, 1))
    label_dhw = np.transpose(label, (2, 0, 1))

    image_dhw = normalize_mri(image_dhw)

    prob_map = sliding_window_prob(
        model=model,
        image=image_dhw,
        patch_size=(64, 96, 96),
        stride=(32, 48, 48),
        device=device,
    )

    pred_raw = (prob_map >= threshold).astype(np.uint8)
    pred_lcc = keep_largest_connected_component(pred_raw)

    raw_dice, raw_iou, raw_inter = dice_iou_numpy(pred_raw, label_dhw)
    lcc_dice, lcc_iou, lcc_inter = dice_iou_numpy(pred_lcc, label_dhw)

    raw_hd95 = hd95(pred_raw, label_dhw, spacing_dhw)
    lcc_hd95 = hd95(pred_lcc, label_dhw, spacing_dhw)

    raw_bbox = get_bbox(pred_raw)
    lcc_bbox = get_bbox(pred_lcc)
    gt_bbox = get_bbox(label_dhw)

    os.makedirs(output_dir, exist_ok=True)

    raw_hwd = np.transpose(pred_raw, (1, 2, 0))
    lcc_hwd = np.transpose(pred_lcc, (1, 2, 0))
    prob_hwd = np.transpose(prob_map, (1, 2, 0))

    raw_save_path = os.path.join(
        output_dir,
        case_name.replace(".nii.gz", "_raw_pred.nii.gz"),
    )

    lcc_save_path = os.path.join(
        output_dir,
        case_name.replace(".nii.gz", "_lcc_pred.nii.gz"),
    )

    prob_save_path = os.path.join(
        output_dir,
        case_name.replace(".nii.gz", "_prob.nii.gz"),
    )

    nib.save(
        nib.Nifti1Image(
            raw_hwd.astype(np.uint8),
            affine=image_nii.affine,
            header=image_nii.header,
        ),
        raw_save_path,
    )

    nib.save(
        nib.Nifti1Image(
            lcc_hwd.astype(np.uint8),
            affine=image_nii.affine,
            header=image_nii.header,
        ),
        lcc_save_path,
    )

    nib.save(
        nib.Nifti1Image(
            prob_hwd.astype(np.float32),
            affine=image_nii.affine,
            header=image_nii.header,
        ),
        prob_save_path,
    )

    result = {
        "case": case_name,
        "raw_dice": raw_dice,
        "raw_iou": raw_iou,
        "raw_hd95_mm": raw_hd95,
        "raw_intersection": raw_inter,
        "raw_pred_voxels": int(pred_raw.sum()),
        "lcc_dice": lcc_dice,
        "lcc_iou": lcc_iou,
        "lcc_hd95_mm": lcc_hd95,
        "lcc_intersection": lcc_inter,
        "lcc_pred_voxels": int(pred_lcc.sum()),
        "gt_voxels": int(label_dhw.sum()),
        "raw_pred_path": raw_save_path,
        "lcc_pred_path": lcc_save_path,
        "prob_path": prob_save_path,
    }

    if raw_bbox is not None:
        result.update({
            "raw_z_min": raw_bbox["z_min"],
            "raw_z_max": raw_bbox["z_max"],
            "raw_y_min": raw_bbox["y_min"],
            "raw_y_max": raw_bbox["y_max"],
            "raw_x_min": raw_bbox["x_min"],
            "raw_x_max": raw_bbox["x_max"],
        })

    if lcc_bbox is not None:
        result.update({
            "lcc_z_min": lcc_bbox["z_min"],
            "lcc_z_max": lcc_bbox["z_max"],
            "lcc_y_min": lcc_bbox["y_min"],
            "lcc_y_max": lcc_bbox["y_max"],
            "lcc_x_min": lcc_bbox["x_min"],
            "lcc_x_max": lcc_bbox["x_max"],
        })

    if gt_bbox is not None:
        result.update({
            "gt_z_min": gt_bbox["z_min"],
            "gt_z_max": gt_bbox["z_max"],
            "gt_y_min": gt_bbox["y_min"],
            "gt_y_max": gt_bbox["y_max"],
            "gt_x_min": gt_bbox["x_min"],
            "gt_x_max": gt_bbox["x_max"],
        })

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = find_best_checkpoint()

    dataset_dir = os.path.join("data", "raw", "Task02_Heart")
    image_dir = os.path.join(dataset_dir, "imagesTr")
    label_dir = os.path.join(dataset_dir, "labelsTr")

    output_dir = os.path.join(
        "results",
        "predictions",
        "volume_eval_nnunet_split",
    )

    val_cases = load_nnunet_fold0_val_cases(
        split_path="results/nnunet_fold0_split.json"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet3D(
        in_channels=1,
        num_classes=checkpoint["num_classes"],
        base_channels=checkpoint["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 50)
    print("Evaluate Self-implemented 3D U-Net on nnU-Net Fold 0 Val Cases")
    print("=" * 50)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Val cases:", val_cases)

    results = []

    for idx, case_name in enumerate(val_cases, start=1):
        print()
        print("=" * 50)
        print(f"Evaluating case {idx}/{len(val_cases)}: {case_name}")
        print("=" * 50)

        result = evaluate_case(
            model=model,
            case_name=case_name,
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            device=device,
            threshold=0.5,
        )

        results.append(result)

        print(
            f"{case_name} | "
            f"Raw Dice: {result['raw_dice']:.4f}, Raw IoU: {result['raw_iou']:.4f}, Raw HD95: {result['raw_hd95_mm']:.2f} | "
            f"LCC Dice: {result['lcc_dice']:.4f}, LCC IoU: {result['lcc_iou']:.4f}, LCC HD95: {result['lcc_hd95_mm']:.2f}"
        )

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    csv_path = os.path.join(
        "results",
        "heart_volume_metrics_nnunet_split_with_hd95.csv",
    )

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
        "raw_pred_voxels",
        "lcc_pred_voxels",
        "gt_voxels",
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
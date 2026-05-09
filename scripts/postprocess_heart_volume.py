import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from model.unet3d import UNet3D
from datasets.heart_patch_dataset import normalize_mri


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
                count_map[z:z + pd, y:y + ph, x:x + pw] += 1.0

                count += 1
                if count % 20 == 0 or count == total:
                    print(f"Processed {count}/{total}")

    prob_map = prob_sum / np.maximum(count_map, 1e-8)
    return prob_map


def keep_largest_connected_component(mask):
    """
    只保留最大的 3D 连通区域。

    mask: [D, H, W], 0/1
    """
    mask = mask.astype(np.uint8)

    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask

    component_sizes = ndimage.sum(
        mask,
        labeled,
        index=range(1, num_features + 1)
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
    print("Postprocess Whole-volume Inference")
    print("=" * 50)
    print("Device:", device)
    print("Case:", case_name)

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

    pred_raw = (prob_map >= 0.5).astype(np.uint8)
    pred_lcc = keep_largest_connected_component(pred_raw)

    raw_dice, raw_iou, raw_inter = dice_iou_numpy(pred_raw, label_dhw)
    lcc_dice, lcc_iou, lcc_inter = dice_iou_numpy(pred_lcc, label_dhw)

    print()
    print("=" * 50)
    print("Raw Prediction")
    print("=" * 50)
    print("Dice:", raw_dice)
    print("IoU:", raw_iou)
    print("Intersection:", raw_inter)
    print("Pred voxels:", int(pred_raw.sum()))
    print("GT voxels:", int(label_dhw.sum()))
    print("Pred bbox:", get_bbox(pred_raw))

    print()
    print("=" * 50)
    print("After Largest Connected Component")
    print("=" * 50)
    print("Dice:", lcc_dice)
    print("IoU:", lcc_iou)
    print("Intersection:", lcc_inter)
    print("Pred voxels:", int(pred_lcc.sum()))
    print("GT voxels:", int(label_dhw.sum()))
    print("Pred bbox:", get_bbox(pred_lcc))
    print("GT bbox:", get_bbox(label_dhw))

    # 保存后处理结果
    os.makedirs(os.path.join("results", "predictions"), exist_ok=True)

    pred_raw_hwd = np.transpose(pred_raw, (1, 2, 0))
    pred_lcc_hwd = np.transpose(pred_lcc, (1, 2, 0))
    prob_hwd = np.transpose(prob_map, (1, 2, 0))

    raw_save_path = os.path.join(
        "results",
        "predictions",
        case_name.replace(".nii.gz", "_raw_pred.nii.gz"),
    )

    lcc_save_path = os.path.join(
        "results",
        "predictions",
        case_name.replace(".nii.gz", "_lcc_pred.nii.gz"),
    )

    prob_save_path = os.path.join(
        "results",
        "predictions",
        case_name.replace(".nii.gz", "_prob.nii.gz"),
    )

    nib.save(
        nib.Nifti1Image(
            pred_raw_hwd.astype(np.uint8),
            affine=image_nii.affine,
            header=image_nii.header,
        ),
        raw_save_path,
    )

    nib.save(
        nib.Nifti1Image(
            pred_lcc_hwd.astype(np.uint8),
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

    print()
    print("Raw prediction saved to:", raw_save_path)
    print("LCC prediction saved to:", lcc_save_path)
    print("Probability map saved to:", prob_save_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
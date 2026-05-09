import os
import sys
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_DIR)

from model.unet3d import UNet3D
from datasets.heart_patch_dataset import normalize_mri
from utils.metrics import dice_score, iou_score


def pad_to_min_size(image,patch_size):
    d,h,w = image.shape
    pd,ph,pw = patch_size
    pad_d = max(pd-d,0)
    pad_h = max(ph-h,0)
    pad_w = max(pw-w,0)

    padded = np.pad(
        image,
        ((0,pad_d),(0,pad_h),(0,pad_w)),
        mode = "constant",
        constant_values = 0,
    )
    return padded, (d,h,w)

def get_start_positions(size,patch_size,stride):

    if size <= patch_size:
        return [0]
    
    
    positions = list(range(0,size-patch_size+1,stride)) 

    if positions[-1] != size-patch_size:
        positions.append(size-patch_size)

    return positions

@torch.no_grad()

def sliding_window_inference(
    model,
    image,
    patch_size = (64,96,96),
    stride = (32,48,48),
    device = "cuda",
):
    model.eval()

    image,original_shape = pad_to_min_size(image,patch_size)

    d,h,w = image.shape
    pd,ph,pw = patch_size
    sd,sh,sw = stride

    prob_sum = np.zeros_like(image,dtype=np.float32)
    count_map = np.zeros_like(image,dtype=np.float32)

    z_starts = get_start_positions(d,pd,sd)
    y_starts = get_start_positions(h,ph,sh)
    x_starts = get_start_positions(w,pw,sw)

    total_patches = len(z_starts)*len(y_starts)*len(x_starts)
    patch_count = 0

    print(f"Sliding window patches: {total_patches}")

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
               patch = image[
                    z:z+pd,
                    y:y+ph,
                    x:x+pw,
            ]
               
            patch_tensor = torch.from_numpy(patch).float()
            patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(device)

            logits = model(patch_tensor)
            probs = F.softmax(logits,dim=1)

            prob = probs[0,1].cpu().numpy()

            prob_sum[z:z+pd,
                     y:y+ph,
                     x:x+pw] += prob
            
            count_map[z:z+pd,
                      y:y+ph,
                      x:x+pw] += 1.0
            
            patch_count += 1

            if patch_count % 20 == 0 or patch_count == total_patches:
                print(f"Processed {patch_count}/{total_patches} patches")

    prob_map = prob_sum / np.maximum(count_map,1e-8)

    od,oh,ow = original_shape
    prob_map = prob_map[:od,:oh,:ow]

    prob_mask = (prob_map >= 0.5).astype(np.uint8)

    return prob_mask, prob_map

def compute_numpy_dice_iou(probs,masks,smooth = 1e-8):
        prob = probs.astype(bool)
        mask = masks.astype(bool)

        intersection = np.logical_and(prob,mask).sum()
        prob_sum = prob.sum()
        mask_sum = mask.sum()

        dice = (2*intersection + smooth)/(prob_sum + mask_sum + smooth)
        iou = (intersection + smooth)/(prob_sum + mask_sum - intersection + smooth)

        return float(dice), float(iou)  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(PROJECT_DIR, "checkpoints","best_heart_unet3d_epoch19.pth")


    case_name = "la_003.nii.gz"

    image_path = os.path.join(PROJECT_DIR, "data","raw","Task02_Heart","imagesTr",case_name)
    label_path = os.path.join(PROJECT_DIR, "data","raw","Task02_Heart","labelsTr",case_name)

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Case {case_name} not found in dataset. Please ensure the case exists in 'data/raw/Task02_Heart/imagesTr' and 'data/raw/Task02_Heart/labelsTr'.")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the checkpoint exists for inference.")
    
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}. Please ensure the label file exists for the specified case.")
    

    print("=" * 50)
    print("Whole-volume Heart MRI Inference")
    print("=" * 50)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Case:", case_name)


    checkpoint = torch.load(checkpoint_path,map_location=device)

    model = UNet3D(
        in_channels = 1,
        num_classes = 2,
        base_channels = 16,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)


    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    label = (label > 0).astype(np.uint8)

    print("Image shape:", image.shape)
    print("Label shape:", label.shape)


    image_dhw = np.transpose(image,(2,0,1))
    label_dhw = np.transpose(label,(2,0,1))

    image_dhw = normalize_mri(image_dhw)

    pred_dhw, prob_map_dhw = sliding_window_inference(
        model = model,
        image = image_dhw,
        patch_size = (64,96,96),
        stride = (32,48,48),
        device = device,
    )

    dice,iou = compute_numpy_dice_iou(pred_dhw,label_dhw)

    print(f"Dice score: {dice:.4f}")
    print(f"IoU score: {iou:.4f}")
    print("Pred positive voxels:", pred_dhw.sum())
    print("GT positive voxels:", label_dhw.sum())

    pred_hwd = np.transpose(pred_dhw,(1,2,0))
    prob_map_hwd = np.transpose(prob_map_dhw,(1,2,0))

    os.makedirs(os.path.join(PROJECT_DIR, "results","predictions"),exist_ok = True)

    pred_save_path = os.path.join(PROJECT_DIR,
                                  "results",
                                  "predictions",
                                  case_name.replace(".nii.gz", "_pred.nii.gz"))


    prob_save_path = os.path.join(PROJECT_DIR,
                                  "results",
                                  "predictions",
                                  case_name.replace(".nii.gz", "_prob.nii.gz"))
    pred_nii = nib.Nifti1Image(
        pred_hwd.astype(np.uint8),
        affine=image_nii.affine,
        header=image_nii.header,
    )

    prob_nii = nib.Nifti1Image(
        prob_map_hwd.astype(np.float32),
        affine=image_nii.affine,
        header=image_nii.header,
    )

    nib.save(pred_nii, pred_save_path)
    nib.save(prob_nii, prob_save_path)

    print("Prediction mask saved to:", pred_save_path)
    print("Probability map saved to:", prob_save_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
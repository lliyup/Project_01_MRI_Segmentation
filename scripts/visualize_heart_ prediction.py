import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from model.unet3d import UNet3D
from datasets.heart_patch_dataset import HeartPatchDataset,list_nii_files
from utils.metrics import dice_score, iou_score, logits_to_prediction

def choose_largest_mask_slice(mask):
    areas = mask.sum(axis=(1,2))
    return int(np.argmax(areas))


def normalize_slice(image_slice):
    p1 = np.percentile(image_slice,1)
    p99 = np.percentile(image_slice,99)

    image_slice = np.clip(image_slice,p1,p99)
    image_slice = (image_slice - image_slice.min())/(image_slice.max()-image_slice.min()+1e-8)
    return image_slice


def create_error_map(image,probs):

    gt = image.astype(bool)
    probs = probs.astype(bool)

    tp = gt & probs
    fp = ~gt & probs
    fn = gt & ~probs

    error_map = np.zeros_like(image,dtype=np.uint8)
    error_map[tp] = 1
    error_map[fp] = 2
    error_map[fn] = 3

    return error_map

def compute_error_map(image,probs):
    gt = image.astype(bool)
    probs = probs.astype(bool)

    tp = gt & probs
    fp = ~gt & probs
    fn = gt & ~probs

    tp = np.logical_and(gt,probs).sum()
    fp = np.logical_and(~gt,probs).sum()
    fn = np.logical_and(gt,~probs).sum()

    return int(tp),int(fp),int(fn)

def visualize_prediction(image,mask,probs,save_path,dice,iou):
    slice_idx = choose_largest_mask_slice(mask)
    
    
    image_slice = normalize_slice(image[slice_idx])
    mask_slice = mask[slice_idx]
    probs_slice = probs[slice_idx]

    error_map = create_error_map(mask_slice,probs_slice)

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.imshow(image_slice,cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(mask_slice,cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(probs_slice,cmap="gray") 
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(image_slice,cmap="gray")
    plt.imshow(error_map,alpha=0.5,cmap="jet")
    plt.title(f"Error Map\nDice: {dice:.4f}, IoU: {iou:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path,dpi=200)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    checkpoint_path = os.path.join("checkpoints","best_heart_unet3d_epoch19.pth")

    checkpoint = torch.load(checkpoint_path,map_location=device)
    model = UNet3D(
        in_channels = 1,
        num_classes = 2,
        base_channels = 16,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_cases = checkpoint.get("val_cases",None)

    if val_cases is None:
        raise RuntimeError("No val_cases found in checkpoint. Please ensure the checkpoint contains 'val_cases' for visualization.")
    
    print("visualize cases:",val_cases)

    dataset = HeartPatchDataset(
        dataset_dir = "data/raw/Task02_Heart",
        patch_size = (64,96,96),
        sample_per_case = 2,
        positive_rate = 1.0,
    )

    image,mask,case_name = dataset[0]
    image_input = image.unsqueeze(0).to(device)
    mask_input = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_input)
        probs = logits_to_prediction(logits)

    dice = dice_score(probs,mask_input,class_id = 1)
    iou = iou_score(probs,mask_input,class_id = 1)

    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    probs_np = probs.squeeze().cpu().numpy()

    tp,fp,fn = compute_error_map(mask_np,probs_np)

    os.makedirs("figures",exist_ok = True)
    save_path = os.path.join("figures",f"{case_name}_prediction_visualization.png")

    visualize_prediction(image_np,mask_np,probs_np,save_path,dice,iou)

    print("=" * 50)
    print("Heart MRI Prediction Visualization")
    print("=" * 50)
    print("Checkpoint:", checkpoint_path)
    print("Case:", case_name)
    print("Patch image shape:", image.shape)
    print("Patch mask shape:", mask.shape)
    print("Dice:", dice)
    print("IoU:", iou)
    print("TP voxels:", tp)
    print("FN voxels:", fn, "  # missed target voxels")
    print("FP voxels:", fp, "  # over-segmented voxels")
    print("Figure saved to:", save_path)
    print("=" * 50)

if __name__ == "__main__":
        main()
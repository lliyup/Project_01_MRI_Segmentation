import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)


from model.unet3d import UNet3D
from utils.metrics import dice_score,iou_score,logits_to_prediction
from scripts.train_synthetic_3dunet import SyntheticSphereDataset


def choose_largest_mask_slice(mask):
        areas = mask.sum(axis=(1,2))
        slice_idx = int(np.argmax(areas))
        return slice_idx



def create_error_map(mask,prediction):
    mask = mask.astype(bool)
    prediction = prediction.astype(bool)

    tp = mask & prediction
    fn = mask & ~prediction
    fp = ~mask & prediction


    error_map = np.zeros_like(mask,dtype = np.uint8)
    error_map[tp] = 1  # True Positives
    error_map[fn] = 2  # False Negatives
    error_map[fp] = 3  # False Positives

    return error_map


def visualize_error_case(image,mask,probs,save_path):
    slice_idx = choose_largest_mask_slice(mask)


    image_slice = image[slice_idx]
    mask_slice = mask[slice_idx]
    prob_slice = probs[slice_idx]

    error_map = create_error_map(mask_slice,prob_slice)

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.title("Image Slice")    
    plt.imshow(image_slice,cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Ground Truth Mask")  
    plt.imshow(mask_slice,cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Predicted Mask")
    plt.imshow(prob_slice,cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Error Map")
    plt.imshow(image_slice,cmap="gray")
    plt.imshow(error_map,alpha = 0.65,vmin = 0,vmax = 3)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path,dpi =200)
    plt.close()


def compute_error_counts(mask,probs):


    gt = mask.astype(bool)
    pred = probs.astype(bool)

    tp = np.logical_and(gt,pred).sum()
    fn = np.logical_and(gt,~pred).sum()
    fp = np.logical_and(~gt,pred).sum()


    return tp,fn,fp


def main():
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

    check_point_path = os.path.join(PROJECT_ROOT,"checkpoints","synthetic_3dunet.pth")

    if not os.path.exists(check_point_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {check_point_path}. "
            f"Please train the model first."
        )
    
    checkpoint = torch.load(check_point_path,map_location = device)

    model = UNet3D(
        in_channels = 1,
        num_classes = checkpoint["num_classes"],
        base_channels = checkpoint["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    dataset = SyntheticSphereDataset(
        num_samples = 1,
        image_size = 64,
    )

    image,mask = dataset[0]

    image_input = image.unsqueeze(0).to(device)
    mask_input = mask.unsqueeze(0).to(device)


    with torch.no_grad():
        logits = model(image_input)
        probs = logits_to_prediction(logits)

        dice = dice_score(probs,mask_input,class_id = 1)
        iou = iou_score(probs,mask_input,class_id = 1)


        image_np = image.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        probs_np = probs.squeeze(0).cpu().numpy()

        
        save_dir = os.path.join(PROJECT_ROOT,"results")
        os.makedirs(save_dir,exist_ok = True)
        save_path = os.path.join(save_dir,"synthetic_error_visualization.png")

        visualize_error_case(image_np,mask_np,probs_np,save_path)

        tp,fn,fp = compute_error_counts(mask_np,probs_np)

        print("=" * 50)
        print("Synthetic Error Visualization")
        print("=" * 50)
        print("Dice:", dice)
        print("IoU:", iou)
        print("TP voxels:", tp)
        print("FN voxels:", fn, "  # missed target voxels")
        print("FP voxels:", fp, "  # over-segmented voxels")
        print("Figure saved to:", save_path)
        print("=" * 50)


if __name__ == "__main__":
    main()
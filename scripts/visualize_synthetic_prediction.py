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


#定义二维切片可视化函数
def visualize_middle_slice(image,mask,prob,save_path):
    #选择z轴中间切片进行可视化
    d =image.shape[0]
    slice_idx = d//2
    image_slice = image[slice_idx]
    mask_slice = mask[slice_idx]
    prediction_slice = prob[slice_idx]

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
    plt.imshow(prediction_slice,cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Overlay")
    plt.imshow(image_slice,cmap="gray")
    plt.imshow(mask_slice,alpha=0.35)
    plt.imshow(prediction_slice,alpha=0.35)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path,dpi=200)
    plt.close()


def main():
        device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

        #加载检查点
        checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "synthetic_3dunet.pth")
        #检查点文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please train the model first.")
        #加载检查点到指定设备
        checkpoint = torch.load(checkpoint_path,map_location = device)

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
        #将图像和掩码添加批次维度并移动到设备上
        image_input = image.unsqueeze(0).to(device)
        mask_input = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_input)
            probs = logits_to_prediction(logits)

            dice = dice_score(probs,mask_input,class_id = 1)
            iou = iou_score(probs,mask_input,class_id = 1)
            #将张量转化为numpy数组进行可视化
            image_np = image.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()
            probs_np = probs.squeeze(0).cpu().numpy()


            save_dir = os.path.join(PROJECT_ROOT, "figures")
            os.makedirs(save_dir,exist_ok = True)
            save_path = os.path.join(save_dir, "synthetic_prediction_visualization.png")


            visualize_middle_slice(image_np,mask_np,probs_np,save_path)
            print("=" * 50)
            print("Synthetic Prediction Visualization")
        print("=" * 50)
        print("Dice:", dice)
        print("IoU:", iou)
        print("Figure saved to:", save_path)
        print("=" * 50)


if __name__ == "__main__":
    main()

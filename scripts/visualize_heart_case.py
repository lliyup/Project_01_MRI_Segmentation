import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def choose_largest_mask_slice(mask):
    areas = mask.sum(axis=(0, 1))
    #求前景面积最大的切片索引
    slice_idx = int(np.argmax(areas))
    return slice_idx


def normalize_image_slice(image_slice):
    #裁剪图像强度范围，避免极端化影响
    p1 = np.percentile(image_slice, 1)
    p99 = np.percentile(image_slice, 99)

    image_slice = np.clip(image_slice, p1, p99)
    #归一化到0-1
    image_slice = (image_slice - image_slice.min())/(image_slice.max()-image_slice.min()+1e-8)
    return image_slice

def visualize_heart_case(image,mask,save_path):
    slice_idx = choose_largest_mask_slice(mask)


    image_slice = image[:,:,slice_idx]
    mask_slice = mask[:,:,slice_idx]
    image_slice = normalize_image_slice(image_slice)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image_slice,cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_slice,cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(image_slice,cmap="gray")
    plt.imshow(mask_slice,alpha=0.35,cmap="jet")
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path,dpi=200)
    plt.close()

    return slice_idx


def main():
    #数据集路径
    dataset_dir = os.path.join("data","raw","Task02_Heart")
    image_dir = os.path.join(dataset_dir,"imagesTr")
    label_dir = os.path.join(dataset_dir,"labelsTr")
    #检查数据集目录和文件
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(".nii.gz") and not f.startswith("._")

    ])
    #检查是否有文件存在
    if len(image_files) == 0:
        raise RuntimeError("No .nii.gz files found in imagesTr. Please check the dataset.")
    
    case_name = image_files[0]
    

    image_path = os.path.join(image_dir,case_name)
    label_path = os.path.join(label_dir,case_name)
    #读取图像和标签
    image = nib.load(image_path).get_fdata()
    mask = nib.load(label_path).get_fdata()
    #将mask二值化，确保只有0和1两类
    mask = (mask > 0).astype(np.uint8)
    #创建保存目录
    os.makedirs("figures",exist_ok = True)
    save_path = os.path.join("figures","heart_case_visualization.png")

    slice_idx = visualize_heart_case(image,mask,save_path)

    print("=" * 50)
    print("Heart MRI Visualization")
    print("=" * 50)
    print("Case:", case_name)
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Selected slice:", slice_idx)
    print("Mask voxels:", int(mask.sum()))
    print("Figure saved to:", save_path)
    print("=" * 50)

if __name__ == "__main__":
    main()
import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def list_nii_files(folder):
    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])
    return files

def normalize_mri(image):
    image = image.astype(np.float32)

    foreground = image[image>0]

    if foreground.size > 0:
        mean = foreground.mean()
        std = foreground.std()

    else:
        mean = image.mean()
        std = image.std()
    
    image = (image-mean)/(std+1e-8)
    image = np.clip(image,-5,5)

    return image.astype(np.float32)

def crop_patch(image,mask,patch_size = (64,96,96),positive_rate = 0.8):
    
    d,h,w = image.shape
    pd,ph,pw = patch_size

    pad_d = max(pd-d,0)
    pad_h = max(ph-h,0)
    pad_w = max(pw-w,0)


    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(
           image,
           ((0,pad_d),(0,pad_h),(0,pad_w)),
           mode = "constant",
              constant_values = 0,
       )


        mask = np.pad(
            mask,
            ((0,pad_d),(0,pad_h),(0,pad_w)),
            mode = "constant",
            constant_values = 0,
        )
        d,h,w = image.shape


    use_positive = random.random() < positive_rate and mask.sum()>0

    if use_positive:
        coords = np.argwhere(mask>0)
        center = coords[random.randint(0,len(coords)-1)]
        cz,cy,cx = center
    else:
        cz = random.randint(0,d-1)
        cy = random.randint(0,h-1)
        cx = random.randint(0,w-1)

    z1 = int(cz - pd//2)
    y1 = int(cy - ph//2)
    x1 = int(cx - pw//2)

    z1 = max(0,min(z1,d-pd))
    y1 = max(0,min(y1,h-ph))
    x1 = max(0,min(x1,w-pw))


    z2 = z1+pd
    y2 = y1+ph
    x2 = x1+pw

    image_patch = image[z1:z2,y1:y2,x1:x2]
    mask_patch = mask[z1:z2,y1:y2,x1:x2]

    return image_patch,mask_patch


class HeartPatchDataset(Dataset):
    def __init__(
            self,
            dataset_dir = "data/raw/Task02_Heart",
            patch_size = (64,96,96),
            case_names = None,
            sample_per_case = 8,
            positive_rate = 0.8,
    ):
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.sample_per_case = sample_per_case
        self.positive_rate = positive_rate
        self.image_dir = os.path.join(self.dataset_dir,"imagesTr")
        self.labels_dir = os.path.join(self.dataset_dir,"labelsTr") 


        if case_names is None:
            self.case_names = list_nii_files(self.image_dir)
        else:
            self.case_names = case_names

        self.index_map = []
        for case_idx in range(len(self.case_names)):
            for _ in range(self.sample_per_case):
                self.index_map.append(case_idx)

        
    def __len__(self):
            return len(self.index_map)
        

    def __getitem__(self,idx):
            case_idx = self.index_map[idx]
            case_name = self.case_names[case_idx]


            image_path = os.path.join(self.image_dir,case_name)
            label_path = os.path.join(self.labels_dir,case_name)

            image = nib.load(image_path).get_fdata()
            mask = nib.load(label_path).get_fdata()

            mask = (mask > 0).astype(np.uint8)
            image = normalize_mri(image)

            image = np.transpose(image,(2,0,1))
            mask = np.transpose(mask,(2,0,1))

            image_patch,mask_patch = crop_patch(
                image = image,
                mask = mask,
                patch_size = self.patch_size,
                positive_rate = self.positive_rate,
            )


            image_patch = np.expand_dims(image_patch,axis = 0)

            image_patch = torch.from_numpy(image_patch).float()
            mask_patch = torch.from_numpy(mask_patch).long()

            return image_patch,mask_patch,case_name
        
if __name__ == "__main__":
        dataset = HeartPatchDataset(
        dataset_dir="data/raw/Task02_Heart",
        patch_size=(64, 96, 96),
        sample_per_case=2,
        positive_rate=1.0,
        )

        print("=" * 50)
        print("HeartPatchDataset Test")
        print("=" * 50)
        print("Dataset length:", len(dataset))

        image, mask, case_name = dataset[0]

        print("Case:", case_name)
        print("Image patch shape:", image.shape)
        print("Mask patch shape:", mask.shape)
        print("Image patch min/max:", image.min().item(), image.max().item())
        print("Mask unique values:", torch.unique(mask))
        print("Mask positive voxels:", int(mask.sum().item()))
        print("=" * 50)
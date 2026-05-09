import os
import json
import nibabel as nib
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)


def main():
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "raw", "Task02_Heart")

    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    json_path = os.path.join(dataset_dir, "dataset.json")

    print("=" * 50)
    print("Check MSD Task02 Heart Dataset")
    print("=" * 50)

    print("Dataset dir:", dataset_dir)
    print("imagesTr exists:", os.path.exists(images_dir))
    print("labelsTr exists:", os.path.exists(labels_dir))
    print("dataset.json exists:", os.path.exists(json_path))

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("imagesTr or labelsTr directory is missing. Please check the dataset.")
        return
    
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])
    label_files = sorted([
        f for f in os.listdir(labels_dir)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])

    print("Number of training images:", len(image_files))
    print("Number of training labels:", len(label_files))

    if len(image_files) == 0:
        print("No .nii.gz files found in imagesTr. Please check the dataset.")
        return
    
    first_image = image_files[0]
    first_label = first_image

    image_path = os.path.join(images_dir, first_image)
    label_path = os.path.join(labels_dir, first_label)

    print("First image file:", first_image)
    print("First label file:", first_label)

    img_nii = nib.load(image_path)
    lbl_nii = nib.load(label_path)

    img = img_nii.get_fdata()
    lbl = lbl_nii.get_fdata()

    print("Image shape:", img.shape)
    print("Label shape:", lbl.shape)
    print("Image dtype:", img.dtype)
    print("Label unique values:", np.unique(lbl))
    print("Image intensity min/max:", img.min(), img.max())
    print("Voxel spacing:", img_nii.header.get_zooms())

    if os.path.exists(json_path):
        with open(json_path, "r",encoding = "utf-8") as f:
            meta = json.load(f)

        print("Dataset name:",meta.get("name"))
        print("Modality:", meta.get("modality"))
        print("Labels:", meta.get("labels"))

    print("=" * 50)
    print("Dataset check finished.")
    print("=" * 50)

if __name__ == "__main__":
    main()

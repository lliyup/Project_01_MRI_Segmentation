import os
import json
import shutil


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def list_nii_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])


def main():
    source_dir = os.path.join("data", "raw", "Task02_Heart")
    source_images_tr = os.path.join(source_dir, "imagesTr")
    source_labels_tr = os.path.join(source_dir, "labelsTr")
    source_images_ts = os.path.join(source_dir, "imagesTs")

    target_root = os.path.join(
        "nnUNet_workspace",
        "nnUNet_raw",
        "Dataset002_Heart",
    )

    target_images_tr = os.path.join(target_root, "imagesTr")
    target_labels_tr = os.path.join(target_root, "labelsTr")
    target_images_ts = os.path.join(target_root, "imagesTs")

    safe_mkdir(target_images_tr)
    safe_mkdir(target_labels_tr)
    safe_mkdir(target_images_ts)

    train_files = list_nii_files(source_images_tr)

    print("=" * 50)
    print("Prepare MSD Heart for nnU-Net v2")
    print("=" * 50)
    print("Source:", source_dir)
    print("Target:", target_root)
    print("Training cases:", len(train_files))

    for fname in train_files:
        case_id = fname.replace(".nii.gz", "")

        src_img = os.path.join(source_images_tr, fname)
        src_lab = os.path.join(source_labels_tr, fname)

        dst_img = os.path.join(target_images_tr, f"{case_id}_0000.nii.gz")
        dst_lab = os.path.join(target_labels_tr, f"{case_id}.nii.gz")

        if not os.path.exists(src_lab):
            raise FileNotFoundError(f"Missing label for {fname}: {src_lab}")

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lab, dst_lab)

    if os.path.exists(source_images_ts):
        test_files = list_nii_files(source_images_ts)
        print("Test cases:", len(test_files))

        for fname in test_files:
            case_id = fname.replace(".nii.gz", "")
            src_img = os.path.join(source_images_ts, fname)
            dst_img = os.path.join(target_images_ts, f"{case_id}_0000.nii.gz")
            shutil.copy2(src_img, dst_img)
    else:
        test_files = []
        print("No imagesTs found, skipped test images.")

    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "left_atrium": 1
        },
        "numTraining": len(train_files),
        "file_ending": ".nii.gz",
        "name": "Heart",
        "description": "MSD Task02 Heart converted to nnU-Net v2 format"
    }

    json_path = os.path.join(target_root, "dataset.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print("dataset.json saved to:", json_path)
    print("Done.")
    print("=" * 50)


if __name__ == "__main__":
    main()
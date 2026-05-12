import os
import json


def main():
    split_path = r"D:\nnunet_work\nnUNet_preprocessed\Dataset002_Heart\splits_final.json"

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cannot find split file: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    print("=" * 50)
    print("nnU-Net Splits")
    print("=" * 50)
    print("Number of folds:", len(splits))

    fold_id = 0
    fold = splits[fold_id]

    train_cases = fold["train"]
    val_cases = fold["val"]

    print("Fold:", fold_id)
    print("Train cases:", len(train_cases), train_cases)
    print("Val cases:", len(val_cases), val_cases)

    os.makedirs("results", exist_ok=True)

    output = {
        "fold": fold_id,
        "train_cases": train_cases,
        "val_cases": val_cases,
    }

    save_path = os.path.join("results", "nnunet_fold0_split.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("Saved split to:", save_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
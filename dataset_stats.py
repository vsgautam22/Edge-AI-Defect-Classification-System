import os
from collections import defaultdict

DATASET_DIR = "dataset"
SPLITS = ["train", "val", "test"]

def count_images():
    total_images = 0
    split_counts = {}
    class_counts = defaultdict(int)

    for split in SPLITS:
        split_path = os.path.join(DATASET_DIR, split)
        split_total = 0

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                continue

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            count = len(images)
            split_total += count
            class_counts[class_name] += count

        split_counts[split] = split_total
        total_images += split_total

    print("\n=== DATASET SUMMARY ===")
    print(f"Total images in dataset: {total_images}\n")

    print("Images per split:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")

    print("\nImages per class (across all splits):")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    count_images()

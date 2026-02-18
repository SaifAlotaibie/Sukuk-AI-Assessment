"""
Interactive labeling script — resumes from where you left off.
Run:  python label_mini.py
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PAGES_DIR = "pages_raw"
CSV_PATH = "mini_labels.csv"

LABEL_MAP = {
    "1": "Independent Auditor's Report",
    "2": "Financial Sheets",
    "3": "Notes (Tabular)",
    "4": "Notes (Text)",
    "5": "Other Pages",
}


def natural_sort_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def load_existing_labels():
    labeled = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] != "image_name":
                    labeled.add(row[0])
    return labeled


def get_all_images():
    files = [
        f for f in os.listdir(PAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort(key=natural_sort_key)
    return files


def main():
    all_images = get_all_images()
    labeled = load_existing_labels()
    unlabeled = [img for img in all_images if img not in labeled]

    total = len(all_images)
    labeled_count = len(labeled)

    if not unlabeled:
        print("All images have been labeled.")
        return

    print(f"\n{'='*50}")
    print(f"  Labeled: {labeled_count} / {total}")
    print(f"  Remaining: {len(unlabeled)}")
    print(f"{'='*50}")
    print("  1 = Independent Auditor's Report")
    print("  2 = Financial Sheets")
    print("  3 = Notes (Tabular)")
    print("  4 = Notes (Text)")
    print("  5 = Other Pages")
    print("  q = Quit")
    print(f"{'='*50}\n")

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_name", "label"])

        for i, img_name in enumerate(unlabeled, start=1):
            img_path = os.path.join(PAGES_DIR, img_name)
            img = mpimg.imread(img_path)

            fig, ax = plt.subplots(figsize=(10, 14))
            ax.imshow(img)
            ax.set_title(
                f"[{labeled_count + i} / {total}]  {img_name}",
                fontsize=12,
            )
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

            while True:
                choice = input(
                    f"  Label for {img_name} (1-5 / q): "
                ).strip()
                if choice == "q":
                    plt.close("all")
                    print("Session saved. Run again to continue.")
                    return
                if choice in LABEL_MAP:
                    break
                print("    Invalid input. Enter 1-5 or q.")

            writer.writerow([img_name, LABEL_MAP[choice]])
            f.flush()
            plt.close("all")

            remaining = len(unlabeled) - i
            print(
                f"    -> {LABEL_MAP[choice]}  "
                f"(Labeled: {labeled_count + i} / {total}, "
                f"Remaining: {remaining})\n"
            )

    print(f"\nDone! All {total} images are now labeled.")
    print(f"Results in {CSV_PATH}")


if __name__ == "__main__":
    main()

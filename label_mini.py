"""
Mini-labeling script — automatically continues with the next 3 unlabeled PDFs.
Run:  python label_mini.py
"""

import os
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PAGES_DIR = "pages_raw"
CSV_PATH = "mini_labels.csv"
BATCH_SIZE = 3

LABEL_MAP = {
    "1": "Independent Auditor's Report",
    "2": "Financial Sheets",
    "3": "Notes (Tabular)",
    "4": "Notes (Text)",
    "5": "Other Pages",
}


def natural_sort_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def extract_prefix(filename):
    """FS1_page_3.jpg -> FS1"""
    m = re.match(r"^(.+?)_page_\d+", filename)
    return m.group(1) if m else None


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


def find_next_prefixes(all_images, labeled):
    """Return the next BATCH_SIZE prefixes that are not fully labeled."""
    prefix_images = defaultdict(list)
    for img in all_images:
        prefix = extract_prefix(img)
        if prefix:
            prefix_images[prefix].append(img)

    sorted_prefixes = sorted(prefix_images.keys(), key=natural_sort_key)

    next_prefixes = []
    for prefix in sorted_prefixes:
        images = prefix_images[prefix]
        if not all(img in labeled for img in images):
            next_prefixes.append(prefix)
            if len(next_prefixes) == BATCH_SIZE:
                break

    return next_prefixes, prefix_images


def main():
    all_images = get_all_images()
    labeled = load_existing_labels()

    next_prefixes, prefix_images = find_next_prefixes(all_images, labeled)

    if not next_prefixes:
        print("All PDFs are fully labeled. Nothing to do.")
        return

    batch_images = []
    for prefix in next_prefixes:
        for img in prefix_images[prefix]:
            if img not in labeled:
                batch_images.append(img)

    total = len(batch_images)
    already_done = len(labeled)

    print(f"\n{'='*55}")
    print(f"  Next PDFs to label: {', '.join(next_prefixes)}")
    print(f"  Images in this batch: {total}")
    print(f"  Already labeled overall: {already_done}")
    print(f"{'='*55}")
    print("  1 = Independent Auditor's Report")
    print("  2 = Financial Sheets")
    print("  3 = Notes (Tabular)")
    print("  4 = Notes (Text)")
    print("  5 = Other Pages")
    print("  q = Quit")
    print(f"{'='*55}\n")

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_name", "label"])

        for i, img_name in enumerate(batch_images, start=1):
            prefix = extract_prefix(img_name)
            img_path = os.path.join(PAGES_DIR, img_name)
            img = mpimg.imread(img_path)

            fig, ax = plt.subplots(figsize=(10, 14))
            ax.imshow(img)
            ax.set_title(f"[{prefix}]  ({i} / {total})  {img_name}", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

            while True:
                choice = input(f"  [{prefix}] Label for {img_name} (1-5 / q): ").strip()
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
            print(f"    -> {LABEL_MAP[choice]}   ({i}/{total} in batch)\n")

    print(f"\nDone! Labeled {total} images for: {', '.join(next_prefixes)}")
    print(f"Results in {CSV_PATH}")


if __name__ == "__main__":
    main()

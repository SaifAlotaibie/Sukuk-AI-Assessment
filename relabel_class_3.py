"""
Re-label images previously assigned "Notes (Tabular)" in mini_labels.csv.
Run:  python relabel_class_3.py
"""

import os
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

TARGET_LABEL = "Notes (Tabular)"


def load_csv():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)
    return header, rows


def save_csv(header, rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    header, rows = load_csv()

    review_indices = [i for i, row in enumerate(rows) if row[1] == TARGET_LABEL]
    total = len(review_indices)

    if total == 0:
        print("No images with label 'Notes (Tabular)' found. Nothing to review.")
        return

    print(f"\n{'='*50}")
    print(f"  Re-labeling '{TARGET_LABEL}'  |  {total} images")
    print(f"{'='*50}")
    print("  1 = Independent Auditor's Report")
    print("  2 = Financial Sheets")
    print("  3 = Notes (Tabular)")
    print("  4 = Notes (Text)")
    print("  5 = Other Pages")
    print("  Enter = keep current (3)")
    print("  q = quit and save progress")
    print(f"{'='*50}\n")

    changed = 0

    for count, idx in enumerate(review_indices, start=1):
        img_name = rows[idx][0]
        img_path = os.path.join(PAGES_DIR, img_name)

        img = mpimg.imread(img_path)
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.imshow(img)
        ax.set_title(f"Reviewing {count} / {total}  |  {img_name}  |  Current: {TARGET_LABEL}", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        while True:
            choice = input(f"  New label for {img_name} (1-5 / Enter=keep / q): ").strip()
            if choice == "":
                break
            if choice == "q":
                plt.close("all")
                save_csv(header, rows)
                print(f"\nSaved. {changed} label(s) changed so far.")
                return
            if choice in LABEL_MAP:
                if LABEL_MAP[choice] != TARGET_LABEL:
                    rows[idx][1] = LABEL_MAP[choice]
                    changed += 1
                    print(f"    -> {LABEL_MAP[choice]}")
                break
            print("    Invalid input. Enter 1-5, Enter, or q.")

        plt.close("all")

    save_csv(header, rows)
    print(f"\nDone! Reviewed {total} images. {changed} label(s) changed.")
    print(f"Updated {CSV_PATH}")


if __name__ == "__main__":
    main()

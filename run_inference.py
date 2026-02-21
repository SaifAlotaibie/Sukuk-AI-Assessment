"""
Standalone inference script for Sukuk document page classification.

Hybrid approach: EfficientNet-B0 CNN (65%) + Arabic OCR title matching (35%).

Usage:
    1. Set FILE_PATH below to the path of your PDF or image file.
    2. Run:  python run_inference.py
"""

# ============================================================
# SET YOUR FILE PATH HERE
# ============================================================
FILE_PATH = "PUT_YOUR_FILE_PATH_HERE"
# ============================================================

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

CLASSES = [
    "Financial Sheets",
    "Independent Auditor's Report",
    "Notes (Tabular)",
    "Notes (Text)",
    "Other Pages",
]

IMG_SIZE = 384

W_CNN = 0.65
W_OCR = 0.35

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_sukuk_model.pth"

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\u0640", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ocr_read_title(img: Image.Image):
    import pytesseract

    width, height = img.size
    cropped = img.crop((0, 0, width, int(height * 0.39)))

    data = pytesseract.image_to_data(
        cropped, lang="ara", output_type=pytesseract.Output.DICT
    )

    words = []
    confidences = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if word != "" and conf > 0:
            words.append(word)
            confidences.append(conf)

    if len(confidences) == 0:
        return "", 0, None

    avg_conf = np.mean(confidences)
    full_text = clean_text(" ".join(words))

    title_part = full_text[:150]

    if "تقرير المراجع المستقل" in title_part:
        return full_text, avg_conf, "auditor"

    if "قائمة المركز المالي" in title_part:
        return full_text, avg_conf, "financial"

    if "قائمة الدخل" in title_part:
        return full_text, avg_conf, "financial"

    if "قائمة التدفقات النقدية" in title_part:
        return full_text, avg_conf, "financial"

    if "قائمة التغيرات في حقوق الملكية" in title_part:
        return full_text, avg_conf, "financial"

    if "إيضاحات حول القوائم المالية" in title_part:
        return full_text, avg_conf, "notes"

    return full_text, avg_conf, None


def load_pages(file_path: Path) -> list:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from pdf2image import convert_from_path

        print(f"Converting PDF to images (200 DPI)...")
        pages = convert_from_path(str(file_path), dpi=200)
        print(f"  {len(pages)} pages extracted.")
        return pages

    elif suffix in (".jpg", ".jpeg", ".png"):
        img = Image.open(file_path).convert("RGB")
        return [img]

    else:
        print(f"Error: Unsupported file type '{suffix}'. Use PDF, JPG, JPEG, or PNG.")
        sys.exit(1)


def main():
    file_path = Path(FILE_PATH)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found: {MODEL_PATH}")
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    print("Model loaded.\n")

    pages = load_pages(file_path)

    predictions = []

    for i, page in enumerate(pages):
        img = page.convert("RGB")

        img_tensor = inference_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            cnn_probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        text, ocr_conf, detected_type = ocr_read_title(img)
        conf_factor = ocr_conf / 100.0

        ocr_scores = np.zeros(len(CLASSES))

        if detected_type == "auditor":
            ocr_scores[CLASSES.index("Independent Auditor's Report")] = conf_factor

        elif detected_type == "financial":
            ocr_scores[CLASSES.index("Financial Sheets")] = conf_factor

        elif detected_type == "notes":
            ocr_scores[CLASSES.index("Notes (Text)")] = 0.5 * conf_factor
            ocr_scores[CLASSES.index("Notes (Tabular)")] = 0.5 * conf_factor

            mask = np.zeros(len(CLASSES))
            mask[CLASSES.index("Notes (Text)")] = 1
            mask[CLASSES.index("Notes (Tabular)")] = 1
            cnn_probs = cnn_probs * mask

        final_scores = (W_CNN * cnn_probs) + (W_OCR * ocr_scores)
        final_idx = np.argmax(final_scores)
        predicted_label = CLASSES[final_idx]

        predictions.append(predicted_label)

        print(f"\nPage {i + 1}")
        print("OCR:", detected_type, "| conf:", round(ocr_conf, 1))
        print("CNN after mask:", dict(zip(CLASSES, np.round(cnn_probs, 3))))
        print("FINAL:", dict(zip(CLASSES, np.round(final_scores, 3))))
        print("Prediction:", predicted_label)

        plt.figure(figsize=(6, 8))
        plt.imshow(img)
        plt.title(predicted_label)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

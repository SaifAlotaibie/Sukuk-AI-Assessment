# ============================================================
# Sukuk Hybrid Inference (Cross-Platform Version)
# EfficientNet-B0 + EasyOCR + PyMuPDF
# ============================================================

import fitz
import easyocr
import numpy as np
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

PDF_PATH = "" ## PUT YOUR PDF PATH HERE
MODEL_PATH = "best_sukuk_model.pth"

classes = [
    "Financial Sheets",
    "Independent Auditor's Report",
    "Notes (Tabular)",
    "Notes (Text)",
    "Other Pages",
]

IMG_SIZE = 384
W_CNN = 0.65
W_OCR = 0.35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# LOAD MODEL
# =========================

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================
# EasyOCR
# =========================

reader = easyocr.Reader(['ar'], gpu=torch.cuda.is_available())

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def ocr_read_title(img):

    width, height = img.size
    cropped = img.crop((0, 0, width, int(height * 0.39)))

    result = reader.readtext(np.array(cropped))

    if len(result) == 0:
        return "", 0, None

    words = []
    confidences = []

    for bbox, text, conf in result:
        words.append(text)
        confidences.append(conf)

    avg_conf = np.mean(confidences) * 100
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

# =========================
# PDF LOADER (PyMuPDF)
# =========================

def load_pdf_pages(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)
    pages = []

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pages.append(img)

    return pages

# =========================
# RUN INFERENCE
# =========================

pages = load_pdf_pages(PDF_PATH, dpi=200)

for i, img in enumerate(pages):

    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        cnn_probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    text, ocr_conf, detected_type = ocr_read_title(img)
    conf_factor = ocr_conf / 100.0

    ocr_scores = np.zeros(len(classes))

    if detected_type == "auditor":
        ocr_scores[classes.index("Independent Auditor's Report")] = conf_factor

    elif detected_type == "financial":
        ocr_scores[classes.index("Financial Sheets")] = conf_factor

    elif detected_type == "notes":
        ocr_scores[classes.index("Notes (Text)")] = 0.5 * conf_factor
        ocr_scores[classes.index("Notes (Tabular)")] = 0.5 * conf_factor

        mask = np.zeros(len(classes))
        mask[classes.index("Notes (Text)")] = 1
        mask[classes.index("Notes (Tabular)")] = 1
        cnn_probs = cnn_probs * mask

    final_scores = (W_CNN * cnn_probs) + (W_OCR * ocr_scores)
    final_idx = np.argmax(final_scores)
    predicted_label = classes[final_idx]

    print(f"\nPage {i+1}")
    print("OCR:", detected_type, "| conf:", round(ocr_conf,1))
    print("CNN after mask:", dict(zip(classes, np.round(cnn_probs,3))))
    print("FINAL:", dict(zip(classes, np.round(final_scores,3))))
    print("Prediction:", predicted_label)

    plt.figure(figsize=(6,8))
    plt.imshow(img)
    plt.title(predicted_label)
    plt.axis("off")
    plt.show()
"""
Sukuk Financial Document Page Classifier — Hybrid Vision + OCR Fusion

Usage:
    python classify_pdf.py                     (interactive)
    python classify_pdf.py path/to/file.pdf    (direct)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import os
import glob
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

CLASSES = [
    "Financial Sheets",
    "Independent Auditor's Report",
    "Notes (Tabular)",
    "Notes (Text)",
    "Other Pages",
]

KEYWORDS = {
    "Financial Sheets": [
        "statement of financial position", "income statement", "cash flow",
        "statement of comprehensive income", "balance sheet", "consolidated",
        "قائمة المركز المالي", "قائمة الدخل", "قائمة التدفقات النقدية",
        "قائمة التغيرات في حقوق الملكية", "قائمة الربح",
    ],
    "Independent Auditor's Report": [
        "independent auditor", "auditor's report", "basis for opinion",
        "audit opinion",
        "تقرير مراجع الحسابات", "تقرير المراجع المستقل",
        "رأي المراجع", "أساس الرأي",
    ],
    "Notes (Tabular)": [
        "notes to the financial statements", "schedule", "table",
        "إيضاحات القوائم المالية", "الإيضاحات", "جدول",
    ],
    "Notes (Text)": [
        "accounting policies", "significant judgments",
        "estimates and assumptions",
        "السياسات المحاسبية", "تقديرات", "افتراضات",
    ],
    "Other Pages": [],
}

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "best_v2_model.pth"

IMG_SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def pad_to_square(img):
    w, h = img.size
    d = max(w, h)
    new = Image.new("RGB", (d, d), (255, 255, 255))
    new.paste(img, ((d - w) // 2, (d - h) // 2))
    return new


def load_model(device):
    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, len(CLASSES)))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def vision_probs(model, img, device):
    img = img.convert("RGB")
    img = pad_to_square(img)
    tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


def init_ocr():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return False, "eng"

    lang = "ara+eng"
    try:
        import subprocess
        langs = subprocess.check_output(
            ["tesseract", "--list-langs"], stderr=subprocess.STDOUT
        ).decode()
        if "ara" not in langs:
            print("WARNING: Arabic OCR data not found — continuing with eng only.")
            lang = "eng"
    except Exception:
        print("WARNING: Could not verify Arabic OCR — continuing with eng only.")
        lang = "eng"

    return True, lang


def ocr_text_top(img, lang):
    """OCR the top 45 % of the page where titles live."""
    try:
        import pytesseract
        rgb = img.convert("RGB")
        w, h = rgb.size
        crop = rgb.crop((0, 0, w, int(h * 0.45)))
        return pytesseract.image_to_string(crop, lang=lang)
    except Exception:
        return ""


def text_scores(text):
    text_lower = text.lower()
    scores = np.full(len(CLASSES), 0.05)
    for i, cls in enumerate(CLASSES):
        kws = KEYWORDS[cls]
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in text_lower or kw in text)
        scores[i] = min(hits / len(kws), 1.0)
    return scores


def classify_page(model, img, device, ocr_available, lang):
    v = vision_probs(model, img, device)

    if ocr_available:
        text = ocr_text_top(img, lang)
        if text.strip():
            t = text_scores(text)
            fused = 0.65 * v + 0.35 * t
            return CLASSES[int(np.argmax(fused))]

    return CLASSES[int(np.argmax(v))]


# ── Path helpers ──

def clean_path(raw):
    return raw.strip().strip('"').strip("'").replace("\\ ", " ").strip()


def find_pdf(raw_path):
    p = clean_path(raw_path)
    if os.path.isfile(p):
        return p
    as_path = Path(p)
    if as_path.is_file():
        return str(as_path)
    resolved = as_path.resolve()
    if resolved.is_file():
        return str(resolved)
    if not as_path.is_absolute():
        for base in [SCRIPT_DIR, Path.cwd()]:
            candidate = base / p
            if candidate.is_file():
                return str(candidate)
    candidates = glob.glob(str(SCRIPT_DIR / "*.pdf"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_pdf_path():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    pdfs_in_dir = sorted(glob.glob(str(SCRIPT_DIR / "*.pdf")))
    if pdfs_in_dir:
        print("PDF files found in project folder:")
        for i, fp in enumerate(pdfs_in_dir, 1):
            name = Path(fp).name
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  [{i}] {name}  ({size_mb:.1f} MB)")
        print()
        print("Enter a number to select, or paste a full path:")
    else:
        print("Enter the full path to your PDF file:")

    raw = input("> ").strip()
    if pdfs_in_dir and raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(pdfs_in_dir):
            return pdfs_in_dir[idx]
    return raw


def save_report(results, pdf_name, out_path):
    total = len(results)
    counts = Counter(r[1] for r in results)

    lines = [
        "# STC Hybrid Vision + OCR Fusion Results",
        "",
        f"**File:** `{pdf_name}`",
        f"**Total pages:** {total}",
        "",
        "## Summary",
        "",
        "| Class | Count | % |",
        "|-------|------:|---:|",
    ]
    for cls in CLASSES:
        c = counts.get(cls, 0)
        pct = 100.0 * c / total if total else 0
        lines.append(f"| {cls} | {c} | {pct:.1f}% |")

    lines += [
        "",
        "## Page-by-Page Predictions",
        "",
        "| Page | Prediction |",
        "|-----:|------------|",
    ]
    for page_num, label in results:
        lines.append(f"| {page_num} | {label} |")

    lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image is not installed.")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    raw = get_pdf_path()
    pdf_path = find_pdf(raw)

    if pdf_path is None:
        print(f"ERROR: Could not find PDF file: {raw}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    ocr_available, lang = init_ocr()

    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"ERROR: Failed to convert PDF. {e}")
        sys.exit(1)

    results = []
    for i, page_img in enumerate(pages, start=1):
        label = classify_page(model, page_img, device, ocr_available, lang)
        results.append((i, label))
        print(f"Page {i} -> {label}")

    pdf_name = Path(pdf_path).name
    report_path = SCRIPT_DIR / "STC_HYBRID_RESULTS.md"
    save_report(results, pdf_name, report_path)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()

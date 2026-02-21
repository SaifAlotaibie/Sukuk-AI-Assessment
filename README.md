# Sukuk AI Assessment

Automated classification of Arabic financial statement pages using a hybrid deep learning and OCR approach.

## Project Overview

Financial institutions publish regulatory documents as multi-page PDFs containing distinct section types. Manual sorting of these pages is time-consuming and error-prone. This project builds an automated page classifier that assigns each page to one of five categories:

| Class | Description |
|---|---|
| Independent Auditor's Report | Auditor opinion and review pages |
| Financial Sheets | Balance sheets, income statements, cash flows, equity changes |
| Notes (Tabular) | Footnotes containing tables and numerical data |
| Notes (Text) | Footnotes containing narrative text |
| Other Pages | Cover pages, table of contents, other non-financial content |

The final model combines a fine-tuned EfficientNet-B0 CNN with rule-based Arabic OCR title detection, using weighted score fusion to produce the final prediction.

## Dataset Summary

- **Source**: 30 Arabic financial statement PDFs
- **Total pages**: 1,179 labeled page images
- **Labeling**: Manual, one-by-one using an interactive labeling script
- **Splitting strategy**: Document-level split (entire PDFs assigned to one set, no page leakage)
  - Train: 740 pages (20 PDFs)
  - Validation: 219 pages (5 PDFs)
  - Test: 220 pages (5 PDFs)
- **Class distribution** (imbalanced):
  - Notes (Tabular): 547
  - Notes (Text): 330
  - Financial Sheets: 125
  - Independent Auditor's Report: 111
  - Other Pages: 66

## Project Pipeline

```
PDF Documents
    |
    v
PDF-to-Image Conversion (200 DPI)
    |
    v
Manual Page Labeling (1-5 per image)
    |
    v
Exploratory Data Analysis (dimensions, brightness, class balance)
    |
    v
Document-Level Train / Val / Test Split
    |
    v
Baseline Training (EfficientNet-B0, IMG_SIZE=224)
    |
    v
Hybrid Model Training (EfficientNet-B0, IMG_SIZE=384, OCR fusion)
    |
    v
Final Inference (CNN + Arabic OCR + Logical Masking + Weighted Fusion)
```

## Model Summary

### Baseline

- Architecture: EfficientNet-B0 (pretrained on ImageNet)
- Input: 224x224 with pad-to-square preprocessing
- Training: Two-stage (frozen backbone, then full fine-tuning)
- Scheduler: CosineAnnealingLR with early stopping

### Hybrid (Final Model)

- Architecture: EfficientNet-B0 (pretrained on ImageNet)
- Input: 384x384
- Training: Full fine-tuning for 15 epochs at LR=3e-5 (AdamW)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- OCR: Arabic title detection via pytesseract (top 39% crop, keyword matching)
- Logical masking: When OCR detects "notes", non-notes CNN probabilities are zeroed out
- Fusion: `final_scores = 0.65 * cnn_probs + 0.35 * ocr_scores`

## How to Run Inference

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure `tesseract-ocr` and the Arabic language pack are installed:

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ara

# For PDF support, also install poppler
# macOS: brew install poppler
# Ubuntu: sudo apt-get install poppler-utils
```

3. Open `run_inference.py` and set the file path at the top of the script:

```python
FILE_PATH = "/path/to/your/file.pdf"
```

4. Run:

```bash
python run_inference.py
```

The script accepts PDF files (converted to images at 200 DPI) or single image files (JPG, JPEG, PNG). For each page, it prints the predicted label and displays the image with the prediction as the title.

## Key Results

- **Best Validation Macro F1**: 0.9678 (Epoch 14 of 15)
- Test set evaluation was not executed in the saved notebook outputs.

## Technical Highlights

- Document-level splitting prevents data leakage between train/val/test sets
- Inverse-frequency class weighting handles the imbalanced label distribution
- Arabic OCR title matching provides a strong prior for structurally identifiable pages
- Logical masking enforces domain constraints (notes-detected pages cannot be classified as non-notes)
- Weighted fusion balances learned visual features with deterministic text signals

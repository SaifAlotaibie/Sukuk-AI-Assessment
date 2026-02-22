# Sukuk AI Assessment

Hybrid CNN + EasyOCR classifier for Arabic financial statement pages. Each page of a multi-page PDF is classified into one of five categories using a fine-tuned EfficientNet-B0 combined with Arabic OCR title detection and weighted score fusion.

| Class | Description |
|---|---|
| Independent Auditor's Report | Auditor opinion and review pages |
| Financial Sheets | Balance sheets, income statements, cash flows, equity changes |
| Notes (Tabular) | Footnotes containing tables and numerical data |
| Notes (Text) | Footnotes containing narrative text |
| Other Pages | Cover pages, table of contents, other non-financial content |

## Project Structure

```
Sukuk-AI-Assessment/
│
├── data/
│   ├── mini_labels.csv
│   ├── train_doc_split.csv
│   ├── val_doc_split.csv
│   ├── test_doc_split.csv
│   └── pages_raw/              # Page images (JPG, 200 DPI)
│
├── notebooks/
│   ├── EDA_visual_analysis.ipynb
│   ├── EDA_LABELED_DATA.ipynb
│   ├── DOCUMENT_LEVEL_AUDIT.ipynb
│   ├── DOCUMENT_LEVEL_SPLIT.ipynb
│   ├── BASELINE_TRAINING.ipynb
│   └── best_model_OCREFF.ipynb  # Final hybrid model training + inference
│
├── scripts/
│   ├── convert_pdf_to_image.py
│   └── label_mini.py
│
├── run_inference.py             # Standalone inference script
├── best_sukuk_model.pth         # Trained model weights
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

No system-level dependencies are required. No Tesseract. No Poppler. All OCR is handled by EasyOCR and all PDF rendering is handled by PyMuPDF — fully cross-platform (Windows, macOS, Linux).

## Model Placement

The trained model weights file `best_sukuk_model.pth` must be placed in the **project root** (same directory as `run_inference.py`). This is the default location and requires no configuration.

## How to Run

1. Open `run_inference.py` and set the PDF path at the top of the script:

```python
PDF_PATH = "path/to/your/file.pdf"
```

2. Run:

```bash
python run_inference.py
```

For each page, the script prints the predicted label with debug information (OCR detection, CNN probabilities, fusion scores) and displays the page image with the prediction as the title.

## Model Summary

- **Architecture**: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Input**: 384 × 384
- **OCR**: EasyOCR Arabic title detection (top 39% crop, keyword matching)
- **Fusion**: `final_scores = 0.65 × cnn_probs + 0.35 × ocr_scores`
- **Masking**: When OCR detects "notes", non-notes CNN probabilities are zeroed out
- **Best Validation Macro F1**: 0.9678

## Cross-Platform Note

This project requires **no system-level OCR or PDF tools**. Everything runs through Python packages:

- **EasyOCR** for Arabic text recognition
- **PyMuPDF** for PDF-to-image conversion

Works on Windows, macOS, and Linux without any additional installation steps beyond `pip install -r requirements.txt`.

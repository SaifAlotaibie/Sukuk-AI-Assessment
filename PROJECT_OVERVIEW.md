# Project Overview

Technical description of the repository structure, components, and design decisions.

## Repository Structure

```
Sukuk-AI-Assessment/
|
|-- data/
|   |-- mini_labels.csv          # All 1,179 manual page labels
|   |-- train_doc_split.csv      # Training set (740 pages, 20 PDFs)
|   |-- val_doc_split.csv        # Validation set (219 pages, 5 PDFs)
|   |-- test_doc_split.csv       # Test set (220 pages, 5 PDFs)
|
|-- models/
|   |-- best_sukuk_model.pth     # Saved weights (EfficientNet-B0, 5 classes)
|
|-- notebooks/
|   |-- EDA_visual_analysis.ipynb     # Image dimension and pixel statistics
|   |-- EDA_LABELED_DATA.ipynb        # Label distribution and per-class analysis
|   |-- DOCUMENT_LEVEL_AUDIT.ipynb    # Dataset integrity checks before splitting
|   |-- DOCUMENT_LEVEL_SPLIT.ipynb    # Document-level train/val/test split logic
|   |-- BASELINE_TRAINING.ipynb       # Baseline EfficientNet-B0 training (IMG_SIZE=224)
|   |-- best_model_OCREFF.ipynb       # Final hybrid model training + OCR inference
|
|-- scripts/
|   |-- convert_pdf_to_image.py  # Batch PDF-to-JPG conversion (PyMuPDF, 200 DPI)
|   |-- label_mini.py            # Interactive page labeling tool
|
|-- run_inference.py             # Standalone hybrid inference script
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project summary and usage instructions
|-- PROJECT_OVERVIEW.md          # This file
```

## Component Details

### Data Preparation

**convert_pdf_to_image.py** -- Converts all PDFs in `Data Set/` to individual JPG page images at 200 DPI using PyMuPDF (`fitz`). Output naming convention: `{pdf_name}_page_{N}.jpg`.

**label_mini.py** -- Interactive labeling script. Displays each page image, prompts the user for a label (1-5), and appends results to `mini_labels.csv`. Supports resuming from where the previous session ended.

### Exploratory Data Analysis

**EDA_visual_analysis.ipynb** -- Analyzes raw page images: resolution distribution (39 unique sizes found), orientation (4.6% landscape), pixel brightness statistics (mean 246.71, std 7.73), and one outlier image at 5334x3750.

**EDA_LABELED_DATA.ipynb** -- Examines the labeled dataset: class balance, per-class image dimension distributions, random sample visualizations per class, and pixel intensity histograms. Uses `tqdm` for iteration progress.

**DOCUMENT_LEVEL_AUDIT.ipynb** -- Validates dataset integrity before splitting. Confirms all 1,179 images are labeled and checks for missing or duplicate entries.

### Data Splitting

**DOCUMENT_LEVEL_SPLIT.ipynb** -- Implements a document-level split to prevent data leakage. Entire PDFs (not individual pages) are assigned to train, validation, or test. Uses a greedy heuristic across 200 random seeds to find a split that maintains class balance. Final split: 20 train / 5 val / 5 test PDFs.

### Baseline Model

**BASELINE_TRAINING.ipynb** -- EfficientNet-B0 transfer learning baseline.

- Input: 224x224 with pad-to-square before resize
- Augmentation: RandomRotation(2), ColorJitter(brightness=0.1, contrast=0.1)
- Training: Two-stage -- 5 epochs frozen backbone (LR=1e-3), then up to 30 epochs full fine-tuning (backbone LR=1e-4, head LR=1e-3)
- Scheduler: CosineAnnealingLR, early stopping (patience=7) on validation loss
- Classifier head: Dropout(0.3) + Linear
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Hardware: Local (MPS/CPU)
- No logged performance metrics in saved outputs

### Final Hybrid Model

**best_model_OCREFF.ipynb** -- Enhanced model with Arabic OCR fusion.

- Input: 384x384 (no pad-to-square)
- Augmentation: RandomAdjustSharpness(2, p=0.5), ColorJitter(brightness=0.05, contrast=0.05)
- Training: Full fine-tuning for 15 epochs at constant LR=3e-5 (AdamW)
- Dropout: Default EfficientNet dropout (0.2)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Hardware: Google Colab (A100 GPU)
- Best Validation Macro F1: 0.9678 (Epoch 14)

### OCR Logic

Implemented in `ocr_read_title()` (defined in both the notebook and `run_inference.py`):

1. Crop the top 39% of the page image
2. Run `pytesseract.image_to_data` with `lang="ara"`
3. Filter words with confidence > 0, compute average confidence
4. Clean text (remove tatweel, normalize whitespace)
5. Take the first 150 characters as the title region
6. Match against six Arabic keyword phrases:
   - "تقرير المراجع المستقل" --> `auditor`
   - "قائمة المركز المالي" --> `financial`
   - "قائمة الدخل" --> `financial`
   - "قائمة التدفقات النقدية" --> `financial`
   - "قائمة التغيرات في حقوق الملكية" --> `financial`
   - "إيضاحات حول القوائم المالية" --> `notes`

### Fusion and Masking

The inference pipeline combines CNN and OCR signals:

1. **CNN inference**: Softmax probabilities from the fine-tuned EfficientNet-B0
2. **OCR scoring**: Based on detected title type and average OCR confidence
   - `auditor`: full confidence assigned to "Independent Auditor's Report"
   - `financial`: full confidence assigned to "Financial Sheets"
   - `notes`: confidence split equally between "Notes (Text)" and "Notes (Tabular)"
3. **Logical masking**: When OCR detects `notes`, all CNN probabilities except the two Notes classes are zeroed out
4. **Weighted fusion**: `final_scores = 0.65 * cnn_probs + 0.35 * ocr_scores`
5. **Prediction**: `argmax(final_scores)`

### Inference Script

**run_inference.py** -- Standalone script. The user sets `FILE_PATH` at the top of the file and runs `python run_inference.py`. Supports PDF (converted at 200 DPI via `pdf2image`) and single images (JPG, JPEG, PNG). Loads model weights from `models/best_sukuk_model.pth`. Applies the full hybrid pipeline (CNN + OCR + masking + fusion) and displays each page with the predicted label.

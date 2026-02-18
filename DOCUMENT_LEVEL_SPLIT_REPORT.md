# Document-Level Split Report

---

## 1. Split Overview

- **Total images:** 1179
- **Total PDFs:** 30
- **Split ratio (PDFs):** 20 / 5 / 5
- **Algorithm:** Greedy heuristic — assign largest PDFs first, minimizing class-distribution deviation
- **Best seed:** 2

## 2. PDFs per Split

### Train — 20 PDFs, 740 images (62.8%)

- FS1 (33 pages)
- FS10 (37 pages)
- FS11 (26 pages)
- FS12 (64 pages)
- FS15 (34 pages)
- FS17 (33 pages)
- FS18 (36 pages)
- FS19 (55 pages)
- FS22 (60 pages)
- FS3 (33 pages)
- FS4 (28 pages)
- FS7 (30 pages)
- FS9 (33 pages)
- RSF1 (31 pages)
- RSF2 (33 pages)
- RSF3 (34 pages)
- RSF4 (35 pages)
- RSF5 (35 pages)
- RSF6 (32 pages)
- RSF7 (38 pages)

### Val — 5 PDFs, 219 images (18.6%)

- FS13 (39 pages)
- FS16 (57 pages)
- FS20 (40 pages)
- FS21 (44 pages)
- FS5 (39 pages)

### Test — 5 PDFs, 220 images (18.7%)

- FS14 (40 pages)
- FS2 (41 pages)
- FS6 (52 pages)
- FS8 (45 pages)
- RSF8 (42 pages)

## 3. Image Counts

| Split | PDFs | Images | % of Total |
|-------|-----:|-------:|-----------:|
| Train | 20 | 740 | 62.8% |
| Val | 5 | 219 | 18.6% |
| Test | 5 | 220 | 18.7% |
| **Total** | **30** | **1179** | **100.0%** |

## 4. Class Distribution per Split

| Split | Financial Sheets | Independent Auditor's Report | Notes (Tabular) | Notes (Text) | Other Pages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11.2% | 9.5% | 45.7% | 27.7% | 5.9% |
| Val | 9.6% | 9.1% | 48.9% | 27.9% | 4.6% |
| Test | 9.5% | 9.5% | 46.4% | 29.1% | 5.5% |
| **Global** | 10.6% | 9.4% | 46.4% | 28.0% | 5.6% |

## 5. Deviation from Global Distribution

| Split | Financial Sheets | Independent Auditor's Report | Notes (Tabular) | Notes (Text) | Other Pages | Max Dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | +0.6% | +0.0% | -0.7% | -0.3% | +0.3% | 0.72% |
| Val | -1.0% | -0.3% | +2.5% | -0.1% | -1.0% | 2.46% |
| Test | -1.1% | +0.1% | -0.0% | +1.1% | -0.1% | 1.10% |

## 6. Recommendation

**Verdict: EXCELLENT**

All splits have class distributions within 3 percentage points of the global distribution. This split is well-balanced and ready for use.

### Key observations

- Worst-case maximum deviation: **2.46%** (val split)
- All 5 classes are represented in every split.
- Skewed PDF (FS19) placed in **train** split.
- Outlier PDFs in train: 12 — ['FS11', 'FS12', 'FS17', 'FS19', 'FS22', 'FS3', 'FS4', 'RSF1', 'RSF2', 'RSF3', 'RSF4', 'RSF7']
- Outlier PDFs in val: 0 — []
- Outlier PDFs in test: 1 — ['RSF8']

### Files generated

- `train_doc_split.csv`
- `val_doc_split.csv`
- `test_doc_split.csv`

---

*Report generated automatically by DOCUMENT_LEVEL_SPLIT.ipynb — no model training performed.*
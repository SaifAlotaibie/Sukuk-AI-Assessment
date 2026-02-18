# Final EDA Report — Full Labeled Dataset

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total labeled images | 1 179 |
| Unique classes | 5 |
| Unique PDFs | 30 |
| Source CSV | `mini_labels.csv` |
| Image directory | `pages_raw/` |

---

## 2. Class Distribution

| Label | Count | Percentage (%) |
|---|---|---|
| Notes (Tabular) | 547 | 46.40 |
| Notes (Text) | 330 | 27.99 |
| Financial Sheets | 125 | 10.60 |
| Independent Auditor's Report | 111 | 9.41 |
| Other Pages | 66 | 5.60 |

**Imbalance ratio (largest / smallest):** 8.29

---

## 3. PDF Distribution

**Total unique PDFs:** 30

### Top 10 PDFs by Page Count

| PDF Prefix | Pages |
|---|---|
| FS12 | 64 |
| FS22 | 60 |
| FS16 | 57 |
| FS19 | 55 |
| FS6 | 52 |
| FS8 | 45 |
| FS21 | 44 |
| RSF8 | 42 |
| FS2 | 41 |
| FS14 | 40 |

---

## 4. Class vs PDF Cross Table

All 30 PDFs contain all 5 classes. No PDFs are missing any class.

| PDF | Financial Sheets | Ind. Auditor's Report | Notes (Tabular) | Notes (Text) | Other Pages | Total |
|---|---|---|---|---|---|---|
| FS12 | 4 | 4 | 37 | 17 | 2 | 64 |
| FS22 | 4 | 5 | 31 | 19 | 1 | 60 |
| FS16 | 5 | 4 | 28 | 18 | 2 | 57 |
| FS19 | 4 | 2 | 35 | 12 | 2 | 55 |
| FS6 | 5 | 5 | 25 | 15 | 2 | 52 |
| FS8 | 4 | 4 | 20 | 16 | 1 | 45 |
| FS21 | 4 | 5 | 18 | 15 | 2 | 44 |
| RSF8 | 4 | 4 | 17 | 12 | 5 | 42 |
| FS2 | 4 | 4 | 20 | 11 | 2 | 41 |
| FS14 | 4 | 4 | 20 | 10 | 2 | 40 |
| FS20 | 4 | 4 | 22 | 8 | 2 | 40 |
| FS13 | 4 | 3 | 18 | 12 | 2 | 39 |
| FS5 | 4 | 4 | 21 | 8 | 2 | 39 |
| RSF7 | 4 | 4 | 13 | 15 | 2 | 38 |
| FS10 | 5 | 3 | 17 | 10 | 2 | 37 |
| FS18 | 4 | 4 | 16 | 10 | 2 | 36 |
| RSF5 | 4 | 4 | 13 | 11 | 3 | 35 |
| RSF4 | 4 | 4 | 12 | 11 | 4 | 35 |
| RSF3 | 5 | 5 | 12 | 8 | 4 | 34 |
| FS15 | 4 | 3 | 16 | 9 | 2 | 34 |
| FS1 | 4 | 3 | 14 | 10 | 2 | 33 |
| FS17 | 4 | 3 | 18 | 6 | 2 | 33 |
| FS3 | 5 | 3 | 17 | 6 | 2 | 33 |
| RSF2 | 4 | 4 | 11 | 12 | 2 | 33 |
| FS9 | 4 | 3 | 15 | 9 | 2 | 33 |
| RSF6 | 4 | 4 | 13 | 9 | 2 | 32 |
| RSF1 | 4 | 4 | 12 | 9 | 2 | 31 |
| FS7 | 4 | 2 | 15 | 7 | 2 | 30 |
| FS4 | 4 | 3 | 9 | 10 | 2 | 28 |
| FS11 | 4 | 3 | 12 | 5 | 2 | 26 |

---

## 5. Image Statistics Summary

| Statistic | Width (px) | Height (px) | Aspect Ratio | Mean Brightness | Std Brightness |
|---|---|---|---|---|---|
| Count | 1 179 | 1 179 | 1 179 | 1 179 | 1 179 |
| Mean | 1 702.27 | 2 255.75 | 0.76 | 246.71 | 36.42 |
| Std | 163.43 | 147.88 | 0.13 | 7.73 | 8.82 |
| Min | 1 649 | 1 651 | 0.70 | 85.09 | 8.46 |
| 25% | 1 654 | 2 200 | 0.71 | 244.99 | 31.68 |
| 50% | 1 654 | 2 338 | 0.71 | 247.25 | 36.80 |
| 75% | 1 700 | 2 339 | 0.77 | 249.17 | 41.99 |
| Max | 5 334 | 3 750 | 1.42 | 254.65 | 72.77 |

---

## 6. Per-Class Image Statistics (Mean)

| Label | Mean Brightness | Std Brightness | Width (px) | Height (px) |
|---|---|---|---|---|
| Financial Sheets | 248.40 | 31.41 | 1 727.00 | 2 233.43 |
| Independent Auditor's Report | 243.66 | 41.41 | 1 661.29 | 2 316.05 |
| Notes (Tabular) | 247.91 | 35.61 | 1 717.93 | 2 229.92 |
| Notes (Text) | 245.22 | 41.49 | 1 673.57 | 2 278.93 |
| Other Pages | 246.26 | 18.85 | 1 738.08 | 2 294.73 |

---

## 7. Outlier Detection (IQR Method)

| Outlier Type | Count |
|---|---|
| Extremely small images | 0 |
| Extremely large images | 1 |
| Extreme dark images | 18 |
| Extreme bright images | 0 |
| **Total outliers** | **19** |

---

## 8. Final Summary

- **Total dataset size:** 1 179 labeled images across 30 PDFs and 5 classes.
- **Class balance:** Moderate imbalance (ratio 8.29). "Notes (Tabular)" dominates at 46.4%, while "Other Pages" is the smallest at 5.6%. Standard stratified splitting should suffice; class weights or oversampling may further help.
- **Severe imbalance:** Not severe (ratio < 10), but the gap between the largest and smallest class is notable.
- **Image size consistency:** Some variation (std width = 163.4 px, std height = 147.9 px). One image is notably large (5 334 x 3 750). Resizing to a uniform dimension is recommended.
- **Anomalies detected:** 19 outlier images — 1 extremely large, 18 with unusually dark brightness. These should be reviewed before training.
- **ML readiness:** The dataset is ready for modeling after standard preprocessing (resize, normalize). Stratified splitting and class weighting are recommended.

---

*Report generated from `LAST_EDA_LABELED_DATA.ipynb`*

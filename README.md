# Speaker-Normalized Vowel Formant Classification

This repository contains code accompanying an ACL Rolling Review submission investigating whether gender-related acoustic structure remains detectable after within-speaker normalization of vowel formants.

## Overview

The project evaluates whether speaker-normalized F1 and F2 (z-scored within speaker) and raw F3 retain multivariate structure predictive of gender under strict Leave-One-Speaker-Out (LOSO) validation.

The pipeline includes:

- Within-speaker z-score normalization (F1, F2)
- Linear mixed-effects modeling
- Random Forest classification with LOSO cross-validation
- Permutation testing for statistical significance
- Within-vowel LOSO evaluation

All modeling decisions correspond to the experimental setup described in the associated manuscript.

---

## Data Format

The dataset is not included in this repository.

Expected CSV format:

Columns:

- `Speaker` — speaker identifier  
- `Gender` — categorical label (e.g., Female, Male)  
- `Vowel` — vowel category  
- `F1` — first formant frequency (Hz)  
- `F2` — second formant frequency (Hz)  
- `F3` — third formant frequency (Hz)  

Place the dataset as:


speaker_data.csv


in the project root directory before running the script.

---

## Requirements

Python 3.9+

Install dependencies:


pip install -r requirements.txt


Required packages:

- numpy
- pandas
- scikit-learn
- statsmodels

---

## Running the Code

After placing `speaker_data.csv` in the root directory:


python f.py


The script will:

- Normalize features by speaker
- Fit mixed-effects models
- Perform LOSO classification
- Compute accuracy and AUC
- Conduct permutation testing
- Report per-vowel LOSO accuracy

---

## Reproducibility

All random seeds are fixed to ensure deterministic results.  
Hyperparameters are fixed prior to evaluation to prevent fold-specific tuning.

---

## License

Provided for research and reproducibility purposes.

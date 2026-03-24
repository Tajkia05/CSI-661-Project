# CSI-661-Project
Investigating demographic privacy leakage in recommender systems  by inferring gender from learned user embeddings using MovieLens 1M.

# Privacy Leakage in Recommender Systems

**Course:** ICSI/CSI 661 — Human-Centered Machine Learning  
**University:** University at Albany, SUNY  
**Authors:** Tajkia Nuri Ananna, Haroun M Hamza  

## Overview

This project investigates whether recommender system models 
unintentionally leak private demographic information through 
their learned user embeddings. We train a Matrix Factorization 
(MF) model on the MovieLens 1M dataset and evaluate whether a 
separate attacker model can infer gender from the resulting 
user embeddings — without gender ever being used during 
recommender training.

## Research Question

> To what extent do recommender system models unintentionally 
> leak private demographic information (e.g., gender) through 
> learned user embeddings?

## Dataset

- **MovieLens 1M** — 1,000,209 ratings from 6,040 users on 
  3,706 movies
- Download: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Gender and age labels used only for evaluation, never for training

## Project Structure
```
├── baseline.py             # Behavioral baseline (Person 1 — EDA + baseline)
├── mf_attack.py            # Alternative MF implementation
├── README.md
```

## How to Reproduce Results

### Hamza: EDA & Behavioral Baseline

1. Download MovieLens 1M and place files in `data/` folder
2. Run `python baseline.py`
3. The script will output EDA plots and baseline results

### Tajkia: MF + Attacker Model

1. Open `mf_attack.py` in Google Colab
2. Run all cells in order from top to bottom
3. The notebook will:
   - Download and load MovieLens 1M automatically
   - Train the MF model on all ratings
   - Extract user embeddings
   - Train Logistic Regression attacker
   - Output Accuracy, F1, and AUC



## Results (Midterm)

| Method | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Majority Baseline | TBD | --- | --- |
| Behavioral Features + LR | TBD | TBD | TBD |
| MF Embeddings + LR | 0.7318 | 0.8354 | 0.7541 |

## Key Finding

MF embeddings achieve an AUC of **0.7541**, substantially above 
the random baseline of 0.50. This indicates that user embeddings 
learned purely from rating interactions encode meaningful gender 
signal, even when gender was never used during training.

## Dependencies
```
torch
pandas
numpy
scikit-learn
matplotlib
```

Install with:
```
pip install torch pandas numpy scikit-learn matplotlib
```

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens 
Datasets: History and Context. ACM Transactions on Interactive 
Intelligent Systems (TiiIS) 5, 4, Article 19.

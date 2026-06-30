# 🚦 Traffic Demand Prediction | Flipkart Gridlock Hackathon Phase 1

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Overview

This repository contains our solution for **Flipkart Gridlock Hackathon – Phase 1**, where the objective was to accurately predict traffic demand using historical traffic patterns, temporal information, road infrastructure, weather conditions, and geographical data.

The solution combines extensive feature engineering with gradient boosting models to learn complex traffic demand patterns and generate accurate predictions.

---

# Problem Statement

Urban traffic demand changes dynamically based on:

- Time of day
- Road infrastructure
- Weather conditions
- Vehicle movement
- Geographic location

The goal was to build a regression model capable of accurately forecasting future traffic demand using the provided dataset.

---

# Dataset

The dataset contains:

### Train Dataset

- Historical traffic demand
- Road information
- Weather
- Geographical location
- Timestamp

### Test Dataset

- Same features except target demand

Target Variable

```
demand
```

---

# Project Workflow

```
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Encoding
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Prediction
        │
        ▼
Submission Generation
```

---

# Feature Engineering

Several engineered features were created to improve model performance.

## Temporal Features

- Hour
- Minute
- Time Slot
- Cyclical Hour Encoding (sin/cos)

Example:

```
timestamp → hour → minute → time_slot
```

---

## Geospatial Features

The provided geohash values were decoded into:

- Latitude
- Longitude

Additional hierarchy features:

- Geo Prefix (4)
- Geo Prefix (5)

These features helped the model capture regional traffic patterns.

---

## Statistical Features

Location-based aggregations:

- Mean Demand
- Median Demand
- Standard Deviation

Count-based encoding:

- Geohash Count
- Location Frequency
- Interaction Counts

---

## Target Encoding

Out-of-Fold Target Encoding was applied on high-cardinality categorical variables to reduce target leakage while preserving useful information.

---

## Data Preprocessing

- Missing value handling
- Label Encoding
- Numerical median imputation
- Log transformation of target variable

```
y = log1p(demand)
```

Predictions were converted back using

```
expm1(prediction)
```

---

# Machine Learning Models

## LightGBM

Gradient Boosted Decision Trees

Used because of:

- Fast training
- Excellent performance on tabular datasets
- Strong handling of mixed numerical and categorical features

---

## CatBoost

Used for:

- High-cardinality categorical variables
- Better generalization
- Reduced preprocessing requirements

---

## Ensemble Learning

Final predictions were generated using a weighted ensemble of:

- CatBoost
- LightGBM

This improved robustness and reduced variance.

---

# Validation Strategy

- K-Fold Cross Validation
- Out-of-Fold Predictions
- Early Stopping

Evaluation Metric

```
R² Score
```

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- CatBoost
- PyGeoHash
- Google Colab
- Kaggle

---

# Repository Structure

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── Traffic_Prediction.ipynb
│
├── outputs/
│   ├── submission.csv
│
├── README.md
│
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/flipkart-gridlock-traffic-demand-prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
jupyter notebook
```

or

```bash
python main.py
```

---

# Results

The project explored multiple feature engineering strategies including:

- Temporal Encoding
- Geospatial Features
- Statistical Aggregations
- Target Encoding
- Ensemble Learning

These approaches significantly improved model performance over the baseline model.

---

# Future Improvements

- Graph Neural Networks
- Spatio-Temporal Transformers
- Deep Sequence Models
- Bayesian Hyperparameter Optimization
- Automated Feature Selection

---

# Team

Developed for

**Flipkart Gridlock Hackathon – Phase 1**

---

# License

This project is intended for educational and hackathon purposes.

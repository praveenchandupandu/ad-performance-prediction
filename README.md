# Ad Click Prediction

Portfolio Project | PantechELearning DataScience Course | 2022

## Overview

Predictive model for click-through rates on online advertisements using real user data and machine learning.

## Dataset

10,000 ad impressions with user and context features:
- User features: age, gender
- Ad features: device type, ad position
- Context: browsing history, time of day
- Target: Click (0/1)

## Analysis

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Standardized numerical features

### 2. Click Prediction Models

**Logistic Regression:**
- AUC-ROC: 0.535
- Accuracy: 64.7%

**Random Forest:**
- AUC-ROC: 0.740
- Accuracy: 69.5%

**Winner: Random Forest**

## Technologies

Python, scikit-learn, pandas, Logistic Regression, Random Forest

## How to Use

1. Place ad_click_dataset.csv in data/ folder
2. Run: python preprocessing.py
3. Run: python prediction.py

## Results

Successfully predicted ad clicks with 69.5% accuracy and 0.740 AUC-ROC on real user data.

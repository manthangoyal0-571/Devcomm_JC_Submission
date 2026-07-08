# Task 1: Spaceship Titanic Transportation Prediction

## Overview
This notebook contains a machine learning solution for predicting passenger transportation in the Spaceship Titanic Kaggle competition using LightGBM gradient boosting.

---

## 1. Setup and Data Loading

### Libraries and Dependencies
```python
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import kagglehub
```

### Loading Datasets
```python
# LOADING DATASETS
print("Loading records...")
train_df = pd.read_csv('/kaggle/input/competitions/jc-dev-comm-recruitment-task/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/jc-dev-comm-recruitment-task/test.csv')

# Saving PassengerIds for the final submission file
test_ids = test_df['PassengerId']
target_col = 'Transported'

# Joining both the datasets for pre-processing
df = pd.concat([train_df, test_df], join="inner", axis=0, ignore_index=True)
```

**Output Files Available:**
- `/kaggle/input/competitions/jc-dev-comm-recruitment-task/sample_submission.csv`
- `/kaggle/input/competitions/jc-dev-comm-recruitment-task/train.csv`
- `/kaggle/input/competitions/jc-dev-comm-recruitment-task/test.csv`

---

## 2. Exploratory Data Analysis

### Target Distribution and Missing Values
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 1. Target Balance
train_df['Transported'].value_counts().plot(kind='bar', ax=ax1, color=['#4f46e5', '#f43f5e'])
ax1.set_title('Target Distribution (Transported)')
ax1.set_xticklabels(['True', 'False'], rotation=0)

# 2. Missing Values Per Column
train_df.isnull().sum().plot(kind='barh', ax=ax2, color='#64748b')
ax2.set_title('Missing Values Count Per Column')

plt.tight_layout()
plt.show()

# CryoSleep vs Transportation Rate
train_df.groupby('CryoSleep')['Transported'].mean().plot(kind='bar', color='#10b981', figsize=(5, 4))
plt.title('Dimension Shift Rate by CryoSleep')
plt.ylabel('Probability of Being Transported')
plt.xticks(rotation=0)
plt.show()
```

**Key Observations:**
- Analysis of target variable balance across Transported status
- Identification of missing values per column
- Relationship between CryoSleep status and transportation probability

---

## 3. Feature Engineering

### Comprehensive Feature Engineering Pipeline
```python
print("Deconstructing the fields...")

# A. Group Size Extraction
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
df['Group_Size'] = df['Group'].map(df['Group'].value_counts())
df['Is_Alone'] = (df['Group_Size'] == 1).astype(int)

# B. Cabin Breakdown (Deck / Num / Side)
df['Cabin'].fillna('None/None/None', inplace=True)
df['Cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
df['Cabin_Num'] = df['Cabin'].apply(lambda x: x.split('/')[1])
df['Cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')

# C. CryoSleep-Aware Luxury Spending Imputation
luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# If a passenger is in CryoSleep, their luxury spend must logically be $0
for amenity in luxury_amenities:
    df.loc[(df['CryoSleep'] == True) & (df[amenity].isna()), amenity] = 0.0
    # Imputing remaining missing values with the median of their respective home planets
    df[amenity] = df.groupby('HomePlanet')[amenity].transform(lambda x: x.fillna(x.median()))

# Calculating cumulative expenditures
df['Total_Spend'] = df[luxury_amenities].sum(axis=1)
df['No_Spending'] = (df['Total_Spend'] == 0).astype(int)

# D. Basic Demographic Imputations
df['Age'] = df.groupby(['HomePlanet', 'No_Spending'])['Age'].transform(lambda x: x.fillna(x.median()))
df['CryoSleep'] = df.groupby('No_Spending')['CryoSleep'].transform(lambda x: x.fillna(x.mode()[0])).astype(int)
df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
df['VIP'] = df['VIP'].fillna(False).astype(int)

# E. Categorical Feature Encoding
categorical_cols = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# F. Dropping uninformative columns
drop_cols = ['PassengerId', 'Cabin', 'Name', 'Group']
if target_col in df.columns:
    drop_cols.append(target_col)
X_processed = df.drop(columns=drop_cols)

# Re-splitting datasets into Train and Test datasets
X_train = X_processed.iloc[:len(train_df)]
y_train = train_df[target_col].astype(int)
X_test = X_processed.iloc[len(train_df):]

print(f"Processed Dataset Shape: {X_train.shape}")
```

**Feature Engineering Steps:**
1. **Group Size Extraction** - Extract passenger group information from PassengerId
2. **Cabin Breakdown** - Decompose cabin into Deck, Number, and Side
3. **Smart Imputation** - Logic-based imputation considering CryoSleep status
4. **Spending Features** - Aggregate luxury amenity spending
5. **Demographic Handling** - Impute Age, CryoSleep, and other demographics
6. **Encoding** - Label encode categorical features

**Result:** Processed dataset shape: (8693, 17)

---

## 4. Model Training with Stratified K-Fold Cross-Validation

### LightGBM Configuration and Training
```python
print("Initializing Stratified LightGBM Gradient Boosting Model...")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))

# Optimal hyperparameters
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.04,
    'num_leaves': 31,
    'max_depth': 7,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # Creating dataset objects
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    # Training classifier
    clf = lgb.train(
        lgb_params, 
        trn_data, 
        num_boost_round=1000, 
        valid_sets=[trn_data, val_data], 
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    oof_predictions[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
    test_predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

# Printing Out-Of-Fold Accuracy Score
oof_accuracy = accuracy_score(y_train, (oof_predictions > 0.5).astype(int))
print(f"Cross-Validation Dimensional Accuracy: {oof_accuracy:.4f}")
```

**Results:**
- **Cross-Validation Accuracy:** 81.20%
- **Validation Strategy:** 5-Fold Stratified Cross-Validation
- **Early Stopping:** Applied with patience of 50 rounds

---

## 5. Model Analysis and Visualization

### Prediction Distribution
```python
plt.figure(figsize=(7, 4))
# Plotting simple histograms for predicted probabilities
plt.hist(oof_predictions[y_train == 1], bins=20, alpha=0.5, label='Actual True', color='#10b981')
plt.hist(oof_predictions[y_train == 0], bins=20, alpha=0.5, label='Actual False', color='#ef4444')
plt.axvline(0.5, color='black', linestyle='--')
plt.title('Validation Prediction Probabilities')
plt.legend()
plt.show()

# Extracting and sorting features
feat_imp = pd.Series(clf.feature_importance(importance_type='gain'), index=X_train.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(8, 5), color='#6366f1')

plt.title('LightGBM Feature Importance (Gain)')
plt.tight_layout()
plt.show()
```

**Visualizations:**
- Prediction probability distributions for positive and negative classes
- Feature importance ranking (Gain metric)

---

## 6. Submission Generation

### Export Predictions to CSV
```python
print("Writing the predictions to CSV...")

submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': (test_predictions > 0.5).astype(bool)})

submission.to_csv('submission.csv', index=False)
print("Analysis completed successfully. submission.csv is ready!")
```

**Output:** 
- `submission.csv` - Final predictions in Kaggle submission format

---

## Summary

### Approach:
1. **Data Preprocessing** - Intelligent handling of missing values with domain logic
2. **Feature Engineering** - Extraction of meaningful features from raw data
3. **Model Selection** - LightGBM gradient boosting for binary classification
4. **Cross-Validation** - 5-fold stratified cross-validation for robust evaluation
5. **Prediction** - Ensemble predictions averaged across folds

### Key Performance:
- **Validation Accuracy:** 81.20%
- **Model:** LightGBM with optimal hyperparameters
- **Strategy:** Out-of-fold predictions with proper train-test splitting

### Technical Stack:
- **Python Libraries:** NumPy, Pandas, Scikit-Learn, LightGBM, Matplotlib
- **Environment:** Kaggle Notebook (Python 3.12.13)

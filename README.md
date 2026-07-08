# Devcomm_JC_Submission

Kaggle Competition: JC DevComm Recruitment Task

## 📋 Project Overview

This repository contains a machine learning solution for the **Spaceship Titanic Transportation Prediction** competition hosted on Kaggle as part of the JC DevComm Recruitment Task. The project demonstrates a complete end-to-end machine learning pipeline from data exploration to model deployment.

## 🎯 Objective

Predict whether passengers aboard the Spaceship Titanic were transported to an alternate dimension using binary classification techniques. This is a supervised learning task using the LightGBM gradient boosting algorithm.

## 📁 Repository Structure

```
Devcomm_JC_Submission/
├── README.md                    # This file
└── task_1/
    ├── task_1.ipynb            # Main Jupyter Notebook with complete solution
    └── task_1.md               # Detailed documentation of approach and results
```

## 🔧 Technologies & Libraries

- **Python 3.12.13**
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-Learn, LightGBM
- **Visualization:** Matplotlib
- **Environment:** Kaggle Notebooks

## 📊 Project Pipeline

### 1. **Data Exploration & Analysis**
   - Target variable distribution analysis
   - Missing values identification
   - Relationship analysis between features and target variable
   - CryoSleep impact on transportation probability

### 2. **Feature Engineering**
   - **Group Size Extraction:** Extract passenger group information from PassengerId
   - **Cabin Decomposition:** Break down cabin into Deck, Number, and Side
   - **Smart Imputation:** Logic-based missing value handling (e.g., CryoSleep passengers have $0 luxury spending)
   - **Spending Features:** Aggregate luxury amenity expenditures (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)
   - **Demographic Processing:** Handle Age, CryoSleep, HomePlanet, and Destination
   - **Categorical Encoding:** Label encode categorical features

**Result:** 17 engineered features from raw dataset

### 3. **Model Training**
   - **Algorithm:** LightGBM Gradient Boosting
   - **Validation Strategy:** 5-Fold Stratified Cross-Validation
   - **Early Stopping:** Enabled with patience of 50 rounds
   - **Hyperparameters:**
     - Learning Rate: 0.04
     - Number of Leaves: 31
     - Max Depth: 7
     - Feature Fraction: 0.8
     - Bagging Fraction: 0.8

### 4. **Model Evaluation**
   - **Cross-Validation Accuracy:** 81.20%
   - **Prediction Probability Analysis**
   - **Feature Importance Ranking**

### 5. **Submission Generation**
   - Final predictions exported to `submission.csv`
   - Kaggle submission format compliance

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Cross-Validation Accuracy | 81.20% |
| Validation Strategy | 5-Fold Stratified |
| Model Type | LightGBM Binary Classifier |
| Training Samples | 8,693 |
| Features | 17 |

## 🚀 How to Run

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook task_1/task_1.ipynb
   ```

2. **Kaggle Notebook:**
   - Upload the notebook to Kaggle
   - Ensure access to the JC DevComm Recruitment Task dataset
   - Run all cells sequentially

3. **Output:**
   - `submission.csv` - Predictions ready for Kaggle submission

## 📝 Dataset Information

**Source:** JC DevComm Recruitment Task on Kaggle

**Files:**
- `train.csv` - Training data with target variable
- `test.csv` - Test data for predictions
- `sample_submission.csv` - Submission format reference

**Key Features:**
- PassengerId, CryoSleep, HomePlanet, Destination
- Cabin (Deck/Number/Side)
- Age, VIP status
- Luxury spending: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- Target: Transported (Boolean)

## 💡 Approach Highlights

1. **Domain-Aware Imputation:** Leverages logical constraints (e.g., CryoSleep passengers cannot spend on amenities)
2. **Smart Feature Engineering:** Creates meaningful features from raw identifiers and categorical data
3. **Robust Validation:** Uses stratified k-fold to handle class imbalance
4. **Ensemble Strategy:** Averages predictions across folds for better generalization
5. **Early Stopping:** Prevents overfitting during gradient boosting training

## 📄 Documentation

For detailed step-by-step explanation of the solution, feature engineering decisions, and visualizations, refer to `task_1/task_1.md`.

## 👤 Author

**manthangoyal0-571** - JC DevComm Recruitment Task Submission

## 📅 Date

Created: July 2026

## 📢 Kaggle Competition

This project participates in the Kaggle competition: **Spaceship Titanic** as part of the **JC DevComm Recruitment Task**

---

**Note:** This repository contains Jupyter Notebooks (100% Jupyter Notebook composition). All analysis, modeling, and predictions are contained within the executable notebook format.

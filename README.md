# 🏋️ Powerlifting Deadlift Prediction

A machine learning regression project that predicts a powerlifter's **Best Deadlift performance (`BestDeadliftKg`)** using competition and athlete-level information from the OpenPowerlifting dataset.

The project follows an end-to-end machine learning workflow, including **exploratory data analysis, data cleaning, leakage detection, outlier analysis, preprocessing, model comparison, hyperparameter tuning, evaluation, and model serialization**.

---

## 📌 Project Overview

Powerlifting performance can vary based on factors such as:

* Squat performance
* Bench press performance
* Bodyweight
* Sex
* Equipment used

This project investigates these factors and develops machine learning models to predict a lifter's **Best Deadlift in kilograms**.

### 🎯 Problem Type

**Supervised Learning → Regression**

### 🎯 Target Variable

```text
BestDeadliftKg
```

The target is a continuous numerical value representing the lifter's best recorded deadlift in kilograms.

---

## 📂 Dataset

The project uses the **OpenPowerlifting dataset**.

Dataset file:

```text
openpowerlifting.csv
```

The EDA notebook performs an initial investigation of:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Unique values
* Numerical distributions
* Categorical distributions
* Correlations
* Outliers
* Relationships between features and the target

---

# 🔎 Exploratory Data Analysis

The EDA was performed separately before model development.

### 1. Data Quality Analysis

The dataset was examined for:

* Missing values
* Duplicate records
* Invalid numerical values
* High-cardinality / identifier columns
* Potential target leakage
* Unusable or redundant columns

The analysis identified **545 exact duplicate rows**, which were removed.

---

## 🔐 Target Leakage Detection

A major part of the EDA was checking for features that could indirectly reveal the target.

`TotalKg` was investigated because:

```text
TotalKg = BestSquatKg + BestBenchKg + BestDeadliftKg
```

The analysis showed that `TotalKg` closely matches this calculation, meaning it directly contains information about the target.

Therefore:

```text
TotalKg → Dropped
```

`Wilks` was also removed because it is calculated using total performance and bodyweight, creating a potential leakage pathway.

Other columns removed during preprocessing included:

```text
MeetID
Division
Squat4Kg
Bench4Kg
Deadlift4Kg
TotalKg
Place
Wilks
```

---

## 🧹 Data Cleaning

The following cleaning steps were performed:

### Duplicate Removal

```python
df = df.drop_duplicates()
```

### Target Missing Values

Rows without a valid `BestDeadliftKg` were removed.

### Weight Class Conversion

`WeightClassKg` values such as:

```text
120+
```

were converted into numerical values.

### Invalid Lift Values

Negative values in:

```text
BestSquatKg
BestBenchKg
BestDeadliftKg
```

were identified as failed attempts rather than valid performance values.

Rows where these lifts were not positive were removed.

### Missing Values

The EDA found substantial missingness in `Age`.

Instead of deleting a large portion of the dataset, missing values were handled later through the modeling pipeline using imputation.

---

# 📊 EDA Insights

Some important observations from the analysis:

### Performance Distributions

`BestSquatKg`, `BestBenchKg`, and `BodyweightKg` show right-skewed distributions, with a smaller number of high-performing lifters forming the upper tail.

### Feature Relationships

Among the numerical features analyzed:

```text
BestSquatKg → strongest relationship with BestDeadliftKg
BestBenchKg → weaker relationship
BodyweightKg → weaker but positive relationship
```

### Categorical Features

The analysis also examined the relationship between:

* Sex and Deadlift
* Equipment and Deadlift
* Sex + Equipment and Deadlift

These variables provide additional information beyond the numerical lifting measurements.

---

# 🧠 Machine Learning Workflow

The modeling notebook follows this pipeline:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Train / Test Split
     ↓
Outlier Analysis
     ↓
Preprocessing Pipeline
     ↓
Multiple Regression Models
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Final XGBoost Model
     ↓
Model Evaluation
     ↓
Save Model as .pkl
```

---

# 🛠️ Features Used for Modeling

The final model uses:

### Numerical Features

```text
BestSquatKg
BestBenchKg
BodyweightKg
```

### Nominal Feature

```text
Sex
```

### Equipment Feature

```text
Equipment
```

### Target

```text
BestDeadliftKg
```

---

# ⚙️ Data Preprocessing

A `ColumnTransformer` and multiple `Pipeline` components were used to keep preprocessing and modeling together.

### Numerical Pipeline

```text
Median Imputation
       ↓
StandardScaler
```

### Nominal Pipeline

```text
Most-Frequent Imputation
       ↓
OneHotEncoder
```

### Equipment Pipeline

The equipment feature is processed separately as an ordinal feature.

This preprocessing is integrated directly into the machine learning pipeline to ensure that the same transformations are applied consistently during training and prediction.

---

# 🚨 Outlier Analysis

Two approaches were investigated:

### IQR Method

The Interquartile Range method was used to calculate:

```text
Q1
Q3
IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

### Isolation Forest

An `IsolationForest` model was also evaluated for detecting anomalous observations.

```python
IsolationForest(
    contamination=0.01,
    random_state=42
)
```

The modeling analysis found that removing the detected outliers did not meaningfully change model performance in this experiment.

---

# 🤖 Models Compared

The following regression algorithms were evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet
* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* AdaBoost Regressor
* Extra Trees Regressor
* HistGradientBoosting Regressor
* XGBoost Regressor

Evaluation metrics included:

```text
MAE
MSE
RMSE
R²
Train R²
```

---

# 📈 Initial Model Comparison

| Model                | MAE (kg) | RMSE (kg) | Test R² |
| -------------------- | -------: | --------: | ------: |
| XGBoost              |    14.93 |     20.22 |  0.8803 |
| HistGradientBoosting |    15.04 |     20.37 |  0.8786 |
| Gradient Boosting    |    15.20 |     20.59 |  0.8760 |
| Random Forest        |    15.39 |     21.08 |  0.8700 |
| KNN Regressor        |    16.24 |     21.84 |  0.8604 |
| Linear Regression    |    16.32 |     22.08 |  0.8573 |
| Ridge                |    16.32 |     22.08 |  0.8573 |
| Lasso                |    16.47 |     22.26 |  0.8550 |
| Extra Trees          |    16.38 |     22.88 |  0.8467 |
| ElasticNet           |    20.14 |     25.99 |  0.8023 |
| Decision Tree        |    19.59 |     27.71 |  0.7753 |
| AdaBoost             |    26.28 |     35.21 |  0.6372 |

---

# 🎯 Hyperparameter Tuning

Hyperparameter optimization was performed using:

```text
GridSearchCV
RandomizedSearchCV
```

with:

```text
5-fold Cross Validation
```

### Grid Search

Used for models with relatively smaller parameter spaces:

* Ridge
* Decision Tree

### Randomized Search

Used for larger parameter spaces:

* Random Forest
* Gradient Boosting
* XGBoost

For Randomized Search:

```text
n_iter = 15
cv = 5
scoring = R²
random_state = 42
```

---

# 📊 Tuning Results

| Model             | Before Tuning | After Tuning | Best CV R² |
| ----------------- | ------------: | -----------: | ---------: |
| Random Forest     |        0.8700 |       0.8824 |     0.8787 |
| XGBoost           |        0.8803 |       0.8807 |     0.8770 |
| Gradient Boosting |        0.8760 |       0.8800 |     0.8760 |
| Decision Tree     |        0.7753 |       0.8742 |     0.8704 |
| Ridge             |        0.8573 |       0.8573 |     0.8544 |

---

# 🚀 Final Model

The final saved model is an **XGBoost regression pipeline**.

The final pipeline contains:

```text
Preprocessing
     ↓
XGBRegressor
```

Final XGBoost configuration:

```python
XGBRegressor(
    random_state=42,
    subsample=1.0,
    reg_lambda=1,
    reg_alpha=0.01,
    n_estimators=300,
    min_child_weight=5,
    max_depth=7,
    learning_rate=0.05,
    gamma=0,
    colsample_bytree=0.7
)
```

---

# 🏆 Final Model Evaluation

The final XGBoost pipeline achieved:

| Metric      |       Result |
| ----------- | -----------: |
| MAE         | **14.90 kg** |
| MSE         |   **407.46** |
| RMSE        | **20.19 kg** |
| R²          |   **0.8807** |
| Adjusted R² |   **0.8807** |

### Interpretation

The final model achieved an **R² of 0.8807**, meaning that the model explains approximately **88.07% of the variance** in `BestDeadliftKg` on the held-out test set.

The MAE of **14.90 kg** means that, on average, the model's predictions differ from the actual deadlift values by approximately 14.9 kg in this test set.

---

# 💾 Model Serialization

The complete preprocessing + model pipeline is saved using `joblib`:

```python
joblib.dump(final_pipeline, 'final_model.pkl')
```

This saves the entire pipeline rather than only the XGBoost estimator, allowing the same preprocessing steps to be applied when making predictions later.

---

# 📁 Project Structure

```text
Powerlifting-Deadlift-Prediction/
│
├── PowerLifting_Dataset_EDA.ipynb
├── powerLifting_models_Notebook.ipynb
├── openpowerlifting.csv
├── openpowerlifting_cleaned.csv
│
├── model/
│   └── final_model.pkl
│
└── README.md
```

---

# 💻 Technologies Used

### Programming

* Python

### Data Manipulation

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Google Colab

---

# 📚 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Data Cleaning
* Missing Value Handling
* Duplicate Detection
* Feature Selection
* Target Leakage Detection
* Outlier Detection
* IQR Method
* Isolation Forest
* Train-Test Split
* Numerical Feature Scaling
* One-Hot Encoding
* Ordinal Feature Processing
* ColumnTransformer
* Scikit-learn Pipeline
* Regression Algorithms
* Cross-Validation
* GridSearchCV
* RandomizedSearchCV
* Hyperparameter Tuning
* Regression Evaluation Metrics
* Model Serialization with Joblib

---


🚀 Deployment

The trained machine learning pipeline has been integrated into a Streamlit web application for interactive predictions.

The application allows users to enter the required powerlifting-related features and receive a predicted Best Deadlift (kg) from the trained XGBoost regression model.

Streamlit Application

Live Demo: Add your deployed Streamlit URL here

Application Workflow
User Input
    ↓
Streamlit Interface
    ↓
Saved ML Pipeline
    ↓
Data Preprocessing
    ↓
XGBoost Regressor
    ↓
Predicted Best Deadlift (kg)

The application uses the saved final_model.pkl, which contains the preprocessing and trained model pipeline. This ensures that the same preprocessing steps used during model training are applied to new user inputs.

## Files

```
app.py                          # Home page — the prediction form
pages/
  1_Visualizations.py           # Data exploration
  2_Model_Insights.py           # Model performance & comparison
final_model.pkl                 # Trained sklearn Pipeline (preprocessing + tuned XGBoost)
metrics.json                    # R², Adjusted R², MAE, RMSE on the test set
model_comparison.csv            # All 12 models tested, before tuning
feature_importance.csv          # XGBoost feature importances
actual_vs_predicted_sample.csv  # 3,000-row sample for the Actual vs Predicted chart
lifting_data_sample.csv         # 20,000-row sample used for the Visualizations page
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit auto-detects the `pages/` folder and adds the other two pages to the sidebar
navigation — no extra config needed.

## Deploy to Streamlit Community Cloud

1. Push this whole folder to a **public GitHub repo** (all the files above, keeping the
   folder structure — `pages/` must stay a folder, not renamed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file path to `app.py`.
4. Deploy. Build takes a couple of minutes the first time (installing scikit-learn/xgboost).

**Notes for a smooth deploy:**
- `requirements.txt` pins `scikit-learn==1.8.0` and `xgboost==3.4.1` to match the versions
  the model was trained with — pickled sklearn/XGBoost models don't always load cleanly
  across version jumps, so keep these pinned unless you retrain.
- All data files are small samples (~3 MB total), not the full raw dataset, so the repo
  stays lightweight and the app loads fast.
- If you retrain the model later, regenerate `final_model.pkl`, `metrics.json`,
  `model_comparison.csv`, and `feature_importance.csv` together so the Model Insights page
  stays consistent with the deployed model.

---

🔮 Future Improvements

Possible future improvements include:

Adding prediction confidence or prediction intervals
Improving feature engineering
Adding more detailed prediction error analysis
Supporting athlete-level historical performance data
Adding additional model comparison options
Improving the Streamlit user interface
Adding automated model retraining
Deploying the model through a production API

---

# 👨‍💻 Author

**Abhijit Palai**

Data Science | Machine Learning | Python | SQL

GitHub: [Abhijit-palai](https://github.com/Abhijit-palai)

---

## ⭐ Project Highlights

> **An end-to-end regression project focused on predicting powerlifting deadlift performance, with emphasis on data quality, leakage prevention, preprocessing pipelines, model comparison, hyperparameter optimization, and reproducible model deployment.**





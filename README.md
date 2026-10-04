# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer churn prediction is a machine learning project that predicts whether a customer is likely to **leave a telecom service (Churn)** or **continue using the service (No Churn)**.

In this project, customer information such as tenure, monthly charges, contract type, internet service, payment method, and other service details are used to train machine learning classification models.

The project includes **data preprocessing, exploratory data analysis (EDA), categorical encoding, SMOTE for handling class imbalance, model comparison, model evaluation, and a predictive system**.

---

## 🎯 Objectives

* Analyze customer data and identify important patterns.
* Preprocess and clean the dataset.
* Convert categorical data into numerical form.
* Handle class imbalance using **SMOTE**.
* Train and compare multiple machine learning models.
* Evaluate the performance of the selected model.
* Save the trained model using Pickle.
* Build a system that predicts whether a customer will churn.

---

## 📊 Dataset

The project uses the **Telco Customer Churn Dataset**.

### Dataset Information

* **Total records:** 7,043
* **Original features:** 21
* **Features after preprocessing:** 19
* **Target variable:** `Churn`

### Target Classes

| Churn | Meaning  | Count |
| ----- | -------- | ----: |
| 0     | No Churn | 5,174 |
| 1     | Churn    | 1,869 |

The dataset contains an imbalance between customers who churn and those who do not.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Pickle
* Google Colab / Jupyter Notebook

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

1. Decision Tree
2. Random Forest
3. XGBoost

### Default Model Comparison

| Model         | Cross-Validation Accuracy |
| ------------- | ------------------------: |
| Decision Tree |                       78% |
| Random Forest |                   **84%** |
| XGBoost       |                       83% |

Based on the 5-fold cross-validation results, **Random Forest** achieved the highest average accuracy among the three models.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Handle Class Imbalance using SMOTE
   ↓
Model Training
   ↓
Cross-Validation
   ↓
Model Selection
   ↓
Test Set Evaluation
   ↓
Save Model
   ↓
Load Model
   ↓
Customer Churn Prediction
```

---

## 🧹 Data Preprocessing

### 1. Removing Customer ID

The `customerID` column was removed because it is an identifier and does not provide useful information for predicting churn.

```python
df = df.drop(columns=["customerID"])
```

### 2. Handling Missing Values

The `TotalCharges` column contained 11 blank values.

These blank values were replaced with `0.0` and converted into the `float` datatype.

```python
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
```

### 3. Encoding Target Variable

The `Churn` column was converted into numerical values:

```text
Yes → 1
No  → 0
```

### 4. Encoding Categorical Features

Categorical columns were converted into numerical values using `LabelEncoder`.

The encoders were saved using Pickle so that the same encoding can be used when making predictions on new customer data.

---

## 📈 Exploratory Data Analysis

EDA was performed to understand the distribution and relationships between different features.

The project includes:

* Histogram analysis
* Box plots
* Correlation heatmap
* Count plots for categorical features
* Numerical feature statistics

### Numerical Features

The main numerical features analyzed were:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

---

## ⚖️ Handling Class Imbalance with SMOTE

The original target distribution was:

```text
No Churn  → 5174
Churn     → 1869
```

Since the dataset is imbalanced, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied only to the training data.

Before SMOTE:

```text
No Churn → 4138
Churn    → 1496
```

After SMOTE:

```text
No Churn → 4138
Churn    → 4138
```

This creates a balanced training dataset.

```python
smote = SMOTE(random_state=42)

X_train_smote, Y_train_smote = smote.fit_resample(
    X_train,
    Y_train
)
```

---

## 🌲 Random Forest Model

Random Forest was selected based on the cross-validation comparison.

```python
rfc = RandomForestClassifier(random_state=42)

rfc.fit(
    X_train_smote,
    Y_train_smote
)
```

---

## 📊 Model Evaluation

The trained Random Forest model was evaluated on the original test dataset.

### Accuracy

```text
Accuracy: 77.86%
```

### Confusion Matrix

```text
[[878 158]
 [154 219]]
```

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| No Churn             |      0.85 |   0.85 |     0.85 |
| Churn                |      0.58 |   0.59 |     0.58 |
| **Overall Accuracy** |           |        | **0.78** |

The model achieved approximately **77.86% test accuracy**.

---

## 🔮 Predictive System

After training, the Random Forest model was saved using Pickle.

```python
model_data = {
    "model": rfc,
    "features_names": X.columns.tolist()
}

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
```

The categorical encoders were also saved:

```python
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
```

These files can be loaded later to make predictions without retraining the model.

---

## 🧪 Example Prediction

A sample customer was provided to the predictive system.

The model returned:

```text
Prediction: No Churn
Prediction Probability: [[0.78 0.22]]
```

This means the model predicted:

```text
No Churn → 78%
Churn    → 22%
```

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── encoders.pkl
└── README.md
```

> If the dataset or model files are not included in your repository, remove them from this structure.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the Notebook

Open:

```text
Customer_Churn_Prediction.ipynb
```

using Google Colab or Jupyter Notebook.

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### 4. Add the Dataset

Place the Telco Customer Churn CSV file in the appropriate location.

If using Google Colab:

```python
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### 5. Run the Notebook

Run the cells sequentially to:

* Load the dataset
* Perform preprocessing
* Perform EDA
* Apply SMOTE
* Train models
* Compare models
* Evaluate Random Forest
* Save the trained model
* Make predictions

---

## 📌 Key Learnings

Through this project, I learned:

* Data loading and preprocessing using Pandas
* Exploratory Data Analysis
* Handling missing values
* Categorical feature encoding
* Train-test splitting
* Handling imbalanced datasets using SMOTE
* Training classification models
* Comparing machine learning models
* Cross-validation
* Confusion matrix and classification reports
* Model persistence using Pickle
* Building a basic machine learning prediction pipeline

---

## 🔧 Future Improvements

The following improvements can be implemented:

* [ ] Hyperparameter tuning
* [ ] Stratified K-Fold Cross-Validation
* [ ] Try different model-selection techniques
* [ ] Experiment with downsampling
* [ ] Address model overfitting
* [ ] Optimize Random Forest parameters
* [ ] Compare additional classification algorithms
* [ ] Perform feature importance analysis
* [ ] Build a web-based prediction interface
* [ ] Deploy the model using Flask or Streamlit

---

## 💡 Future Scope

This project can be extended into a complete **Customer Churn Prediction Application** where users can enter customer information through a web interface and receive:

```text
Customer Information
        ↓
Machine Learning Model
        ↓
Churn Probability
        ↓
Churn / No Churn
```

The system could also provide recommendations for identifying customers who have a high probability of leaving the service.

---

## 👨‍💻 Author

**Paramesh Gorentla**

B.Tech — Artificial Intelligence & Machine Learning

---

## ⭐ Project Highlights

```text
Dataset              → 7,043 customers
Problem Type         → Binary Classification
Class Imbalance      → Handled using SMOTE
Models Tested        → Decision Tree, Random Forest, XGBoost
Best CV Model        → Random Forest
Test Accuracy        → 77.86%
Model Saving         → Pickle
Prediction           → Churn / No Churn
```

If you found this project useful, consider giving the repository a ⭐.

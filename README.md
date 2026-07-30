# 🏆 The-Classification-Models-League-Adults-Income-Edition 
A comparative benchmarking of 6 machine learning classification models (RF, Logistic Regression, SVM, KNN, Decision Trees, Naive Bayes) predicting individual income levels (More or Less than 50K) using the UCI Adult dataset.

---

## 📌 Project Overview
The objective of this project is to evaluate and rank six different classification algorithms head-to-head on the **UCI Adult Income** dataset. We handled missing data, processed numerical and categorical features using `scikit-learn` Pipelines, tuned hyperparameters (finding optimal $K$ for KNN), and compared the performance across multiple evaluation metrics.

---

## 🚀 Key Features & Pipeline Steps
* **Data Cleaning & Preprocessing:** Handled missing `?` values, removed duplicates, and dropped non-predictive columns (`fnlwgt`).
* **Scikit-Learn Pipelines:** Built modular `ColumnTransformer` pipelines with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features.
* **Model Training & Optimization:** Evaluated 6 algorithms with class balancing and optimal parameter selection.
* **Evaluation Metrics:** Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.

---

## 📊 Final Ranks & Model Comparison

| Rank | Model | ROC-AUC | Accuracy | F1-Score |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 **1** | **Random Forest** | **0.91** | **79.78%** | **80.94%** |
| 🥈 **2** | **SVM** | **0.90** | **80.14%** | **81.24%** |
| 🥉 **3** | **Decision Tree** | **0.89** | **79.66%** | **80.83%** |
| **4** | **Logistic Regression** | 0.89 | 79.66% | 80.73% |
| **5** | **KNN** | 0.81 | 82.50% | 80.01% |
| **6** | **Naive Bayes** | 0.72 | 50.64% | 51.47% |
---

## 💡 Key Takeaways
1. **Random Forest** achieved the highest overall ROC-AUC score, effectively capturing non-linear feature interactions in demographic data. Since RF is not affected by outliers, it's the most suitable model of the 6 due to the realistic outliers in the data's nature (especially the features 'Capital Gain' and 'Capital Loss'.
2. **Logistic Regression** served as a surprisingly strong baseline model when paired with standardized numerical scaling and one-hot encoding.
3. Feature scaling and proper class balancing (`class_weight='balanced'`) were crucial due to the dataset's target imbalance (~76% `<=50K` vs ~24% `>50K`).
---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

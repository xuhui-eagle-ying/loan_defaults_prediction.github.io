# Loan Default Prediction - Machine Learning Project

## Project Overview

This project involves a detailed analysis of a dataset containing over 30,000 data points on loan status. The goal is to predict loan status (approved, rejected, or pending) and reduce loan defaults by identifying the key factors influencing loan approval. The analysis includes data cleansing, feature engineering, model training, and performance evaluation.

## Steps Involved

### 1. Data Preprocessing
- **Data Cleansing**: Removed missing, duplicate, and irrelevant data points to ensure the dataset is accurate and ready for analysis.
- **Feature Engineering**: Derived new features and transformed existing ones to enhance the predictive power of the models.
- **Exploratory Data Analysis (EDA)**: Conducted thorough exploratory analysis to understand distributions, detect trends, and visualize relationships between features.

### 2. Anomaly Detection
- **Isolation Forest**: Used the Isolation Forest algorithm to detect and isolate anomalies in the data.
- **Anomalies Removed**: Detected and removed 5 anomalies that could negatively affect model performance.

### 3. Model Building

#### a. Logistic Regression & Lasso
- **Objective**: Used Logistic Regression with Lasso regularization to identify significant predictors of loan status and reduce the risk of loan defaults.
- **Key Findings**: Identified critical factors influencing loan approval and rejection.

#### b. Advanced Machine Learning Models
- **Neural Networks**: Trained a neural network to capture complex non-linear relationships in the data.
- **Random Forest**: Applied Random Forest to evaluate the importance of each feature and predict loan status.
- **XGBoost**: Utilized XGBoost for its high efficiency and performance in classification tasks.

### 4. Model Evaluation
- **Performance Metrics**: Evaluated all models using key performance indicators, including:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **AUC-ROC Curve**: Achieved 98% AUC, demonstrating excellent model performance in distinguishing between loan statuses.

- **Confusion Matrix**: Created confusion matrices to visualize the model's performance in terms of true positives, false positives, true negatives, and false negatives.

### 5. Results and Recommendations
- **Interpretation**: Presented the results in both technical and non-technical formats, ensuring stakeholders from various backgrounds can understand the analysis and conclusions.
- **Actionable Insights**: Provided actionable recommendations to the financial institution to help reduce loan defaults and improve the loan approval process.

## Conclusion

This project demonstrates the effective use of data science techniques, including machine learning, anomaly detection, and model evaluation, to solve real-world problems in the financial industry. By identifying key factors influencing loan status, the models provide valuable insights that can help reduce defaults and improve decision-making processes in the lending sector.

---

### Tools and Technologies Used:
- **Programming Language**: Python
- **Libraries/Packages**: `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`, `xgboost`
- **Algorithms**: Logistic Regression, Lasso, Neural Network, Random Forest, XGBoost
- **Tools**: Jupyter Notebooks

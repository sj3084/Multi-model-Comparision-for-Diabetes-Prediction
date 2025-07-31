# Evaluating Machine Learning Models for Diabetes Prediction and Classification

## Objectives

- Predict diabetes risk using demographic, clinical, and behavioral data.
- Evaluate and compare ML models including Random Forest, Logistic Regression, KNN, Decision Tree, Naive Bayes, Gradient Boosting, and XGBoost.
- Address class imbalance using SMOTE.
- Implement hybrid feature selection using Random Forest importance and Recursive Feature Elimination (RFE).
- Deploy the best-performing model in a web-based prediction tool.

---

## Methodology

1. **Data Acquisition**  
   Publicly available diabetes dataset with features like BMI, age, cholesterol, physical health, etc.

2. **Preprocessing**  
   - Missing value imputation  
   - Normalization  
   - Categorical encoding  
   - SMOTE for class balancing

3. **Feature Selection**  
   - Random Forest Importance  
   - Recursive Feature Elimination (RFE)

4. **Model Training**  
   - Algorithms: RF, LR, KNN, DT, NB, GB, XGB  
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score  

5. **Model Evaluation**  
   - Confusion Matrix  
   - ROC-AUC  
   - Class-wise performance  
   - Visualizations for performance comparison

6. **Web Deployment (Future work)**  
   - Frontend: User inputs health metrics  
   - Backend: Best model (XGBoost) provides risk prediction

---

## Results Summary

| Model                   | Accuracy | Class 0 F1 | Class 1 F1 | Class 2 F1 | Macro Avg F1 | Weighted Avg F1 |
|------------------------|----------|------------|------------|------------|----------------|-------------------|
| RandomForestClassifier | 0.64     | 0.78       | 0.06       | 0.43       | 0.42           | 0.72              |
| LogisticRegression     | 0.73     | 0.84       | 0.05       | 0.37       | 0.42           | 0.76              |
| KNeighborsClassifier   | 0.77     | 0.87       | 0.03       | 0.31       | 0.40           | 0.77              |
| DecisionTreeClassifier | 0.62     | 0.75       | 0.05       | 0.42       | 0.41           | 0.69              |
| GaussianNB             | 0.82     | 0.90       | 0.00       | 0.43       | 0.44           | 0.82              |
| GradientBoosting       | 0.85     | 0.91       | 0.00       | 0.33       | 0.41           | 0.82              |
| XGBClassifier          | 0.85     | 0.92       | 0.00       | 0.28       | 0.40           | 0.81              |

### Key Observations

- **GradientBoosting** and **XGBoost** achieved the highest **accuracy (85%)** with excellent **Class 0 F1**, but failed to detect **Class 1** (F1 = 0).
- **GaussianNB** had the **highest macro average F1-score (0.44)**, making it the most balanced overall performer.
- **RandomForestClassifier** showed the **highest recall for Class 1 (0.32)**, despite having lower overall accuracy.
- All models struggled with **Class 1**, indicating that further class imbalance mitigation is needed.

---


# diabetes_model_evaluation.py

import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from xgboost import XGBClassifier


def load_data(path):
    """Load dataset and return feature/target split."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Diabetes_012"])
    y = df["Diabetes_012"].astype(int)
    return df, X, y


def preprocess(X, y):
    """Split and scale the data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def define_models():
    """Define and return all models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train, evaluate, and store results for all models."""
    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        end = time.time()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Time": end - start,
            "y_pred": y_pred,
            "y_score": y_score
        })
    
    return pd.DataFrame(results), results


def plot_accuracy(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Accuracy", y="Model", palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.show()


def plot_stability(results_df):
    metrics_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].set_index("Model")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=metrics_df.T)
    plt.title("Model Stability - Box Plot")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.show()


def plot_roc(results, y_test):
    plt.figure(figsize=(10, 6))
    for res in results:
        if res["y_score"] is not None:
            fpr, tpr, _ = roc_curve(y_test == 2, res["y_score"][:, 2])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{res["Model"]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve (Class 2 = Diabetes)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def plot_precision_recall(results, y_test):
    plt.figure(figsize=(10, 6))
    for res in results:
        if res["y_score"] is not None:
            precision, recall, _ = precision_recall_curve(y_test == 2, res["y_score"][:, 2])
            plt.plot(recall, precision, label=res["Model"])
    plt.title("Precision-Recall Curve (Class 2 = Diabetes)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, res in enumerate(results[:9]):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(res["Model"])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.suptitle("Confusion Matrices", fontsize=16, y=1.02)
    plt.show()


def plot_execution_time(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Time", y="Model", palette="mako")
    plt.title("Execution Time per Model")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Model")
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.drop(columns=["Diabetes_012"]).corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()


def main():
    df, X, y = load_data("dataset2.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X, y)
    models = define_models()
    results_df, results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)

    # Plots
    plot_accuracy(results_df)
    plot_stability(results_df)
    plot_roc(results, y_test)
    plot_precision_recall(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_execution_time(results_df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()

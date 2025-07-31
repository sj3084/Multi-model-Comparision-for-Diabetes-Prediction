# diabetes_classification_pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


def load_and_prepare_data(file_path):
    """Load dataset and split into features and target."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=["Diabetes_012"])
    y = df["Diabetes_012"]
    return df, X, y


def balance_and_scale(X, y):
    """Apply train-test split, SMOTE, and standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Class distribution BEFORE SMOTE:", Counter(y_train))

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Class distribution AFTER SMOTE:", Counter(y_train_resampled))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test


def get_models():
    """Return dictionary of classifiers."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train and evaluate each model; return results."""
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

        print(f"\n----- {name} -----")
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted Precision: {prec:.4f}")
        print(f"Weighted Recall: {rec:.4f}")
        print(f"Weighted F1 Score: {f1:.4f}")
        print("\nPer-Class Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))

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
    return results


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
    df, X, y = load_and_prepare_data("dataset2.csv")
    X_train_scaled, X_test_scaled, y_train_resampled, y_test = balance_and_scale(X, y)
    models = get_models()
    results = evaluate_models(models, X_train_scaled, y_train_resampled, X_test_scaled, y_test)
    results_df = pd.DataFrame(results)

    plot_accuracy(results_df)
    plot_stability(results_df)
    plot_roc(results, y_test)
    plot_precision_recall(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_execution_time(results_df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()

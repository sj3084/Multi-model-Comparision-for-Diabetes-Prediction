# feature_importance_rf.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath):
    """Load dataset and split into features and target."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Diabetes_012"])
    y = df["Diabetes_012"]
    return X, y

def plot_feature_importance(X, y):
    """Train RandomForest and plot sorted feature importances."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    importances_sorted.plot(kind='barh', color='skyblue')
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importances_sorted

def main():
    X, y = load_data("dataset2.csv")
    feature_importances = plot_feature_importance(X, y)
    print("\nTop Features:")
    print(feature_importances)

if __name__ == "__main__":
    main()

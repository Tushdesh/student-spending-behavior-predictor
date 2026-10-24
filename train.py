"""
train.py
--------
Full ML pipeline for the Student Spending Behaviour Predictor.

Two tasks are solved on the same dataset:
  1. Regression     -> predict the exact monthly spending (INR)
  2. Classification -> predict the spending category (Saver / Balanced / Overspender)

Steps: load data -> EDA plots -> preprocessing -> train & compare models
       -> cross-validation -> evaluate best model -> save model + plots + report.

Run:  python train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/student_spending.csv"
OUT = "outputs"
MODELS = "models"

CATEGORICAL = ["gender", "course", "city_tier", "living_situation",
               "transport_mode", "tracks_budget", "uses_bnpl"]
NUMERIC = ["age", "year_of_study", "monthly_allowance", "part_time_income",
           "food_orders_per_week", "outings_per_month", "online_shopping_orders",
           "subscriptions", "social_media_hours", "upi_txn_per_week"]
FEATURES = NUMERIC + CATEGORICAL
REG_TARGET = "monthly_spending"
CLS_TARGET = "spending_category"
CLASS_ORDER = ["Saver", "Balanced", "Overspender"]

sns.set_theme(style="whitegrid")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def eda(df: pd.DataFrame) -> None:
    print("\n=== Dataset overview ===")
    print(df.shape)
    print(df.describe().T[["mean", "std", "min", "max"]].round(1))
    print("Missing values:", int(df.isna().sum().sum()))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[REG_TARGET], bins=40, kde=True, ax=ax[0], color="#3b6ea5")
    ax[0].set_title("Distribution of monthly spending (INR)")
    sns.countplot(x=CLS_TARGET, data=df, order=CLASS_ORDER, ax=ax[1],
                  palette=["#4caf50", "#ffb300", "#e53935"], hue=CLS_TARGET, legend=False)
    ax[1].set_title("Spending category counts")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_target_distribution.png", dpi=120); plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[NUMERIC + [REG_TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(x="living_situation", y=REG_TARGET, data=df, ax=ax[0])
    ax[0].set_title("Spending by living situation")
    sns.boxplot(x="city_tier", y=REG_TARGET, data=df, ax=ax[1],
                order=["Tier 1", "Tier 2", "Tier 3"])
    ax[1].set_title("Spending by city tier")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_spending_by_group.png", dpi=120); plt.close()


def train_regression(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n=== Regression: predicting monthly spending ===")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "CV_R2": round(float(cv), 4),
            "Test_R2": round(r2_score(y_te, pred), 4),
            "MAE": round(mean_absolute_error(y_te, pred), 1),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 1),
        }
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda k: results[k]["Test_R2"])
    print(f"Best regressor: {best}")
    best_pipe = fitted[best]
    joblib.dump(best_pipe, f"{MODELS}/spending_regressor.pkl")

    pred = best_pipe.predict(X_te)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, alpha=0.4, s=14, color="#3b6ea5")
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("Actual spending (INR)"); plt.ylabel("Predicted spending (INR)")
    plt.title(f"Actual vs Predicted — {best}")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_actual_vs_predicted.png", dpi=120); plt.close()

    pd.DataFrame(results).T.to_csv(f"{OUT}/regression_results.csv")
    return {"best": best, "results": results}


def train_classification(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n=== Classification: predicting spending category ===")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                        random_state=42),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "CV_Accuracy": round(float(cv), 4),
            "Test_Accuracy": round(accuracy_score(y_te, pred), 4),
            "F1_macro": round(f1_score(y_te, pred, average="macro"), 4),
        }
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda k: results[k]["F1_macro"])
    print(f"Best classifier: {best}")
    best_pipe = fitted[best]
    joblib.dump(best_pipe, f"{MODELS}/spending_classifier.pkl")

    pred = best_pipe.predict(X_te)
    report = classification_report(y_te, pred, labels=CLASS_ORDER)
    print(report)
    with open(f"{OUT}/classification_report.txt", "w") as f:
        f.write(f"Best model: {best}\n\n{report}")

    cm = confusion_matrix(y_te, pred, labels=CLASS_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix — {best}")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_confusion_matrix.png", dpi=120); plt.close()

    # Feature importance (tree models) — mapped back to readable names
    model = best_pipe.named_steps["model"]
    names = best_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):            # linear model: mean |coefficient|
        scores = np.abs(model.coef_).mean(axis=0)
    else:
        scores = None
    if scores is not None:
        imp = pd.Series(scores, index=names).sort_values(ascending=False)[:15]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=imp.values, y=imp.index, color="#3b6ea5")
        plt.title("Top 15 features driving spending category")
        plt.tight_layout(); plt.savefig(f"{OUT}/06_feature_importance.png", dpi=120); plt.close()
        print("Top drivers:", ", ".join(imp.index[:5]))

    pd.DataFrame(results).T.to_csv(f"{OUT}/classification_results.csv")
    return {"best": best, "results": results}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise SystemExit("Dataset not found. Run `python generate_data.py` first.")

    df = pd.read_csv(DATA_PATH)
    eda(df)

    X = df[FEATURES]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, df[REG_TARGET], df[CLS_TARGET], test_size=0.2, random_state=42,
        stratify=df[CLS_TARGET])

    summary = {
        "regression": train_regression(X_tr, X_te, yr_tr, yr_te),
        "classification": train_classification(X_tr, X_te, yc_tr, yc_te),
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nModels saved in models/, plots and reports in outputs/")


if __name__ == "__main__":
    main()

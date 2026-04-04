import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json


def train_models():
    # Load the cleaned + feature-engineered data
    df = pd.read_csv("../data/car_data_features.csv")

    # Separate features (X) from target (y)
    # Drop price AND log_price — both are target variants
    X = df.drop(["price", "log_price", "is_luxury"], axis=1)
    y = df["price"]

    # ■■ TRAIN/TEST SPLIT ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # This is the leakage boundary — happens here, not before
    # test_size=0.2 = 20% for testing, 80% for training
    # random_state=42 = same split every time you run (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    # ■■ BUILD PIPELINES ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
            # alpha=1.0 = regularisation strength
            # Ridge adds a penalty for large coefficients
            # prevents overfitting on small datasets
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1.0))
            # Lasso can set some coefficients to exactly zero
            # effectively doing feature selection automatically
        ]),
    }

    results = {}
    best_r2 = -999
    best_name = None

    for name, pipeline in models.items():
        # ■■ TRAIN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        pipeline.fit(X_train, y_train)

        # ■■ PREDICT ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        y_pred = pipeline.predict(X_test)

        # ■■ EVALUATE ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # R2 = 1.0 is perfect, 0.0 = no better than guessing mean
        # Target: R2 > 0.80

        # Cross-validation: split train into 5 parts,
        # train on 4, test on 1, repeat 5 times — more reliable
        cv = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

        results[name] = {
            "MAE":     round(mae, 2),   # average dollar error
            "RMSE":    round(rmse, 2),  # penalises big errors more
            "R2":      round(r2, 4),    # % of variance explained
            "CV_mean": round(cv.mean(), 4),
            "CV_std":  round(cv.std(), 4)
        }

        print(f"{name}: MAE=${mae:.0f} RMSE=${rmse:.0f} R2={r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline

    # ■■ SAVE MODEL ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # joblib serialises the entire Pipeline to a file
    # The app loads this file instead of retraining every time
    joblib.dump(best_pipeline,         "../models/best_model.joblib")
    joblib.dump(list(X.columns),       "../models/feature_columns.joblib")

    with open("../models/results.json", "w") as f:
        json.dump(results, f)

    print(f"\nBest model: {best_name} (R2={best_r2:.3f})")
    print("Model saved to models/")


if __name__ == "__main__":
    train_models()
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train(data_path: str, outdir: str):
    df = pd.read_csv(data_path)
    # features and target
    X = df[["square_feet", "bedrooms", "bathrooms", "age_years"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, out / "model.pkl")

    # Save metrics
    metrics = {"r2": float(r2), "rmse": float(rmse)}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot predictions vs actual
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Linear Regression — Predictions vs Actual")
    # y=x reference line
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    fig_path = Path("reports/figures/fit_plot.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(f"Saved model to {out/'model.pkl'}")
    print(f"Saved metrics to {out/'metrics.json'}")
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/house_prices.csv")
    parser.add_argument("--outdir", type=str, default="models")
    args = parser.parse_args()
    train(args.data, args.outdir)

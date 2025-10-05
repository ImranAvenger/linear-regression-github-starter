# Linear Regression Project — GitHub Showcase

![CI](https://img.shields.io/github/actions/workflow/status/USER/REPO/ci.yml?label=CI&logo=github)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

A clean, ready-to-publish template for a Linear Regression project using scikit-learn.  
It includes a synthetic dataset, a reproducible training script, tests, a Jupyter notebook, and GitHub Actions CI.

## 🚀 Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# 2) Train the model
python src/train.py --data data/raw/house_prices.csv --outdir models

# 3) See metrics
cat models/metrics.json
```

## 🧠 What’s inside
- **`data/raw/house_prices.csv`** — synthetic data to get started
- **`src/train.py`** — training script (saves metrics, model, and a plot)
- **`notebooks/Linear_Regression.ipynb`** — EDA + model notebook
- **`tests/test_train.py`** — smoke test for CI
- **`.github/workflows/ci.yml`** — runs tests and training on push/PR
- **`reports/figures/fit_plot.png`** — generated regression fit plot (after training)

## 📈 Example Results
After running `train.py`, artifacts are saved to `models/`:
- `model.pkl` — trained scikit-learn model
- `metrics.json` — R² and RMSE
- `reports/figures/fit_plot.png` — predictions vs. actual

## 🧪 Run tests
```bash
pytest -q
```

## 📄 License
MIT © 2025

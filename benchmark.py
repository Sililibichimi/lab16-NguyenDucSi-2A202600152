#!/usr/bin/env python3
"""
LightGBM Credit Card Fraud Detection Benchmark
Run on the EC2 CPU node after SSH-ing in:
  python3 benchmark.py
  python3 benchmark.py --use-kaggle   # if kaggle.json is configured
"""
import argparse
import time
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
import lightgbm as lgb

RESULT_FILE = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
MODEL_FILE  = os.path.expanduser("~/ml-benchmark/model.lgb")


def load_kaggle_dataset(path: str) -> pd.DataFrame:
    print(f"Loading Kaggle dataset from {path} ...")
    t0 = time.time()
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.2f}s")
    return df, time.time() - t0


def generate_synthetic(n: int = 50_000) -> tuple:
    print(f"Generating synthetic dataset ({n:,} rows) ...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 28))
    amount = np.abs(rng.exponential(88, n))
    time_col = np.linspace(0, 172_792, n)
    # Weak signal so AUC is realistic but not trivial
    logit = X[:, 0] * 0.5 + X[:, 1] * 0.3 - 1.5
    fraud_p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < fraud_p * 0.1).astype(int)
    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = amount
    df["Time"] = time_col
    df["Class"] = y
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.2f}s")
    return df, elapsed


def train_and_evaluate(df: pd.DataFrame, load_time: float) -> dict:
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,} | Fraud rate: {y.mean()*100:.3f}%")

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds   = lgb.Dataset(X_test,  label=y_test, reference=train_ds)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "n_jobs":           -1,
        "verbose":          -1,
        "scale_pos_weight": int((y == 0).sum() / max((y == 1).sum(), 1)),
    }

    print("\nTraining LightGBM ...")
    t0 = time.time()
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=callbacks,
    )
    train_time = time.time() - t0
    print(f"Training done in {train_time:.2f}s | Best iteration: {model.best_iteration}")

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_prob)
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred)

    # Latency benchmarks
    single = X_test.iloc[:1]
    t0 = time.time()
    for _ in range(100):
        model.predict(single)
    lat_1 = (time.time() - t0) / 100 * 1000  # ms

    batch = X_test.iloc[:1000]
    t0 = time.time()
    model.predict(batch)
    lat_1000 = (time.time() - t0) * 1000  # ms
    throughput = 1000 / (lat_1000 / 1000)

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"  Load / generate time:    {load_time:.2f}s")
    print(f"  Training time:           {train_time:.2f}s")
    print(f"  Best iteration:          {model.best_iteration}")
    print(f"  AUC-ROC:                 {auc:.4f}")
    print(f"  Accuracy:                {acc:.4f}")
    print(f"  F1-Score:                {f1:.4f}")
    print(f"  Precision:               {precision:.4f}")
    print(f"  Recall:                  {recall:.4f}")
    print(f"  Inference latency (1):   {lat_1:.3f} ms")
    print(f"  Throughput (1000 rows):  {throughput:.0f} rows/s")
    print("=======================================")

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    result = {
        "instance_type":                   "r5.2xlarge",
        "dataset_rows":                    len(df),
        "load_time_s":                     round(load_time, 2),
        "train_time_s":                    round(train_time, 2),
        "best_iteration":                  model.best_iteration,
        "auc_roc":                         round(auc, 4),
        "accuracy":                        round(acc, 4),
        "f1_score":                        round(f1, 4),
        "precision":                       round(precision, 4),
        "recall":                          round(recall, 4),
        "inference_latency_1row_ms":       round(lat_1, 3),
        "inference_throughput_1000rows_per_s": round(throughput, 0),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults written to {RESULT_FILE}")

    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-kaggle", action="store_true",
                        help="Download Credit Card Fraud dataset from Kaggle first")
    parser.add_argument("--csv", default="~/ml-benchmark/creditcard.csv",
                        help="Path to creditcard.csv if already downloaded")
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv)

    if args.use_kaggle and not os.path.exists(csv_path):
        import subprocess
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "mlg-ulb/creditcardfraud",
             "--unzip", "-p", os.path.dirname(csv_path)],
            check=True,
        )

    if os.path.exists(csv_path):
        df, load_time = load_kaggle_dataset(csv_path)
    else:
        df, load_time = generate_synthetic()

    train_and_evaluate(df, load_time)


if __name__ == "__main__":
    main()

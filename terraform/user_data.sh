#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== CPU ML benchmark setup (LightGBM on t2.micro) ==="

# Add 1 GB swap to help with limited RAM
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Update system and install Python
dnf update -y
dnf install -y python3 python3-pip gcc gcc-c++ make

pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy flask

# Create working directory
mkdir -p /home/ec2-user/ml-benchmark
chown ec2-user:ec2-user /home/ec2-user/ml-benchmark

# Write benchmark.py
cat > /home/ec2-user/ml-benchmark/benchmark.py << 'PYEOF'
#!/usr/bin/env python3
"""LightGBM benchmark — sized for t2.micro (1 GB RAM + 1 GB swap)."""
import time
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)
import lightgbm as lgb

RESULT_FILE = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
MODEL_FILE  = os.path.expanduser("~/ml-benchmark/model.lgb")

# t2.micro has 1 GB RAM — keep dataset at 50k rows to avoid OOM
N_ROWS = 50_000

print("=== LightGBM Credit Card Fraud Detection Benchmark ===")
print(f"LightGBM version: {lgb.__version__}")

csv_path = os.path.expanduser("~/ml-benchmark/creditcard.csv")
if os.path.exists(csv_path):
    print(f"Loading real dataset from {csv_path} ...")
    t0 = time.time()
    df = pd.read_csv(csv_path, nrows=N_ROWS)
    load_time = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {load_time:.2f}s")
else:
    print(f"Generating synthetic dataset ({N_ROWS:,} rows) ...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_ROWS, 28))
    amount = np.abs(rng.exponential(88, N_ROWS))
    time_col = np.linspace(0, 172_792, N_ROWS)
    logit = X[:, 0] * 0.5 + X[:, 1] * 0.3 - 1.5
    fraud_p = 1 / (1 + np.exp(-logit))
    y_arr = (rng.random(N_ROWS) < fraud_p * 0.1).astype(int)
    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = amount
    df["Time"] = time_col
    df["Class"] = y_arr
    load_time = time.time() - t0
    print(f"  Generated in {load_time:.2f}s")

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Fraud rate: {y.mean()*100:.3f}%")

train_ds = lgb.Dataset(X_train, label=y_train)
val_ds   = lgb.Dataset(X_test,  label=y_test, reference=train_ds)

params = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "n_jobs":           1,
    "verbose":          -1,
    "scale_pos_weight": int((y == 0).sum() / max((y == 1).sum(), 1)),
}

print("\nTraining LightGBM ...")
t0 = time.time()
callbacks = [lgb.early_stopping(30, verbose=True), lgb.log_evaluation(30)]
model = lgb.train(
    params, train_ds,
    num_boost_round=300,
    valid_sets=[val_ds],
    callbacks=callbacks,
)
train_time = time.time() - t0
print(f"Done in {train_time:.2f}s | Best iteration: {model.best_iteration}")

y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

auc       = roc_auc_score(y_test, y_prob)
acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred)

single = X_test.iloc[:1]
t0 = time.time()
for _ in range(100):
    model.predict(single)
lat_1 = (time.time() - t0) / 100 * 1000

batch = X_test.iloc[:min(1000, len(X_test))]
t0 = time.time()
model.predict(batch)
lat_batch = (time.time() - t0) * 1000
throughput = len(batch) / (lat_batch / 1000)

print("\n========== BENCHMARK RESULTS ==========")
print(f"  Load time:               {load_time:.2f}s")
print(f"  Training time:           {train_time:.2f}s")
print(f"  Best iteration:          {model.best_iteration}")
print(f"  AUC-ROC:                 {auc:.4f}")
print(f"  Accuracy:                {acc:.4f}")
print(f"  F1-Score:                {f1:.4f}")
print(f"  Precision:               {precision:.4f}")
print(f"  Recall:                  {recall:.4f}")
print(f"  Inference latency (1):   {lat_1:.3f} ms")
print(f"  Throughput ({len(batch)} rows):  {throughput:.0f} rows/s")
print("=======================================")

result = {
    "instance_type":                      "t2.micro",
    "dataset_rows":                       len(df),
    "load_time_s":                        round(load_time, 2),
    "train_time_s":                       round(train_time, 2),
    "best_iteration":                     model.best_iteration,
    "auc_roc":                            round(auc, 4),
    "accuracy":                           round(acc, 4),
    "f1_score":                           round(f1, 4),
    "precision":                          round(precision, 4),
    "recall":                             round(recall, 4),
    "inference_latency_1row_ms":          round(lat_1, 3),
    "inference_throughput_rows_per_s":    round(throughput, 0),
}
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, "w") as fh:
    json.dump(result, fh, indent=2)
print(f"\nResults saved to {RESULT_FILE}")

model.save_model(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
PYEOF

# Write API server
cat > /home/ec2-user/ml-benchmark/api_server.py << 'APIEOF'
#!/usr/bin/env python3
"""Inference API — LightGBM on port 8000."""
import json
import os
import time
import numpy as np
from flask import Flask, request, jsonify
import lightgbm as lgb

app = Flask(__name__)
MODEL_PATH = os.path.expanduser("~/ml-benchmark/model.lgb")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = lgb.Booster(model_file=MODEL_PATH)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": model is not None}), 200

@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def completions():
    data = request.get_json(force=True)
    if model is None:
        return jsonify({"error": "Model not ready yet"}), 503
    features = data.get("features")
    if features is None:
        return jsonify({
            "id": "lgb-cpu-01",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content":
                "LightGBM fraud-detection API. POST with {\"features\": [V1..V28, Amount, Time]}"
            }}],
        })
    t0 = time.time()
    prob = float(model.predict(np.array(features, dtype=float).reshape(1, -1))[0])
    return jsonify({
        "id": "lgb-cpu-01",
        "object": "completion",
        "model": "lightgbm-fraud-detector",
        "fraud_probability": round(prob, 6),
        "prediction": "FRAUD" if prob >= 0.5 else "LEGITIMATE",
        "inference_latency_ms": round((time.time() - t0) * 1000, 3),
    })

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000)
APIEOF

chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark

echo "=== Running benchmark ==="
sudo -u ec2-user python3 /home/ec2-user/ml-benchmark/benchmark.py

echo "=== Starting API server on port 8000 ==="
sudo -u ec2-user nohup python3 /home/ec2-user/ml-benchmark/api_server.py \
  > /var/log/api_server.log 2>&1 &

echo "=== Setup complete ==="

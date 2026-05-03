import os
import sys
import subprocess
import time
import json
import signal
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

API_URL         = "http://127.0.0.1:8000"
PREDICT_URL     = f"{API_URL}/predict"
HEALTH_URL      = f"{API_URL}/health"
LOG_PATH        = "logs/predictions.jsonl"


def run_script(description: str, script: str):
    """Run a Python script and print its output."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {description}")
    print(f"  Running: python {script}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n❌ FAILED: {script} exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} — complete\n")


def start_api():
    """Start the FastAPI server in the background."""
    print(f"\n{'='*60}")
    print(f"  STAGE: Start API Server")
    print(f"{'='*60}\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the server to be ready
    for attempt in range(20):
        time.sleep(1)
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            if resp.status_code == 200:
                print(f"✅ API is healthy: {resp.json()}")
                return proc
        except requests.ConnectionError:
            pass

    print("❌ API failed to start within 20 seconds")
    proc.terminate()
    sys.exit(1)


def send_test_traffic(n_normal=15, n_drifted=10):
    """Send a mix of normal and drifted requests."""
    print(f"\n{'='*60}")
    print(f"  STAGE: Send Production Traffic ({n_normal} normal + {n_drifted} drifted)")
    print(f"{'='*60}\n")

    import random
    random.seed(42)

    for i in range(n_normal):
        payload = {
            "distance_km": round(random.uniform(0.5, 10.0), 1),
            "items_count": random.randint(1, 6),
            "is_peak_hour": random.choice([0, 1]),
            "traffic_level": random.choice([1, 2, 3]),
        }
        resp = requests.post(PREDICT_URL, json=payload)
        pred = resp.json()["delivery_time_min"]
        print(f"  [Normal  {i+1:02d}] dist={payload['distance_km']:5.1f} → {pred:.1f} min")

    print("\n  --- Drift begins ---\n")

    for i in range(n_drifted):
        payload = {
            "distance_km": round(random.uniform(15.0, 25.0), 1),
            "items_count": random.randint(7, 12),
            "is_peak_hour": 1,
            "traffic_level": 3,
        }
        resp = requests.post(PREDICT_URL, json=payload)
        pred = resp.json()["delivery_time_min"]
        print(f"  [Drifted {i+1:02d}] dist={payload['distance_km']:5.1f} → {pred:.1f} min")

    print(f"\n✅ Sent {n_normal + n_drifted} predictions\n")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     QuickFoods MLOps — Full Lifecycle Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Clean up previous prediction logs
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        print(f"Cleaned previous log: {LOG_PATH}")

    # ── Stage 1: Train baseline ───────────────────────────────────────
    run_script(
        "Train Baseline Model (Exercise 1)",
        "src/train.py"
    )

    # ── Stage 3: Multi-metric comparison ──────────────────────────────
    run_script(
        "Multi-Metric Model Comparison (Exercise 3)",
        "src/train_multi_metrics_with_mlflow.py"
    )

    # ── Stage 4: Hyperparameter tuning ────────────────────────────────
    run_script(
        "Hyperparameter Tuning (Exercise 5)",
        "src/train_hyperparameter_tuning.py"
    )

    # ── Stage 5: Register best model ──────────────────────────────────
    run_script(
        "Register Best Model (Exercise 5)",
        "src/register_best_model.py"
    )

    # ── Stage 6: Promote to champion ──────────────────────────────────
    run_script(
        "Assign Champion Alias (Exercise 8)",
        "src/promote_model.py"
    )

    # ── Stage 7: Start API and serve ──────────────────────────────────
    api_proc = start_api()

    try:
        # ── Stage 8: Send traffic ─────────────────────────────────────
        send_test_traffic(n_normal=15, n_drifted=10)

        # ── Stage 9: Monitor ──────────────────────────────────────────
        run_script(
            "Monitor Predictions and Check Drift (Exercise 7)",
            "src/monitor.py"
        )

    finally:
        # Stop the API server
        print("\nStopping API server...")
        api_proc.terminate()
        api_proc.wait(timeout=5)
        print("API server stopped.\n")

    # ── Stage 10: Retrain ─────────────────────────────────────────────
    run_script(
        "Retrain on New Data and Promote (Exercise 9)",
        "src/retrain.py"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               Pipeline Complete                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                        ║")
    print("║  ✅ Trained baseline model                              ║")
    print("║  ✅ Tracked experiments in MLflow                       ║")
    print("║  ✅ Compared 3 models on multiple metrics               ║")
    print("║  ✅ Tuned hyperparameters (grid + random search)        ║")
    print("║  ✅ Registered best model in MLflow registry            ║")
    print("║  ✅ Served model via FastAPI                            ║")
    print("║  ✅ Logged predictions (normal + drifted traffic)       ║")
    print("║  ✅ Detected input drift via monitoring                 ║")
    print("║  ✅ Retrained on combined data                          ║")
    print("║  ✅ Promoted new champion (if improved)                 ║")
    print("║                                                        ║")
    print("║  View full history: mlflow ui → http://127.0.0.1:5000  ║")
    print("║  Prediction log: logs/predictions.jsonl                ║")
    print("║                                                        ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
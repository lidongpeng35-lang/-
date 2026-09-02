from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "hypoglycemia_calculator_bundle.joblib"

app = Flask(__name__)


def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Missing hypoglycemia_calculator_bundle.joblib. "
            "Place the model bundle next to app.py."
        )
    return joblib.load(BUNDLE_PATH)


bundle = load_bundle()


def predict_probability(raw_values):
    # Keep the original GitHub prediction pipeline unchanged.
    features = bundle["features"]
    x_raw = pd.DataFrame([[raw_values[f] for f in features]], columns=features)
    x_imp = bundle["imputer"].transform(x_raw)
    x_scaled = bundle["scaler"].transform(x_imp)
    x_model = pd.DataFrame(x_scaled, columns=features)
    return float(bundle["model"].predict_proba(x_model)[:, 1][0])


@app.get("/")
def home():
    med = bundle["feature_medians"]
    defaults = {
        "landmark_hour": float(med["landmark_hour"]),
        "weight_kg": float(np.clip(med["weight_kg"], 30, 180)),
        "glu_min_6h": float(np.clip(med["glu_min_6h"], 30, 350)),
        "glu_last_6h": float(np.clip(med["glu_last_6h"], 30, 400)),
        "glu_cv_6h": float(np.clip(med["glu_cv_6h"], 0, 1.5)),
        "insulin_units_6h": float(np.clip(med["insulin_units_6h"], 0, 100)),
        "sofa_score": int(np.clip(round(med["sofa_score"]), 0, 24)),
        "creatinine_last_24h": float(np.clip(med["creatinine_last_24h"], 0, 15)),
        "base_excess_last_24h": float(np.clip(med["base_excess_last_24h"], -25, 20)),
    }
    return render_template(
        "index.html",
        defaults=defaults,
        threshold=float(bundle["threshold"]),
    )


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    no_glu = int(data.get("no_glu_6h", 0))

    raw_values = {
        "base_excess_last_24h": float(data["base_excess_last_24h"]),
        "creatinine_last_24h": float(data["creatinine_last_24h"]),
        "diabetes": int(data["diabetes"]),
        "glu_cv_6h": np.nan if no_glu else float(data["glu_cv_6h"]),
        "glu_last_6h": np.nan if no_glu else float(data["glu_last_6h"]),
        "glu_min_6h": np.nan if no_glu else float(data["glu_min_6h"]),
        "insulin_units_6h": float(data["insulin_units_6h"]),
        "landmark_hour": float(data["landmark_hour"]),
        "no_glu_6h": no_glu,
        "sofa_score": int(data["sofa_score"]),
        "weight_kg": float(data["weight_kg"]),
    }

    prob = predict_probability(raw_values)
    threshold = float(bundle["threshold"])

    return jsonify(
        probability=prob,
        probability_pct=prob * 100.0,
        threshold=threshold,
        threshold_pct=threshold * 100.0,
        high_risk=bool(prob >= threshold),
        landmark_hour=raw_values["landmark_hour"],
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

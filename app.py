from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Folder name is "Model" based on your dir
model = joblib.load("Model/model.pkl")

REQUIRED_FIELDS = [
    "delivery_days",
    "delivery_vs_estimated",
    "price",
    "freight_value",
    "product_category",
    "seller_state",
    "payment_type",
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})

def validate_record(rec):
    errors = {}
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errors[f] = "missing"
    return errors

def predict_batch(records):
    df = pd.DataFrame(records)
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    labels = np.where(preds == 1, "positive", "negative")
    return preds, proba, labels

@app.route("/predict", methods=["POST"])
def predict_single():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid input"}), 400
    errors = validate_record(payload)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400
    preds, proba, labels = predict_batch([payload])
    return jsonify({
        "prediction": int(preds[0]),
        "probability": float(proba[0]),
        "label": labels[0],
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    payload = request.get_json()
    if not isinstance(payload, list) or len(payload) == 0:
        return jsonify({"error": "Expected non-empty list"}), 400
    if len(payload) > 100:
        return jsonify({"error": "Limit 100 records per request"}), 400

    all_errors = [validate_record(r) for r in payload]
    if any(all_errors):
        return jsonify({"error": "Invalid input", "details": all_errors}), 400

    preds, proba, labels = predict_batch(payload)
    results = [
        {"prediction": int(p), "probability": float(pr), "label": lab}
        for p, pr, lab in zip(preds, proba, labels)
    ]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
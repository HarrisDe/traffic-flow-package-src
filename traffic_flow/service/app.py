# traffic_flow/service/app.py
from __future__ import annotations
import os
from flask import Flask, request, jsonify
import pandas as pd
from .runtime import InferenceRuntime

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "artifacts/traffic_pipeline_h-15.joblib")

def create_app() -> Flask:
    app = Flask(__name__)
    rt = InferenceRuntime(ARTIFACT_PATH)  # load model + states once at startup

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"}), 200

    @app.get("/metadata")
    def metadata():
        return jsonify({
            "horizon": rt.horizon,
            "n_features": len(rt.feature_cols),
            "feature_example": rt.feature_cols[:10],
        }), 200

    @app.post("/predict")
    def predict():
        """
        Expect JSON:
        {
          "records": [
            {"sensor_id":"...", "date":"YYYY-MM-DD HH:MM:SS", "value": 83.0},
            ...
          ]
        }
        """
        payload = request.get_json(force=True)
        if not payload or "records" not in payload:
            return jsonify({"error": "Payload must include 'records' list"}), 400

        df_raw = pd.DataFrame(payload["records"])
        needed = ("sensor_id", "date", "value")
        missing = [c for c in needed if c not in df_raw.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        preds, _ = rt.predict_df(df_raw)
        return jsonify({
            "horizon": rt.horizon,
            "n": int(len(preds)),
            "predictions": preds.astype(float).tolist()
        }), 200

    return app

if __name__ == "__main__":
    # Dev runner; for prod use gunicorn
    app = create_app()
    app.run("0.0.0.0", int(os.getenv("PORT", "8080")), debug=False)
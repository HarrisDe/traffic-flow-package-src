# traffic_flow/service/app.py
from __future__ import annotations
import os
from flask import Flask, request, jsonify
import pandas as pd
from .runtime import InferenceRuntime

# We read where the artifact lives from an environment variable.
# If you don't set it, the default points to artifacts/traffic_pipeline_h-15.joblib
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "artifacts/traffic_pipeline_h-15.joblib")
print(f"Artifact path: {ARTIFACT_PATH}")

def create_app() -> Flask:
    app = Flask(__name__)
    
    # Load the artifact ONCE at startup: model + pipeline states.
    # This avoids expensive reloading on every request.
    rt = InferenceRuntime(ARTIFACT_PATH)  # load model + states once at startup

    @app.get("/healthz")
    def healthz():
        # If this function returns 200 OK, an orchestrator (or you)
        # knows the process is running and accepting requests
        return jsonify({"status": "ok"}), 200

    @app.get("/metadata")
    def metadata():
        # Useful to confirm what model is loaded,
        # how many features it expects, and a sample of feature names.
        return jsonify({
            "horizon": rt.horizon,
            "n_features": len(rt.feature_cols),
            "feature_example": rt.feature_cols[:10],
        }), 200

    @app.post("/predict")
    def predict():
        """
        This endpoint expects a JSON object with a 'records' array, where each item
        is a row of your RAW data schema, e.g.:

        {
          "records": [
            {"sensor_id":"...", "date":"YYYY-MM-DD HH:MM:SS", "value": 83.0},
            ...
          ]
        }
        """
        # request.get_json(force=True) reads the HTTP request body and parses JSON.
        payload = request.get_json(force=True)
        
        # Validate that we actually got JSON with a 'records' field
        if not payload or "records" not in payload:
            return jsonify({"error": "Payload must include 'records' list"}), 400
        
        # Turn JSON rows into a pandas DataFrame for the inference pipeline
        df_raw = pd.DataFrame(payload["records"])
        # Validate required columns are present
        needed = ("sensor_id", "date", "value")
        missing = [c for c in needed if c not in df_raw.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # Run the runtime: this applies your feature engineering + model predict
        preds, _ = rt.predict_df(df_raw)
        
        # Return predictions as a JSON response
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
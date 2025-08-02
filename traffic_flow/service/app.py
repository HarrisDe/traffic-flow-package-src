# traffic_flow/service/app.py
from __future__ import annotations
import os
from flask import Flask, request, jsonify
import pandas as pd
from .runtime import InferenceRuntime

def create_app(artifact_path: str | None = None) -> Flask:
    """
    Create and return a Flask app. If artifact_path is None, we read
    ARTIFACT_PATH from the environment (default: artifacts/traffic_pipeline_h-15.joblib).
    """
    artifact_path = artifact_path or os.getenv("ARTIFACT_PATH", "artifacts/traffic_pipeline_h-15.joblib")
    print(f"[service] Using artifact: {artifact_path}")

    app = Flask(__name__)
    rt = InferenceRuntime(artifact_path)  # load model + states once at startup

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"}), 200

    @app.get("/metadata")
    def metadata():
        return jsonify({
            "horizon": rt.horizon,
            "n_features": len(rt.feature_cols),
            "features": rt.feature_cols,
        }), 200

    # @app.post("/predict")
    # def predict():
    #     """
    #     Expects JSON:
    #     {
    #       "records": [
    #         {"sensor_id": "...", "date":"YYYY-MM-DD HH:MM:SS", "value": 83.0, ...},
    #         ...
    #       ]
    #     }
    #     Returns JSON with rows including: sensor_id, date, prediction_time, y_pred_delta, y_pred_total.
    #     """
    #     payload = request.get_json(force=True)
    #     if not payload or "records" not in payload:
    #         return jsonify({"error": "Payload must include 'records' list"}), 400

    #     raw_df = pd.DataFrame(payload["records"])
    #     needed = ("sensor_id", "date", "value")
    #     missing = [c for c in needed if c not in raw_df.columns]
    #     if missing:
    #         return jsonify({"error": f"Missing columns: {missing}"}), 400

    #     # 1) Feature engineering + model delta predictions
    #     pred_delta, feats = rt.predict_df(raw_df)   # <- delta only

    #     # 2) Add baseline (current value) to get total speed
    #     if "value" not in feats.columns:
    #         return jsonify({"error": "The engineered features are missing 'value' column."}), 500
    #     pred_total = (pred_delta + feats["value"].to_numpy()).astype(float)

    #     # 3) Attach timestamps & ids for plotting/verification
    #     dt_col = rt.states["datetime_state"]["datetime_col"]  # usually "date"
    #     dt_in = pd.to_datetime(raw_df[dt_col], errors="coerce")
    #     dt_pred = dt_in + pd.to_timedelta(rt.horizon, unit="m")

    #     out = pd.DataFrame({
    #         "sensor_id": raw_df.get("sensor_id", None),
    #         dt_col: dt_in.dt.strftime("%Y-%m-%d %H:%M:%S"),
    #         "prediction_time": dt_pred.dt.strftime("%Y-%m-%d %H:%M:%S"),
    #         "y_pred_delta": pred_delta.astype(float),
    #         "y_pred_total": pred_total,
    #     })

    #     return jsonify({
    #         "horizon": rt.horizon,
    #         "n": int(len(out)),
    #         "predictions": out.to_dict(orient="records")
    #     }), 200

    # return app

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True)
        if not payload or "records" not in payload:
            return jsonify({"error": "Payload must include 'records' list"}), 400

        df_raw = pd.DataFrame(payload["records"])
        # minimal validation
        for c in ("sensor_id", "date", "value"):
            if c not in df_raw.columns:
                return jsonify({"error": f"Missing column: {c}"}), 400

        pred_df, _ = rt.predict_df(df_raw)
        return jsonify({
            "horizon": rt.horizon,
            "n": int(len(pred_df)),
            "predictions": pred_df.to_dict(orient="records")
        }), 200

    return app

if __name__ == "__main__":
    # Dev runner; for prod use gunicorn
    port = int(os.getenv("PORT", "8080"))
    app = create_app()
    app.run("0.0.0.0", port, debug=False)

# We read where the artifact lives from an environment variable.
# If you don't set it, the default points to artifacts/traffic_pipeline_h-15.joblib
# ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "artifacts/traffic_pipeline_h-15.joblib")
# print(f"Artifact path: {ARTIFACT_PATH}")

# def create_app(artifact_path: str | None = None) -> Flask:
#     artifact_path = artifact_path or os.getenv("ARTIFACT_PATH", "artifacts/traffic_pipeline_h-15.joblib")
#     app = Flask(__name__)
    
#     # Load the artifact ONCE at startup: model + pipeline states.
#     # This avoids expensive reloading on every request.
#     rt = InferenceRuntime(ARTIFACT_PATH)  # load model + states once at startup

#     @app.get("/healthz")
#     def healthz():
#         # If this function returns 200 OK, an orchestrator (or you)
#         # knows the process is running and accepting requests
#         return jsonify({"status": "ok"}), 200

#     @app.get("/metadata")
#     def metadata():
#         # Useful to confirm what model is loaded,
#         # how many features it expects, and a sample of feature names.
#         return jsonify({
#             "horizon": rt.horizon,
#             "n_features": len(rt.feature_cols),
#             "features": rt.feature_cols,
#         }), 200


#     @app.post("/predict")
#     def predict():
#         """
#         Expects JSON:
#         {
#           "records": [
#             {"sensor_id": "...", "date":"YYYY-MM-DD HH:MM:SS", "value": 83.0, ...},
#             ...
#           ]
#         }
#         """
#         payload = request.get_json(force=True)
#         if not payload or "records" not in payload:
#             return jsonify({"error": "Payload must include 'records' list"}), 400

#         # 1) Raw input to DataFrame
#         raw_df = pd.DataFrame(payload["records"])

#         # Basic validation for the core columns the pipeline needs
#         needed = ("sensor_id", "date", "value")
#         missing = [c for c in needed if c not in raw_df.columns]
#         if missing:
#             return jsonify({"error": f"Missing columns: {missing}"}), 400
 
#         # 2) Build features + get model predictions (delta if trained on deltas)
#         pred_delta, feats = rt.predict_df(raw_df)

#         # 3) Compute total if the model was delta-based and 'value' is among inputs
#         if "value" in feats.columns:
#             pred_total = (pred_delta + feats["value"].to_numpy()).astype(float)
#         else:
#             return jsonify({"error": f"Missing columns: value"}), 400

#         # 4) Attach original datetime and computed prediction_time
#         dt_col = rt.states["datetime_state"]["datetime_col"]  # usually "date"
#         dt     = pd.to_datetime(raw_df[dt_col], errors="coerce")
#         pred_time = dt + pd.to_timedelta(rt.horizon, unit="m")

#         out = pd.DataFrame({
#             "sensor_id": raw_df.get("sensor_id", None),
#             dt_col: dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
#             "prediction_time": pred_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
#             "y_pred_delta": pred_delta.astype(float),
#             "y_pred_total": pred_total,
#         })

#         return jsonify({
#             "horizon": rt.horizon,
#             "n": int(len(out)),
#             "predictions": out.to_dict(orient="records")
#         }), 200

#     return app

# if __name__ == "__main__":
#     # Dev runner; for prod use gunicorn
#     app = create_app()
#     app.run("0.0.0.0", int(os.getenv("PORT", "8080")), debug=False)
    
    
    
    # @app.post("/predict")
    # def predict():
    #     """
    #     This endpoint expects a JSON object with a 'records' array, where each item
    #     is a row of your RAW data schema, e.g.:

    #     {
    #       "records": [
    #         {"sensor_id":"...", "date":"YYYY-MM-DD HH:MM:SS", "value": 83.0},
    #         ...
    #       ]
    #     }
    #     """
    #     # request.get_json(force=True) reads the HTTP request body and parses JSON.
    #     payload = request.get_json(force=True)
        
    #     # Validate that we actually got JSON with a 'records' field
    #     if not payload or "records" not in payload:
    #         return jsonify({"error": "Payload must include 'records' list"}), 400
        
    #     # Turn JSON rows into a pandas DataFrame for the inference pipeline
    #     df_raw = pd.DataFrame(payload["records"])
    #     # Validate required columns are present
    #     needed = ("sensor_id", "date", "value")
    #     missing = [c for c in needed if c not in df_raw.columns]
    #     if missing:
    #         return jsonify({"error": f"Missing columns: {missing}"}), 400

    #     # Run the runtime: this applies your feature engineering + model predict
    #     preds, _ = rt.predict_df(df_raw)
        
    #     # Return predictions as a JSON response
    #     return jsonify({
    #         "horizon": rt.horizon,
    #         "n": int(len(preds)),
    #         "predictions": preds.astype(float).tolist()
    #     }), 200

    # return app

# scripts/tri_compare_predictions.py
from __future__ import annotations
import os, json, time, argparse, threading
from pathlib import Path
from typing import Iterable, Tuple
import sys
import numpy as np
import pandas as pd
import requests
import joblib

# Add the repo root (parent of `scripts/`) to sys.path
try:
    ROOT = Path(__file__).resolve().parents[1]  # When run as .py
except NameError:
    ROOT = Path().resolve().parents[0]          # When run in Jupyter

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from your package (assumes `pip install -e .`)
from traffic_flow.service.app import create_app
from traffic_flow.service.runtime import InferenceRuntime
from traffic_flow.pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
from traffic_flow.inference.prediction_protocol import make_prediction_frame
from traffic_flow.evaluation.model_comparison import ModelEvaluator
print("OKKKK!!!!!!")
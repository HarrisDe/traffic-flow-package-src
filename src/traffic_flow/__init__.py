# src/traffic_flow/__init__.py
from __future__ import annotations

# Always-light runtime exports
from .service.runtime import InferenceRuntime
from .inference.inference_pipeline import TrafficInferencePipeline

# Optional/heavier training APIs (don’t break import if absent)
try:
    from .pipeline.data_pipeline_orchestrator import TrafficDataPipelineOrchestrator
except ImportError:
    TrafficDataPipelineOrchestrator = None  

try:
    from .modeling.model_tuning import ModelTunerXGB
except ImportError:
    ModelTunerXGB = None  

try:
    from .evaluation.model_comparison import ModelEvaluator
except ImportError:
    ModelEvaluator = None  

# Optional: package version
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("traffic-flow")
except PackageNotFoundError:  # e.g., during editable dev without dist metadata
    __version__ = "0.0.0"

__all__ = [
    "InferenceRuntime",
    "TrafficInferencePipeline",
    "TrafficDataPipelineOrchestrator",
    "ModelTunerXGB",
    "ModelEvaluator",
    "__version__",
]
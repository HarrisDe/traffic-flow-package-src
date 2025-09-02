from abc import ABC, abstractmethod
from gc import disable
import pandas as pd
from     typing import Iterable, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from ...utils.helper_utils import LoggingMixin 


    
    
    
class FeatureTransformer(ABC, LoggingMixin):
    """
    Abstract base class for all feature transformation classes.
    Includes logging and a unified transform interface.
    """

    def __init__(self, disable_logs: bool = False):
        super().__init__(disable_logs=disable_logs)

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature transformation logic.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        pass
    

class BaseAggregator(ABC):
    """Strategy interface for any aggregation flavour."""

    @abstractmethod
    def aggregate(self, values: Iterable[float]) -> float:
        """Return a single statistic from `values` (NaNs ignored)."""
        
        

class BaseFeatureTransformer(BaseEstimator, TransformerMixin, LoggingMixin):
    """
    Common utilities & logging for all custom transformers.
    Uses sklearn's BaseEstimator and TransformerMixin to make it a sklearn transformer.
    This allows to use the feature classes in a production pipeline (which transforms any input data to ML-ready data)
    """

    def __init__(self, disable_logs: bool = False):
        super().__init__(disable_logs=disable_logs)
    

    # sklearn calls fit once, then transform many times
    def fit(self, X, y=None):
        return self  # most simple transformers don’t need to learn anything

    def transform(self, X):
        raise NotImplementedError("Every subclass must implement transform()")

    # Optional: keep a small helper
    def _log(self, msg: str):
        if not self.disable_logs:
            print(f"[{self.__class__.__name__}] {msg}")
    
        # Helper called at end of from_state() in subclasses
    def _mark_fitted(self):
        self.fitted_ = True
        return self
            
            
            
class SensorEncodingStrategy(BaseFeatureTransformer,LoggingMixin):
    """Abstract base for all sensor encoders."""

    def __init__(
        self,
        *,
        sensor_col: str = "sensor_id",
        new_sensor_col: str = "sensor_uid",
        disable_logs: bool = False,
    ):
        super().__init__(disable_logs=disable_logs)
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col
        self.fitted_ = False  # sklearn convention: trailing _ means set in fit()

    def fit(self, X, y=None):  # overridden in subclasses
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # must override
        raise NotImplementedError

    # -- persistence helpers (optional but recommended) --
    def export_state(self) -> Dict[str, Any]:
        """Return JSON-serializable state for inference."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SensorEncodingStrategy":
        """Rebuild instance from export_state()."""
        raise NotImplementedError
    
    
    
    
from abc import ABC, abstractmethod
import pandas as pd
from ..helper_utils import LoggingMixin 

class SensorEncodingStrategy(ABC, LoggingMixin):
    """
    Abstract base class for all sensor encoding strategies with logging support.
    """

    def __init__(self, sensor_col: str = "sensor_id", new_sensor_col: str = "sensor_uid", disable_logs: bool = False):
        super().__init__(disable_logs)
        self.sensor_col = sensor_col
        self.new_sensor_col = new_sensor_col

    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the given sensor column in the dataframe.

        Args:
            df (pd.DataFrame): Full dataset.
            is_train (pd.Series): Boolean mask for training rows.

        Returns:
            pd.DataFrame: Modified dataframe with encoded sensor_col.
        """
        pass
    
    
    
class FeatureTransformer(ABC, LoggingMixin):
    """
    Abstract base class for all feature transformation classes.
    Includes logging and a unified transform interface.
    """

    def __init__(self, disable_logs: bool = False):
        super().__init__(disable_logs)

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
    
    
    
    
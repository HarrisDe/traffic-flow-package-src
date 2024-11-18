import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .constants import colnames

class TrafficFlowDataProcessing:
    """
    A class to process and prepare traffic flow data for time-series prediction.
    This includes methods for loading, cleaning, feature engineering, and data splitting.
    """

    def __init__(self, file_path='../data/estimated_average_speed_selected_timestamps-edited-new.csv', column_names=None, lags=20, test_size=0.75, correlation_threshold=0.01,columns_to_use=None,sensor_id_col_name = 'sensor_uid', random_state=69):
        """
        Initialize data processing parameters.

        Parameters:
        - file_path (str): Path to the CSV file with traffic data.
        - column_names (list): Column names for the data file. If None, defaults to colnames.
        - lags (int): Number of temporal lag features to generate.
        - test_size (float): Proportion of data to reserve for testing.
        - correlation_threshold (float): Minimum correlation for feature selection.
        - random_state (int): Seed for reproducible train-test split.
        """
        self.file_path = file_path
        self.csv_column_names = column_names if column_names else colnames  # Use default if None
        self.lags = lags
        self.test_size = test_size
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.sensor_id_col_name = sensor_id_col_name # The col name after transofrming the column sensor_uid
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        if columns_to_use is None:
             
            # Define features dynamically based on existing columns (if features are not selected based on correlation)
            self.columns_to_use = [self.sensor_id_col_name, 'datetime', 'incremental_id', 'value', 'longitude', 'latitude',
                    'Storm_relative_helicity_height_above_ground_layer',
                    'U-Component_Storm_Motion_height_above_ground_layer',
                    'Wind_speed_gust_surface', 'u-component_of_wind_maximum_wind',
                    'v-component_of_wind_height_above_ground'] 
        else:
             self.columns_to_use = columns_to_use

    def load_data(self, nrows=None,sort_by_datetime=True):
        """
        Loads and preprocesses raw data, converting 'date' column to datetime and sorting by it.

        Parameters:
        - nrows (int, optional): Number of rows to load from the CSV file. If None, loads the entire file.
        """
        self.df = pd.read_csv(self.file_path, names=self.csv_column_names, nrows=nrows)
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        if sort_by_datetime:
            self.df = self.df.sort_values('datetime').reset_index(drop=True)

    def preprocess_data(self,select_relevant_cols = False):
        """Run the data preprocessing pipeline to clean and prepare the data."""
        self._map_sensor_ids()
        self._discard_misformatted_values()
        if select_relevant_cols:
            self._select_relevant_columns()
        else:
             self._select_fixed_columns()

    def _map_sensor_ids(self):
        """Map unique sensor IDs to integers for categorical feature conversion."""
        self.df[self.sensor_id_col_name] = self.df['sensor_id'].map({s: i for i, s in enumerate(set(self.df['sensor_id']))})

    def _discard_misformatted_values(self):
        """Remove rows with misformatted values and convert 'value' column to float."""
        mask = self.df['value'].apply(lambda row: ' ' in str(row))
        self.df = self.df[~mask]
        self.df.value = self.df.value.astype(float)

    def _select_relevant_columns(self,method = None):
            """Filter out columns with low correlation to the target variable 'value'."""
            accepted_corr_method = ['pearson','spearman','kendall']
            df_dropped = self.df.drop(['sensor_id', 'date'], axis=1).dropna()
            df_dropped_clean = df_dropped.dropna()
            if method is None:
                correlations = df_dropped_clean.corr()['value'].abs()
            else:
                 assert method in accepted_corr_method, f'The correlation method {method} (input variable) must be one of the following: {accepted_corr_method}.'
                 correlations = df_dropped_clean.corr(method)['value'].abs()

            relevant_columns = correlations[correlations >= self.correlation_threshold].index
            print(f'Selected relevant columns based on {method} correlation are now : {relevant_columns}')
            new_columns_to_use = [self.sensor_id_col_name, 'date'] + list(relevant_columns)
            self.columns_to_use = new_columns_to_use
            #self.df = self.df[['sensor_id', 'date'] + list(relevant_columns)].reset_index(drop=True)


    def _select_fixed_columns(self):
            """Select fixed columns based on the original notebook provided."""
            self.df = self.df[self.columns_to_use]

    def add_temporal_lags(self):
        """Add temporal lag features, creating 'lag1', 'lag2', ..., 'lagN' features for model input."""
        for i in range(self.lags):
            lag_col_name = f'lag{i+1}'
            self.df[lag_col_name] = self.df.groupby('sensor_uid')['value'].shift(i+1) - self.df['value']
            self.columns_to_use += [lag_col_name]

    def convert_datetime(self):
        """Convert datetime to UNIX timestamp for uniform modeling input."""
        self.df['datetime'] = self.df['datetime'].astype(int) / 10**9

    def create_target_variable(self, horizon=15):
        """Generate target variable as speed delta based on specified horizon."""
        self.df['target'] = self.df.groupby('sensor_uid')['value'].shift(-horizon) - self.df['value']
        self.df = self.df.dropna()  # Remove rows with missing target values

    def split_data(self):
        """Split the data into training and testing sets based on defined features."""
        X = self.df[self.columns_to_use]
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False, random_state=self.random_state
        )

    def get_train_test_split(self,reset_index=True):
        """Return train and test data splits."""
        if reset_index:
            for dataset in [self.X_train,self.X_test,self.y_train,self.y_test]:
                dataset.reset_index(drop=True,inplace=True)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def prepare_data(self,select_relevant_cols = False, horizon=15):
        """Run the full data preparation pipeline and return train/test splits for modeling."""
        self.load_data()
        self.preprocess_data(select_relevant_cols = select_relevant_cols)
        self.add_temporal_lags()
        self.convert_datetime()
        self.create_target_variable(horizon=horizon)
        self.split_data()

        return self.get_train_test_split()

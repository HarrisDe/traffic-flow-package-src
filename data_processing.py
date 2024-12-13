import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from .constants import colnames
import random
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm
from .helper_functions import get_adjacent_sensors_dict
import pickle

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # You can set this to DEBUG, WARNING, etc. as needed
)

class TrafficFlowDataProcessing:
    
    """
    A class to process and prepare traffic flow data for time-series prediction.
    This includes methods for loading, cleaning, feature engineering, and data splitting.
    """

    def __init__(self, data_path = '../data',file_name='estimated_average_speed_selected_timestamps-edited-new.csv', adj_sensors_file_name = None,
                 column_names=None, lags=20,spatial_lags=20,
                  correlation_threshold=0.01, columns_to_use=None, 
                  time_col_name = 'sensor_time_min', sensor_id_col_name='sensor_uid', random_state=69):
        """
        Initialize data processing parameters.

        Parameters:
        - file_path (str): Path to the CSV file with traffic data.
        - column_names (list): Column names for the data file. If None, defaults to colnames.
        - lags (int): Number of temporal lag features to generate.
        - spatial_lags (int): Number of spatial lag sensors to include.
        - correlation_threshold (float): Minimum correlation for feature selection.
        - random_state (int): Seed for reproducible train-test split.
        """
        self.data_path = data_path
        self.file_name = file_name
        self.file_path = os.path.join(self.data_path,self.file_name)
        self.csv_column_names = column_names if column_names else colnames  # Use default if None
        self.lags = lags
        self.spatial_lags = spatial_lags
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.sensor_id_col_name = sensor_id_col_name  # The col name after transforming the column sensor_uid
        self.time_col_name = time_col_name
        self.adj_sensors_file_name = adj_sensors_file_name
        if adj_sensors_file_name is not None:
            self.adj_sensors_file_path =  os.path.join(self.data_path,adj_sensors_file_name)
        self.previous_plotted_sensors = set()
        self.df = None
        self.df_orig = None
        self.test_size = None
        self.cache = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        if columns_to_use is None:
            # Define features dynamically based on existing columns (if features are not selected based on correlation)
            self.columns_to_use = [
                self.sensor_id_col_name, 'incremental_id', 'value', 'longitude', 'latitude',
                'Storm_relative_helicity_height_above_ground_layer',
                'U-Component_Storm_Motion_height_above_ground_layer',
                'Wind_speed_gust_surface', 'u-component_of_wind_maximum_wind',
                'v-component_of_wind_height_above_ground'
            ]
            self.columns_to_use.remove('incremental_id')
            
        else:
            self.columns_to_use = columns_to_use

    def load_data(self, nrows=None, sort_by_datetime=True):
        """Loads and preprocesses raw data, converting 'date' column to datetime and sorting by it."""

        # load df with pyarrow for faster loading, if nrows is specified, then you can't use pyarrow
        if nrows is None:
            self.df = pd.read_csv(self.file_path, names=self.csv_column_names)
        else:
            logging.info(f"selecting df with {nrows} nrows.")
            self.df = pd.read_csv(self.file_path, names=self.csv_column_names, nrows= nrows)
        self.df_orig = self.df.copy()
        rows_that_target_var_is_nan = self.df['value'].isna().sum()
        if rows_that_target_var_is_nan > 0:
            warnings.warn(f'Target variable (column "value") is Nan {rows_that_target_var_is_nan} times.')
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        if sort_by_datetime:
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            logging.info(f"df got sorted by datetime.")
    
    def load_data_parquet(self, nrows=None):
        """Reads the parquet file and selects nr of rows (it's already sorted by datetime)."""
        logging.info(f"Reading parquet file: {self.file_path}")
        self.df = pd.read_parquet(self.file_path,engine='pyarrow')
        
        # Ensure that date and datetime columns are present and converted to datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['datetime'] = pd.to_datetime(self.df['date'])
        elif 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df['date'] = self.df['datetime']

        else:
            raise ValueError("The 'date' or 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")

        logging.info(f"df['date'] converted to datetime.")

        if nrows is not None:
            nrows = int(nrows)
            self.df = self.df.iloc[:nrows]
        rows_that_target_var_is_nan = self.df['value'].isna().sum()
        if rows_that_target_var_is_nan > 0:
            warnings.warn(f'Target variable (column "value") is Nan {rows_that_target_var_is_nan} times.')

        if nrows is not None:
            nrows = int(nrows)
            self.df = self.df.iloc[:nrows]

    def preprocess_data(self, select_relevant_cols=False):
        """Run the data preprocessing pipeline to clean and prepare the data."""
        logging.info("Starting preprocessing of data.")
        self._map_sensor_ids()
        self._discard_sensor_uids_w_uncommon_nrows()
        self._discard_misformatted_values()
        if select_relevant_cols:
            self._select_relevant_columns()
        else:
            self._select_fixed_columns()
        logging.info("Preprocessing completed.")

    def _map_sensor_ids(self):
        """Map unique sensor IDs to integers for categorical feature conversion."""
        logging.info("Mapping sensor IDs to integers.")

        nr_unique_sensors = self.df['sensor_id'].nunique()

        self.df[self.sensor_id_col_name] = self.df['sensor_id'].map({s: i for i, s in enumerate(set(self.df['sensor_id']))})
        # Log the number of sensors enumerated
        logging.info(f"Enumerated {nr_unique_sensors} sensors, starting from 1.")

    def _discard_misformatted_values(self):
        """Remove rows with misformatted values and convert 'value' column to float."""
        logging.info("Discarding misformatted values in 'value' column.")

        # Count the number of rows before filtering
        initial_row_count = len(self.df)
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
        self.df = self.df.dropna(subset=['value'])
        # Count the number of rows after filtering
        final_row_count = len(self.df)
    
        # Log the number of rows dropped
        rows_dropped = initial_row_count - final_row_count
        logging.info(f"Discarded {rows_dropped} rows with misformatted values in 'value' column.")

    def align_sensors_to_common_timeframe(self):
        """
        Align all sensors to the same timeframe based on the sensor with the fewest recordings.
        This ensures all sensors have data for the same timestamps.
        This method should be used if only a subset of the dataframe is to be read.
        """
        logging.info("Aligning sensors to a common timeframe based on the sensor with the fewest recordings (becase a subset of ).")
        
        # Count the number of recordings for each sensor
        sensor_counts = self.df.groupby('sensor_id').size()
        
        # Identify the sensor with the fewest recordings
        min_recording_sensor = sensor_counts.idxmin()
        min_recording_count = sensor_counts.min()
        
        # Get the timeframe for the sensor with the fewest recordings
        common_timeframe = self.df[self.df['sensor_id'] == min_recording_sensor]['datetime']
        min_time = common_timeframe.min()
        max_time = common_timeframe.max()
        
        # Filter the DataFrame to include only data within the common timeframe for all sensors
        original_row_count = len(self.df)
        self.df = self.df[(self.df['datetime'] >= min_time) & (self.df['datetime'] <= max_time)]
        filtered_row_count = len(self.df)
        
        logging.info(f"Aligned all sensors to the common timeframe: {min_time} to {max_time}.")
        logging.info(f"Rows before alignment: {original_row_count}, Rows after alignment: {filtered_row_count}.")
    

    def _select_relevant_columns(self, method=None):
        """Filter out columns with low correlation to the target variable 'value'."""
        logging.info(f"Selecting relevant columns using correlation method: {method or 'default'}.")
        accepted_corr_method = ['pearson', 'spearman', 'kendall']
        df_dropped = self.df.drop(['sensor_id', 'date'], axis=1).dropna()
        df_dropped_clean = df_dropped.dropna()
        if method is None:
            correlations = df_dropped_clean.corr()['value'].abs()
        else:
            assert method in accepted_corr_method, f'The correlation method {method} (input variable) must be one of the following: {accepted_corr_method}.'
            correlations = df_dropped_clean.corr(method)['value'].abs()

        relevant_columns = correlations[correlations >= self.correlation_threshold].index
        logging.info(f'Selected relevant columns based on {method} correlation are now : {relevant_columns}')
        new_columns_to_use = [self.sensor_id_col_name, 'date'] + list(relevant_columns)
        self.columns_to_use = new_columns_to_use

    def _select_fixed_columns(self):
        """Select fixed columns based on the original notebook provided."""
        logging.info("Selecting fixed columns for modeling.")
  

    def add_temporal_lags(self,fill_nans_value = -1):
        """Add temporal lag features, creating 'lag1', 'lag2', ..., 'lagN' features for model input."""
        logging.info("Adding temporal lags.")
        for i in range(self.lags):
            lag_col_name = f'lag{i+1}'
            #self.df_for_ML[lag_col_name] = self.df_for_ML.groupby(self.sensor_id_col_name)['value'].shift(i+1) - self.df['value']
            self.df[lag_col_name] = self.df.groupby(self.sensor_id_col_name)['value'].shift(i+1) - self.df['value']
            self.df[lag_col_name].fillna(fill_nans_value, inplace=True)
            self.columns_to_use += [lag_col_name]



    def convert_datetime(self):
        """Extract hour, day, and month from the datetime column and create new columns."""
        logging.info("Extracting hour, day, and month from the datetime column.")
        if 'datetime' not in self.df.columns:
            raise ValueError("The 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")
    
        # Create new columns
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day'] = self.df['datetime'].dt.dayofweek
        #self.df['month'] = self.df['datetime'].dt.month

        # Add new columns to `columns_to_use`
        #self.columns_to_use += ['hour', 'day', 'month']
        self.columns_to_use += ['hour', 'day']

    def create_target_variable(self, horizon=15):
        """Generate target variable as speed delta based on specified horizon."""
        logging.info("Creating target variable.")
        self.df_for_ML = self.df[self.columns_to_use]
        if 'datetime' in self.df_for_ML.columns:
            self.df_for_ML.drop(columns=['datetime'],inplace=True)
        self.df_for_ML['target'] = self.df_for_ML.groupby(self.sensor_id_col_name)['value'].shift(-horizon) - self.df_for_ML['value']
        self.df_for_ML = self.df_for_ML.dropna(subset='target')  # Remove rows with missing target values (the last ones (from the horizon value) for each sensor)

    def prepare_data(self, add_spatial_lags=False,select_relevant_cols=False, horizon=15,nrows=None,sort_by_datetime=True,use_weekend_var=True):
        """Run the full data preparation pipeline without splitting."""
        logging.info('Preparing the dataset')
        #self.load_data(nrows=nrows,sort_by_datetime=sort_by_datetime)
        self.load_data_parquet(nrows=nrows)
        print('Data loaded as parquet')
        #self.add_adjusent_sensors()
        self.preprocess_data(select_relevant_cols=select_relevant_cols)
        self.add_temporal_lags()
        if add_spatial_lags:

            #self.add_spatial_lags_from_cache_dict()
            self.add_spatial_lags_columns()
        self.convert_datetime()
        if use_weekend_var:
            self.add_weekend_columns()
        self.create_target_variable(horizon=horizon)

    def split_data(self,test_size):
        """Split the data into training and testing sets based on defined features."""
        logging.info('Splitting the dataset.')
        logging.info(f"columns to use are: {self.columns_to_use}, in split_data method.")
        X = self.df_for_ML[self.columns_to_use]
        y = self.df_for_ML['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=self.random_state
        )

    def add_time_column(self,sort_by_datetime=False):
        """Adds time column for the sensors."""
        self.df = self.df.sort_values(['sensor_id', 'datetime']).reset_index(drop=True)
    
        # Add the 'time' column as the difference in minutes (dt)
        self.df[self.time_col_name] = self.df.groupby('sensor_id')['datetime'].diff().dt.total_seconds() / 60
        self.df[self.time_col_name].fillna(0,inplace=True) # ensure the first value of each sensor is 0
        self.df[self.time_col_name] = self.df.groupby('sensor_id')[self.time_col_name].cumsum() 
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

    def add_weekend_columns(self):
        """Add columns indicating whether the date is a Saturday or Sunday."""
        logging.info("Adding weekend columns for Saturday and Sunday.")
        self.df['is_saturday'] = self.df['datetime'].dt.dayofweek == 5  # Saturday
        self.df['is_sunday'] = self.df['datetime'].dt.dayofweek == 6  # Sunday
        self.df['is_saturday'] = self.df['is_saturday'].astype(int)
        self.df['is_sunday'] = self.df['is_sunday'].astype(int)
        self.columns_to_use += ['is_saturday']
        self.columns_to_use += ['is_sunday']
        #logging.info(f"columns to use are: {self.columns_to_use}, after adding weekend")

    def add_hour_column(self):
        """Add a column with the hour of the day extracted from the datetime column."""
        logging.info("Adding hour column to the DataFrame.")
        if 'datetime' not in self.df.columns:
            raise ValueError("The 'datetime' column is missing. Ensure the data is preprocessed with datetime conversion.")
        
        self.df['hour'] = self.df['datetime'].dt.hour
        self.columns_to_use += ['hour']

    def _discard_sensor_uids_w_uncommon_nrows(self):
        """Remove rows where the sensor_uid group size is not equal to the most common size."""
        logging.info('Discarding sensor ids with uncommon nr of rows.')
        # Calculate the group sizes
        group_sizes = self.df.groupby(self.sensor_id_col_name).size()

        # Find the most common group size
        most_common_size = group_sizes.mode().iloc[0]

        # Identify sensor_uids with the most common size
        valid_sensors = group_sizes[group_sizes == most_common_size].index

            # Count sensors
        total_sensors = len(group_sizes)
        valid_sensor_count = len(valid_sensors)
        discarded_sensor_count = total_sensors - valid_sensor_count

        # Print details
        logging.info(f"Most common group size (mode): {most_common_size}")
        logging.info(f"Number of sensors with this size: {valid_sensor_count}")
        logging.info(f"Number of sensors discarded: {discarded_sensor_count}")

        # Filter the DataFrame to include only rows with valid sensor_uids
        initial_row_count = len(self.df)
        self.df = self.df[self.df[self.sensor_id_col_name].isin(valid_sensors)]
        final_row_count = len(self.df)


        logging.info(f"Discarded {initial_row_count - final_row_count} rows with uncommon sensor_uid group sizes.")
    
    def get_train_test_split(self,reset_index=True):
        """Return train and test data splits."""
        if reset_index:
            for dataset in [self.X_train,self.X_test,self.y_train,self.y_test]:
                dataset.reset_index(drop=True,inplace=True)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_clean_train_test_split(self, test_size = 0.5,nrows=None,select_relevant_cols = False, horizon=15,add_train_test_flag=False,
                                   use_weekend_var=True,add_spatial_lags = True,reset_index=True,print_nans= True):
        """Split the data into train and test sets and optionally add a 'train_set' flag."""
        
        self.test_size = test_size
        self.prepare_data(nrows=nrows,select_relevant_cols=select_relevant_cols,horizon=horizon,use_weekend_var=use_weekend_var,add_spatial_lags=add_spatial_lags)
        self.split_data(test_size=test_size)

        if add_train_test_flag:
            # Mark rows belonging to the train set
            self.df['train_set'] = False
            self.df.loc[self.X_train.index, 'train_set'] = True
        
        if print_nans:
            logging.info(f"number of nans in X_train: {self.X_train.isna().sum().sum()}")
            logging.info(f"number of nans in X_test: {self.X_test.isna().sum().sum()}")
            logging.info(f"number of nans in y_train: {self.y_train.isna().sum().sum()}")
            logging.info(f"number of nans in y_test: {self.y_test.isna().sum().sum()}")

        return self.get_train_test_split(reset_index=reset_index)
    
    def plot_sensor_train_test_split(self, test_size=0.5):
        """
        Randomly sample a sensor and plot its train-test split.
        Ensures a previously plotted sensor is not plotted again consecutively.
        """
        if self.df is None:
            raise ValueError("Dataframe is empty. Please load and preprocess the data first.")
        if self.test_size is None:
            raise ValueError("No test size specified. Please load and preprocess the data first.")

        # Add time column 
        self.add_time_column(sort_by_datetime=False)

        # Get unique sensors and ensure no duplicate consecutive plotting
        unique_sensors = set(self.df[self.sensor_id_col_name].unique())
        available_sensors = unique_sensors - self.previous_plotted_sensors
        if not available_sensors:
            self.previous_plotted_sensors.clear()  # Reset once all sensors are plotted
            available_sensors = unique_sensors

        # Randomly sample a sensor
        sensor_to_plot = random.choice(list(available_sensors))
        self.previous_plotted_sensors.add(sensor_to_plot)

        # Filter data for the selected sensor
        sensor_data = self.df[self.df[self.sensor_id_col_name] == sensor_to_plot].copy()
        sensor_to_plot_name = sensor_to_plot

        # Assign train/test flags chronologically
        sensor_data = sensor_data.sort_values(self.time_col_name)
        cutoff = int(len(sensor_data) * (1 - test_size))
        sensor_data['train_set'] = False
        sensor_data.iloc[:cutoff, sensor_data.columns.get_loc('train_set')] = True

        # Plot the train-test split
        plt.figure(figsize=(12, 6))
        plt.scatter(
            sensor_data[self.time_col_name], sensor_data['value'],
            c=sensor_data['train_set'].map({True: 'blue', False: 'red'}),
            alpha=0.6
        )
        plt.title(f"Train-Test Split Visualization for {sensor_to_plot_name}")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Speed [kph]")
        plt.grid()
        plt.show()

    def read_or_generate_cache_dict_for_spatial_lags(self):
        """
        Creates ore reads an existing cache file.
        This file has keys ('sensor_id', 'datetime') and value 'value'
        Is used to efficiently find the spatial lags for every row.
        """
        
        file_name = os.path.basename(self.file_path)
        file_name = os.path.splitext(file_name)[0]  # Remove the extension (splitext gives a list of the file name and the extension [eg .csv])
        cache_name = file_name + '_cache.pkl'
        cache_loc = os.path.join(self.data_path,cache_name)
        try:
            logging.info(f"Trying to load {cache_loc} file.")
            with open(cache_loc,'rb') as f:
                self.cache = pickle.load(f)
                logging.info(f"cache file {cache_loc} has been loaded from data folder.")
        
        except FileNotFoundError:
            logging.info(f"cache file {cache_loc} hasn't been found in data folder. Creating & saving a cache file...")
            self.cache = self.df.set_index(['sensor_id', 'datetime'])['value'].to_dict()
            with open(cache_loc, 'wb') as f:
                pickle.dump(self.cache, f)
            logging.info(f"Cache file has been generated and saved in: {cache_loc}")

    def add_spatial_lags_columns(self):
        col_spl = [col for col in self.df.columns if 'spl' in col]
        self.columns_to_use +=col_spl

    def add_spatial_lags_from_cache_dict(self):
        """
        Adds the required spatial lags.
        """
        logging.info("Adding spatial lags using a preprocessed dictionary for lookups.")
        
        # Create a dictionary with (sensor_id, datetime) as keys for fast lookups
        self.read_or_generate_cache_dict_for_spatial_lags()

        previous_sensors_dict = get_adjacent_sensors_dict(
            adj_sensors_csv_loc=self.adj_sensors_file_path,
            nr_of_adj_sensors=self.spatial_lags,
            delete_duplicate_rows=True
        )

        def get_lagged_sensor_value(datetime, sensor_to_find_spatial_lagged_sensor, spatial_lag_nr,previous_sensors_dict=previous_sensors_dict):

            # Find the name of the lagged sensor (based on the ith lag [spatial_lag_nr])
            lagged_sensor = previous_sensors_dict[sensor_to_find_spatial_lagged_sensor]['previous_sensors'][spatial_lag_nr]
            # Use the name of the lagged sensor and the datetime (of the row) to find the value of the lagged sensor
            lagged_sensor_value = self.cache.get((lagged_sensor, datetime), -1)


        tqdm.pandas()
        for i in range(self.spatial_lags):

            # self.df[f"spatial_lag_{i+1}"] = self.df.progress_apply(
            #     lambda row: get_lag(
            #         row['datetime'],
            #         row['sensor_id'],
            #         previous_sensors_dict.get(row['sensor_id'], {}).get('previous_sensors', [None])[i]
            #     ),
            #     axis=1
            # )
            self.df[f"spatial_lag_{i+1}"] = self.df.progress_apply(
                lambda row: get_lagged_sensor_value(row['datetime'],row['sensor_id'],i),
            axis=1)



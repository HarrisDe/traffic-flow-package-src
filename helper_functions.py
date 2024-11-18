from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def normalize_data(X_train, X_test, use_minmax_norm=False, use_full_data=False):
    """
    Normalizes training and testing data using StandardScaler or MinMaxScaler.
    
    Parameters:
    - X_train (array-like): Training dataset.
    - X_test (array-like): Testing dataset.
    - use_minmax_norm (bool): If True, uses MinMaxScaler; otherwise, uses StandardScaler.
    - use_full_data (bool): If True, normalizes using both training and testing data combined.
        WARNING: Using this option introduces data leakage because the test data influences 
        the scaling applied to the training data. This approach may lead to overly optimistic 
        performance metrics and is not recommended for real-world scenarios where the test 
        set must remain unseen until final evaluation.
    
    Returns:
    - X_train_normalized (array-like): Normalized training dataset.
    - X_test_normalized (array-like): Normalized testing dataset.

    Notes:
    - Default behavior (`use_full_data=False`) ensures that scaling is based solely on the training data, 
      which is a best practice to avoid data leakage.
    - Use `use_full_data=True` only in controlled experiments where you need consistent scaling across 
      both training and test sets and are aware of the potential risks.

    Example Usage:
    - Normalize with StandardScaler (default):
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test)
    - Normalize with MinMaxScaler:
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test, use_minmax_norm=True)
    - Normalize using both training and test data:
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test, use_full_data=True)
    - Normalize with MinMaxScaler using both datasets:
        X_train_normalized, X_test_normalized = normalize_data(X_train, X_test, use_minmax_norm=True, use_full_data=True)
    """
    scaler = MinMaxScaler() if use_minmax_norm else StandardScaler()
    
    if use_full_data:
        # Concatenate training and test data for joint scaling
        full_data = np.concatenate([X_train, X_test], axis=0)
        full_data_normalized = scaler.fit_transform(full_data)
        # Split back into training and test sets
        X_train_normalized = full_data_normalized[:X_train.shape[0], :]
        X_test_normalized = full_data_normalized[X_train.shape[0]:, :]
    else:
        # Scale based only on training data
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized

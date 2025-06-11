#  🚦 Traffic Flow Forecasting Package
> ⚠️ **Note**: This repository is currently under active development. Expect frequent updates and changes.

This repository provides a modular Python package for traffic state forecasting using time-series sensor data. Developed as part of the EMERALDS EU Horizon project, the package supports data ingestion, feature engineering, model training, and post-processing.

It is designed to forecast short-term traffic speed across multiple sensors, with compatibility for traditional ML models and integration with advanced spatio-temporal models such as GMAN.

---

## Core Features

- **Flexible Data Pipeline**: Preprocessing, normalization (if needed), and transformation of large-scale traffic datasets.
- **Feature Engineering**: Includes spatial adjacency, lagged variables and hybrid model integration (e.g. GMAN predictions).
- **Modeling**: Supports classical ML models like XGBoost, with extensible hooks for deep learning integration.
- **Evaluation**: Includes detailed error analysis (MAE, RMSE, MAPE, SMAPE, etc.) and naive baselines.
- **Post-Processing**: (Residual) correction, and interactive plotting for analysis and presentation.

---

## Repository Structure

traffic_flow_package_src/
│
├── constants/              # Project-wide constant values (e.g., column names)
├── data_loading/           # Data loading and preprocessing logic
├── features/               # Feature engineering methods
├── modeling/               # Model training and tuning
├── evaluation/             # Model evaluation and metric reporting
├── pipeline/               # End-to-end orchestration pipeline
├── post_processing/        # Plotting utilities, post-processing of predictions
├── utils/                  # Helper functions
├── README.md               # This file



##  Example Use

To run a full modeling pipeline:

file_path = 'data/example_file.parquet'

# Create train/test splits
tdp = TrafficDataPipelineOrchestrator(orig_file_path)
X_train, X_test, y_train, y_test = tdp.run_pipeline()

# HP-tune to find best XGB model
mt = ModelTunerXGB(X_train, X_test, y_train, y_test)
    model_path, best_params, training_time, total_time = mt.tune_xgboost(
        objective="reg:pseudohubererror",
        use_gpu=True)

# Evaluate model
me = ModelEvaluator(
        X_test,
        y_test,
        y_train,
        df_for_ML=tdp.df,  # full df containing train and test sets, and respective features and target
    )

# Get performance metrics 
results = me.evaluate_model_from_path(model_path) # results is a dictionary
metrics = results['metrics']
metrics_std = results['metrics_std']
naive_metrics = results['naive_metrics']
naive_metrics_std = results['naive_metrics_std']

##  Deployment
Parts of this repository are intended to be deployed on the EMERALDS platform using [KServe](https://github.com/kserve/kserve).  
Due to confidentiality constraints, it is currently uncertain whether this deployment setup can be made publicly available or forked from this repository.

## License

This project is part of the EMERALDS EU Horizon initiative.

**Note**: The original datasets used in this repository (e.g. NDW data) are confidential and cannot be publicly shared.

**Demo Use**: To help users test the pipeline, we may provide small-scale **toy datasets** in the future that simulate the structure of the real data. These will be hosted either within the repository or as separate downloadable files.

If you're interested in toy datasets or have specific use cases, feel free to open an issue or contact the maintainer.



👤 Author

Harris Deralas
📧 [harideralas@gmail.com](mailto:harideralas@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/harris-deralas)
Freelance Machine Learning Engineer
Currently affiliated with Data Science Lab, University of Piraeus


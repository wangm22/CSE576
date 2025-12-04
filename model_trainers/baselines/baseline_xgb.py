#!/usr/bin/env python3

# Standard
import os
import argparse
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score
# from tsfm_public.toolkit.dataset import ForecastDFDataset
# from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
# from tsfm_public.toolkit.util import select_by_index

def within_delta_accuracy(y_true, y_pred, delta=0.1):
    """
    Computes the proportion of predictions within `delta` of the true value.
    
    Parameters:
    - y_true: np.array, shape (n_samples,) or (n_samples, n_features)
    - y_pred: np.array, same shape as y_true
    - delta: float, the allowed deviation
    
    Returns:
    - accuracy: float, fraction of predictions within delta
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Absolute difference
    diff = np.abs(y_true - y_pred)
    
    # Count predictions within delta
    within = diff <= delta
    
    # Compute proportion
    accuracy = np.mean(within)
    return accuracy

if __name__ == "__main__":
   if sys.version_info[:2] == (3, 12):
      print(
         f"ERROR: Python 3.12 detected ({sys.version_info.major}.{sys.version_info.minor}). "
         "Please use Python 3.11 instead."
      )
      sys.exit(1)


   parser = argparse.ArgumentParser()

   ###REMOVE BEFORE SUBMITTING
   parser.add_argument("--n_estimators", default=1000, type=int)
   parser.add_argument("--lr", default=.05, type=float)
   parser.add_argument("--max_depth", default=6, type=int)
   parser.add_argument("--subsample", default=.8, type=float)
   parser.add_argument("--colsample_bytree", default=.8, type=float)
   parser.add_argument("--alpha", default=.5, type=float)
   parser.add_argument("--lmbda", default=1, type=float)
   parser.add_argument("--early_stopping", default=20, type=int)
   ###REMOVE BEFORE SUBMITTING

   # parser.add_argument("--n_estimators", required=True, type=int)
   # parser.add_argument("--lr", required=True, type=float)
   # parser.add_argument("--max_depth", required=True, type=int)
   # parser.add_argument("--subsample", required=True, type=float)
   # parser.add_argument("--colsample_bytree", required=True, type=float)
   # parser.add_argument("--alpha", required=True, type=float)
   # parser.add_argument("--lmbda", required=True, type=float)
   # parser.add_argument("--early_stopping", required=True, type=int)

   args = parser.parse_args()
   n_estimators = args.n_estimators
   lr = args.lr
   max_depth = args.max_depth
   subsample = args.subsample
   colsample_bytree = args.colsample_bytree
   alpha = args.alpha
   lmbda = args.lmbda
   early_stopping = args.early_stopping

   output_dir = (
      f"saved_models/baseline_xgb/"
      f"n_{n_estimators}_lr_{lr}_depth_{max_depth}_sub_{subsample}_col_{colsample_bytree}"
      f"_alpha_{alpha}_lmbda_{lmbda}_early_{early_stopping}/checkpoints"
   )

   logging_dir = (
      f"logs/baseline_xgb/"
      f"n_{n_estimators}_lr_{lr}_depth_{max_depth}_sub_{subsample}_col_{colsample_bytree}"
      f"_alpha_{alpha}_lmbda_{lmbda}_early_{early_stopping}"
   )

   save_dir = (
      f"saved_models/baseline_xgb/"
      f"n_{n_estimators}_lr_{lr}_depth_{max_depth}_sub_{subsample}_col_{colsample_bytree}"
      f"_alpha_{alpha}_lmbda_{lmbda}_early_{early_stopping}.json"
   )

   dataset_path = "datasets/train_final.csv"
   timestamp_column = "date"
   id_columns = []

   data = pd.read_csv(
      dataset_path,
      parse_dates=[timestamp_column],
   )
   forecast_columns = list(data.columns[2:-1]) + [data.columns[0]]
   # print(f"Forecast columns: {forecast_columns}")

   len_train = len(data[data["split"] == 0])
   len_val = len(data[data["split"] == 2])
   len_test = len(data[data["split"] == 1])
   # ---

   train_start_index = 0
   train_end_index = len_train - 1

   valid_start_index = len_train
   valid_end_index = len_train + len_val - 1

   test_start_index = len_train + len_val
   test_end_index = len_train + len_val + len_test - 1

   X_train = data.loc[train_start_index : train_end_index, forecast_columns]
   y_train = data.loc[train_start_index + 1 : train_end_index + 1, "market_forward_excess_returns"]
   X_valid = data.loc[valid_start_index : valid_end_index, forecast_columns]
   y_valid = data.loc[valid_start_index + 1 : valid_end_index + 1, "market_forward_excess_returns"]
   X_test = data.loc[test_start_index : test_end_index - 1, forecast_columns]
   y_test = data.loc[test_start_index + 1 : , "market_forward_excess_returns"]
   # ---

   model = xgb.XGBRegressor(
         n_estimators=n_estimators,
         learning_rate=lr,
         max_depth=max_depth,
         subsample=subsample,
         colsample_bytree=colsample_bytree,
         reg_alpha=alpha,
         reg_lambda=lmbda,
         objective="reg:squarederror",
         eval_metric="rmse",
         early_stopping_rounds=early_stopping
   )

   # Convert datasets
   
   print("Training XGBoost model...")
   # Train with early stopping
   model.fit(
      X_train, y_train,
      eval_set=[(X_valid, y_valid)],
      verbose=False
   )

   y_pred = model.predict(X_test)
   rmse = root_mean_squared_error(y_test, y_pred)
   print(f"Test RMSE: {rmse:.4f}")
   r2 = r2_score(y_test, y_pred)
   print(f"Test R^2: {r2:.4f}")
   delta = 0.002
   acc_test = within_delta_accuracy(y_test, y_pred, delta=delta)
   print(f"Test accuracy (within 0.005): {acc_test:.4f}")

   # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
   # model.save_model(save_dir)
# Standard
import os
import argparse

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd

# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

parser = argparse.ArgumentParser()
parser.add_argument("--patch_length", default=14, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
args = parser.parse_args()

patch_length = args.patch_length
lr = args.lr

output_dir = f"saved_models/patchtst/patch_{patch_length}_{lr}/checkpoints"
logging_dir = f"logs/patchtst/patch_{patch_length}_{lr}"
save_dir = f"saved_models/patchtst/patch_{patch_length}_{lr}/finals"

dataset_path = "datasets/train_final.csv"
timestamp_column = "date"
id_columns = []

data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)
forecast_columns = list(data.columns[2:-1])

len_train = len(data[data["split"] == 0])
len_val = len(data[data["split"] == 2])
len_test = len(data[data["split"] == 1])

context_length = 128
forecast_horizon = 1
num_workers = 16
batch_size = 64

# ---

train_start_index = 0
train_end_index = len_train - 1

valid_start_index = len_train
valid_end_index = len_train + len_val - 1

test_start_index = len_train + len_val
test_end_index = len_train + len_val + len_test - 1

train_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=train_start_index,
    end_index=train_end_index,
)
valid_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=valid_start_index,
    end_index=valid_end_index,
)
test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=test_start_index,
    end_index=test_end_index,
)

time_series_preprocessor = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
time_series_preprocessor = time_series_preprocessor.train(train_data)

# ---

train_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(train_data),
    id_columns=id_columns,
    timestamp_column="date",
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
valid_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(valid_data),
    id_columns=id_columns,
    timestamp_column="date",
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
test_dataset = ForecastDFDataset(
    time_series_preprocessor.preprocess(test_data),
    id_columns=id_columns,
    timestamp_column="date",
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)

# ---

config = PatchTSTConfig(
    num_input_channels=len(forecast_columns),

    context_length=context_length,
    patch_length=patch_length,
    patch_stride=patch_length,
    prediction_length=forecast_horizon,

    random_mask_ratio=0.0,

    d_model=64,
    num_attention_heads=16,
    num_hidden_layers=3,
    ffn_dim=256,

    dropout=0.2,
    head_dropout=0.2,

    pooling_type=None,
    channel_attention=False,

    scaling="std",

    loss="mse",

    pre_norm=True,
    norm_type="layernorm",
)
model = PatchTSTForPrediction(config)

# ---

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,

    learning_rate=args.lr,
    num_train_epochs=30,

    do_eval=True,
    eval_strategy="epoch",

    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=num_workers,

    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    logging_dir=logging_dir,  # Make sure to specify a logging directory

    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss

    label_names=["future_values"],
    
    weight_decay=0.1,
    max_grad_norm=1.0,

    lr_scheduler_type="constant"
)

# Create the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
)

# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
    # compute_metrics=compute_metrics,
)

# pretrain
trainer.train()

# ---

os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
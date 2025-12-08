"""
Improved TFT Implementation with Enhanced Configuration

Key improvements:
1. Longer encoder length for better historical context
2. Proper input type classification (known vs unknown)
3. Learning rate finder for optimal LR
4. Better early stopping configuration
5. Enhanced feature engineering
6. Optimized hyperparameters based on TFT paper
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Enhanced Configuration
# ============================================================
OUTPUT_DIR = 'outputs/tft_improved'
DATA_PATH = 'datasets/train_2000.csv'
MAX_EPOCHS = 30  # Increased from 30
BATCH_SIZE = 128  # Increased from 64 for better gradient estimates
MAX_ENCODER_LENGTH = 128  # Increased to 128 for even more historical context
GRADIENT_CLIP_VAL = 1.0  # Increased from 0.1 for faster learning


def create_enhanced_features(df):
    """Create enhanced features with more lags and rolling statistics."""

    # Create multiple lags
    for lag in [1, 2, 3, 5, 7]:
        df[f'forward_returns_lag{lag}'] = df['forward_returns'].shift(lag)
        df[f'risk_free_rate_lag{lag}'] = df['risk_free_rate'].shift(lag)
        df[f'market_forward_excess_returns_lag{lag}'] = df['market_forward_excess_returns'].shift(lag)

    # Rolling statistics (7-period window)
    for col in ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
        df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
        df[f'{col}_rolling_std_7'] = df[col].rolling(window=7, min_periods=1).std()

    # Rolling statistics (14-period window)
    for col in ['forward_returns']:
        df[f'{col}_rolling_mean_14'] = df[col].rolling(window=14, min_periods=1).mean()
        df[f'{col}_rolling_std_14'] = df[col].rolling(window=14, min_periods=1).std()

    return df


def prepare_data_for_tft(train_path=DATA_PATH, max_encoder_length=MAX_ENCODER_LENGTH):
    """
    Prepare data with enhanced feature engineering.
    """
    print("=" * 80)
    print("LOADING AND PREPARING DATA FOR IMPROVED TFT")
    print("=" * 80)

    # Load data
    df = pd.read_csv(train_path)
    print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns")

    # Handle missing values
    df = df.ffill().bfill()

    # Original feature subset from paper
    feature_subset = ['V2', 'M14', 'M18', 'V11', 'E17', 'V4', 'V3', 'E16', 'E2', 'V1', 'M16']

    # Create enhanced features
    df = create_enhanced_features(df)

    # Drop rows with NaN (from shifting and rolling)
    df = df.dropna()

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Define feature groups
    # Time-varying unknown: Features we don't know in the future
    time_varying_unknown = feature_subset + [
        'forward_returns_lag1', 'forward_returns_lag2', 'forward_returns_lag3',
        'forward_returns_lag5', 'forward_returns_lag7',
        'forward_returns_rolling_mean_7', 'forward_returns_rolling_std_7',
        'forward_returns_rolling_mean_14', 'forward_returns_rolling_std_14',
    ]

    # Time-varying known: Features we know in advance (risk-free rate, market returns)
    time_varying_known = [
        'risk_free_rate_lag1', 'risk_free_rate_lag2', 'risk_free_rate_lag3',
        'risk_free_rate_lag5', 'risk_free_rate_lag7',
        'risk_free_rate_rolling_mean_7', 'risk_free_rate_rolling_std_7',
        'market_forward_excess_returns_lag1', 'market_forward_excess_returns_lag2',
        'market_forward_excess_returns_lag3', 'market_forward_excess_returns_lag5',
        'market_forward_excess_returns_lag7',
        'market_forward_excess_returns_rolling_mean_7',
        'market_forward_excess_returns_rolling_std_7',
    ]

    all_features = time_varying_unknown + time_varying_known

    # Create required columns for TFT
    df = df.reset_index(drop=True)
    df['time_idx'] = np.arange(len(df))
    df['series_id'] = 'stock'
    df['target'] = df['forward_returns'].astype(float)

    # Verify all features are numeric and have no NaN/inf
    for feat in all_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')

    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    # Drop any rows with NaN in features or target
    df = df.dropna(subset=all_features + ['target'])

    # Remove infinite values
    for feat in all_features + ['target']:
        df = df[~np.isinf(df[feat])]

    # Reset time_idx after cleaning
    df = df.reset_index(drop=True)
    df['time_idx'] = np.arange(len(df))

    # Use split column from dataset
    train_df = df[df['split'] == 0].copy()
    test_df = df[df['split'] == 1].copy()
    val_df = df[df['split'] == 2].copy()

    print(f"\nSplit distribution:")
    print(f"  Train (split=0): {len(train_df)} samples")
    print(f"  Test (split=1):  {len(test_df)} samples")
    print(f"  Val (split=2):   {len(val_df)} samples")

    print(f"\nTotal features: {len(all_features)}")
    print(f"  Time-varying unknown: {len(time_varying_unknown)}")
    print(f"  Time-varying known: {len(time_varying_known)}")

    return df, train_df, test_df, val_df, time_varying_unknown, time_varying_known


def create_tft_dataset(df, time_varying_unknown, time_varying_known, max_encoder_length=MAX_ENCODER_LENGTH):
    """Create TimeSeriesDataSet with proper input classification."""

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["series_id"],
        min_encoder_length=max_encoder_length // 2,  # Allow flexibility
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=time_varying_unknown + ["target"],
        time_varying_known_reals=time_varying_known,  # Separated known inputs
        static_categoricals=["series_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    return dataset


def train_improved_tft(df, train_df, val_df, time_varying_unknown, time_varying_known,
                       max_epochs=MAX_EPOCHS, find_learning_rate=True):
    """Train improved TFT model with optimized hyperparameters."""

    print("\n" + "=" * 80)
    print("CREATING IMPROVED TFT MODEL")
    print("=" * 80)

    max_encoder_length = MAX_ENCODER_LENGTH

    # Create training dataset
    training = create_tft_dataset(train_df, time_varying_unknown, time_varying_known, max_encoder_length)

    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df['split'].isin([0, 2])],
        predict=True,
        stop_randomization=True
    )

    # Dataloaders
    batch_size = BATCH_SIZE
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    print(f"\nDataset created successfully")
    print(f"  Encoder length: {max_encoder_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")

    # Create model with improved hyperparameters
    # Based on TFT paper recommendations for financial data
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,  # Will be tuned with LR finder
        hidden_size=128,  # Increased from 64
        attention_head_size=4,
        dropout=0.2,  # Increased from 0.1 for better regularization
        hidden_continuous_size=32,  # Increased from 16
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=8,  # Increased from 4
    )

    print(f"\nModel size: {tft.size()/1e3:.1f}k parameters")

    # Training setup
    print("\n" + "=" * 80)
    print("TRAINING IMPROVED TFT")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Improved early stopping - less aggressive
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,  # More sensitive
        patience=20,  # Increased from 10
        verbose=True,
        mode="min"
    )

    # Model checkpoint - save best model
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=OUTPUT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min"
    )

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, "lightning_logs"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[lr_logger, early_stop, checkpoint],
        logger=logger,
        enable_progress_bar=True,
    )

    # Learning rate finder (optional but recommended)
    if find_learning_rate:
        print("\n" + "=" * 80)
        print("FINDING OPTIMAL LEARNING RATE")
        print("=" * 80)

        try:
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                min_lr=1e-5,
                max_lr=1e-1,
                num_training=200,
            )

            # Get suggestion
            new_lr = lr_finder.suggestion()
            print(f"\nSuggested learning rate: {new_lr:.6f}")

            # Plot
            fig = lr_finder.plot(suggest=True)
            fig.savefig(os.path.join(OUTPUT_DIR, "lr_finder.png"))
            plt.close()

            # Update model learning rate
            tft.learning_rate = new_lr
            print(f"Updated model learning rate to: {new_lr:.6f}")

        except Exception as e:
            print(f"LR finder failed: {e}")
            print("Continuing with default learning rate...")

    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model saved at epoch: {checkpoint.best_model_score}")

    return tft, trainer, training


def evaluate_tft(tft, df, test_df, time_varying_unknown, time_varying_known, training):
    """Evaluate improved TFT on test set."""

    print("\n" + "=" * 80)
    print("EVALUATING IMPROVED TFT ON TEST SET")
    print("=" * 80)

    max_encoder_length = MAX_ENCODER_LENGTH

    # Test dataset
    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=False,
        stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Total samples in dataset: {len(test_dataset)}")
    print(f"Test set size (split=1): {len(test_df)}")

    # Get predictions
    predictions_list = []
    actuals_list = []
    time_idx_list = []

    tft.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_dataloader):
            output = tft(batch_x)
            pred_tensor = output['prediction'][:, 0, 3]  # Median quantile

            predictions_list.append(pred_tensor.cpu().numpy())
            actuals_list.append(batch_y[0][:, 0].cpu().numpy())

            # Extract time_idx
            if 'encoder_time_idx_start' in batch_x:
                time_idx = (batch_x['encoder_time_idx_start'] + max_encoder_length).cpu().numpy()
            elif 'decoder_time_idx' in batch_x:
                time_idx = batch_x['decoder_time_idx'][:, 0].cpu().numpy()
            else:
                batch_size = pred_tensor.shape[0]
                time_idx = np.arange(i * BATCH_SIZE, i * BATCH_SIZE + batch_size) + max_encoder_length

            time_idx_list.append(time_idx)

    # Concatenate
    all_predictions = np.concatenate(predictions_list)
    all_actuals = np.concatenate(actuals_list)
    all_time_idx = np.concatenate(time_idx_list)

    # Filter to test set
    test_time_indices = test_df['time_idx'].values
    test_mask = np.isin(all_time_idx, test_time_indices)
    median_predictions = all_predictions[test_mask]
    actuals = all_actuals[test_mask]

    print(f"Test set samples: {len(median_predictions)}")

    # Metrics
    mse = mean_squared_error(actuals, median_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, median_predictions)
    direction_accuracy = np.mean(np.sign(median_predictions) == np.sign(actuals)) * 100
    interval_accuracy = np.mean(np.abs(median_predictions - actuals) <= 0.001) * 100

    print("\n" + "=" * 80)
    print("TEST SET RESULTS (IMPROVED TFT)")
    print("=" * 80)
    print(f"MSE:                  {mse:.6f}")
    print(f"RMSE:                 {rmse:.5f}")
    print(f"R² Score:             {r2:.4f}")
    print(f"Direction Accuracy:   {direction_accuracy:.2f}%")
    print(f"Interval Accuracy:    {interval_accuracy:.2f}%")
    print(f"\nPrediction Std:       {np.std(median_predictions):.6f}")
    print(f"Target Std:           {np.std(actuals):.6f}")
    print(f"Std Ratio:            {np.std(median_predictions)/np.std(actuals):.4f}")
    print("=" * 80)

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    try:
        raw_predictions = tft.predict(test_dataloader, mode="raw", return_x=True)
        interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")

        if "encoder_variables" in interpretation:
            encoder_importance = interpretation["encoder_variables"]
            all_features = time_varying_unknown + time_varying_known
            print("\nTop 20 Most Important Features:")

            # Combine features with importance
            feature_importance = list(zip(all_features[:len(encoder_importance)], encoder_importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            for i, (var, imp) in enumerate(feature_importance[:20], 1):
                print(f"{i:2d}. {var:50s}: {imp:.4f}")

    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    print("=" * 80)

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'interval_accuracy': interval_accuracy,
        'predictions': median_predictions,
        'actuals': actuals
    }


def save_results(results, model_name='tft_improved', output_dir=OUTPUT_DIR):
    """Save evaluation results."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metrics
    metrics = {
        'model': model_name,
        'timestamp': timestamp,
        'mse': float(results['mse']),
        'rmse': float(results['rmse']),
        'r2': float(results['r2']),
        'direction_accuracy': float(results['direction_accuracy']),
        'interval_accuracy': float(results['interval_accuracy']),
        'prediction_std': float(np.std(results['predictions'])),
        'actual_std': float(np.std(results['actuals']))
    }

    metrics_path = os.path.join(output_dir, f'{model_name}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")

    # Predictions
    predictions_df = pd.DataFrame({
        'actual': results['actuals'],
        'predicted': results['predictions'],
        'error': results['actuals'] - results['predictions']
    })

    predictions_path = os.path.join(output_dir, f'{model_name}_predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved: {predictions_path}")

    return metrics_path, predictions_path, timestamp


def plot_predictions(predictions, actuals, output_dir=OUTPUT_DIR, timestamp=None):
    """Plot predictions vs actuals."""

    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series
    n = min(100, len(predictions))
    x = np.arange(n)
    axes[0].plot(x, actuals[:n], 'o-', label='Actual', color='black', markersize=3)
    axes[0].plot(x, predictions[:n], 's-', label='Improved TFT', color='blue', markersize=3, alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Forward Returns')
    axes[0].set_title('Improved TFT: Predictions vs Actuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter
    axes[1].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', label='Perfect')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Improved TFT: Prediction Scatter')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f'tft_improved_predictions_{timestamp}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved: {save_path}")

    return save_path


if __name__ == "__main__":
    print("=" * 80)
    print("IMPROVED TFT WITH ENHANCED CONFIGURATION")
    print("=" * 80)

    # Check installation
    try:
        import pytorch_forecasting
        print(f"\n✓ pytorch-forecasting version: {pytorch_forecasting.__version__}")
    except ImportError:
        print("\n❌ ERROR: pytorch-forecasting not installed")
        print("\nInstall with: pip install pytorch-forecasting lightning")
        exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare data
    df, train_df, test_df, val_df, time_varying_unknown, time_varying_known = prepare_data_for_tft(DATA_PATH)

    # Train
    tft, trainer, training = train_improved_tft(
        df, train_df, val_df,
        time_varying_unknown, time_varying_known,
        max_epochs=MAX_EPOCHS,
        find_learning_rate=False  # Set to False to skip LR finder
    )

    # Evaluate
    results = evaluate_tft(tft, df, test_df, time_varying_unknown, time_varying_known, training)

    # Save
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    metrics_path, predictions_path, timestamp = save_results(results)
    plot_predictions(results['predictions'], results['actuals'], timestamp=timestamp)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'tft_improved_model_{timestamp}.ckpt')
    trainer.save_checkpoint(model_path)
    print(f"✓ Model saved: {model_path}")

    print("\n" + "=" * 80)
    print("IMPROVED TFT COMPLETE!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  R² Score:           {results['r2']:.4f}")
    print(f"  RMSE:               {results['rmse']:.5f}")
    print(f"  Direction Accuracy: {results['direction_accuracy']:.2f}%")
    print(f"  Interval Accuracy:  {results['interval_accuracy']:.2f}%")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)

"""
Hyperparameter Grid Search for Simple Transformer

Systematically tests different hyperparameter combinations to find the best configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from itertools import product
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from data_preprocessing import StockDataPreprocessor
from transformer import create_model


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DirectionalMSELoss(nn.Module):
    """Custom loss combining MSE with directional accuracy."""

    def __init__(self, alpha=0.3, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        sign_match = torch.sign(pred) * torch.sign(target)
        direction_loss = torch.mean(1 - sign_match)
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        variance_penalty = torch.abs(target_std - pred_std) / (target_std + 1e-8)
        total_loss = self.alpha * mse + self.beta * direction_loss + 0.1 * variance_penalty
        return total_loss


def train_model_quick(model, train_loader, val_loader, device, lr, num_epochs=25, show_progress=True):
    """Train model with given hyperparameters (fast version for grid search)."""

    model = model.to(device)
    criterion = DirectionalMSELoss(alpha=0.5, beta=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="  Training", leave=False, disable=not show_progress)

    for epoch in epoch_pbar:
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'patience': f'{patience_counter}/{patience}'
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.set_postfix_str(f'Early stopped at epoch {epoch+1}')
                break

    epoch_pbar.close()
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Compute metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    # Interval accuracy: percentage within ±0.005
    interval_accuracy = np.mean(np.abs(predictions - targets) <= 0.005) * 100

    # Directional accuracy: percentage where sign matches
    directional_accuracy = np.mean(np.sign(predictions) == np.sign(targets)) * 100

    return {
        'r2': r2,
        'rmse': rmse,
        'accuracy': interval_accuracy,
        'dir_accuracy': directional_accuracy,
        'predictions': predictions,
        'targets': targets
    }


def grid_search():
    """Perform grid search over hyperparameters."""

    print("=" * 80)
    print("TRANSFORMER HYPERPARAMETER GRID SEARCH")
    print("=" * 80)

    # Define hyperparameter grid
    # Full grid search: 108 combinations
    param_grid = {
        'window': [20, 30, 40],       # 3 values
        'batch': [32, 64],            # 2 values
        'lr': [0.0001, 0.0005, 0.001],  # 3 values
        'hidden': [64, 128, 256],     # 3 values
        'dropout': [0.2, 0.3]         # 2 values
    }
    # Total combinations: 3 × 2 × 3 × 3 × 2 = 108

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    total_combinations = len(combinations)
    print(f"\nTotal hyperparameter combinations: {total_combinations}")
    print(f"\nHyperparameter ranges:")
    for key, val in param_grid.items():
        print(f"  {key}: {val}")

    print(f"\nTraining epochs per combination: 25 (with early stopping)")
    print(f"Estimated total time: ~{total_combinations * 2} minutes")
    print("\n" + "=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Results storage
    results = []

    # Main grid search loop
    for idx, params in enumerate(combinations, 1):
        window, batch, lr, hidden, dropout = params

        print(f"[{idx}/{total_combinations}] Testing: window={window}, batch={batch}, lr={lr}, hidden={hidden}, dropout={dropout}")

        try:
            # Load data with current window size
            preprocessor = StockDataPreprocessor(lookback_window=window, prediction_horizon=1)

            data = preprocessor.preprocess_pipeline(
                train_path='datasets/train_2000.csv',
                missing_strategy='mean',
                test_size=0.1,
                val_size=100
            )

            # Create datasets
            train_dataset = StockDataset(data['X_train'], data['y_train'])
            val_dataset = StockDataset(data['X_val'], data['y_val'])
            test_dataset = StockDataset(data['X_test'], data['y_test'])

            train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

            # Create model
            model = create_model(
                input_dim=data['n_features'],
                d_model=hidden,
                nhead=4 if hidden >= 64 else 2,  # Adjust heads based on hidden size
                num_layers=4,
                dim_feedforward=hidden * 4,
                dropout=dropout
            )

            # Train
            model = train_model_quick(model, train_loader, val_loader, device, lr, num_epochs=25)

            # Evaluate on test set
            metrics = evaluate_model(model, test_loader, device)

            # Store results
            result = {
                'window': window,
                'batch': batch,
                'lr': lr,
                'hidden': hidden,
                'dropout': dropout,
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'accuracy': metrics['accuracy'],
                'dir_accuracy': metrics['dir_accuracy']
            }
            results.append(result)

            print(f"  R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.6f} | Accuracy (±0.005): {metrics['accuracy']:.2f}% | Dir Acc: {metrics['dir_accuracy']:.2f}%")

            # Clean up
            del model, train_loader, val_loader, test_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            result = {
                'window': window,
                'batch': batch,
                'lr': lr,
                'hidden': hidden,
                'dropout': dropout,
                'r2': -999,
                'rmse': 999,
                'accuracy': 0,
                'dir_accuracy': 0
            }
            results.append(result)

        print()

    return results, param_grid


def save_and_display_results(results, param_grid):
    """Save results to CSV and display top 10."""

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by R² (descending)
    df = df.sort_values('r2', ascending=False)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'transformer_grid_search_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("TOP 10 RESULTS (sorted by R²)")
    print("=" * 80)

    # Display top 10
    top_10 = df.head(10)
    print(top_10.to_string(index=False))

    # Best hyperparameters
    best = df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"window: {int(best['window'])}")
    print(f"batch: {int(best['batch'])}")
    print(f"lr: {best['lr']}")
    print(f"hidden: {int(best['hidden'])}")
    print(f"dropout: {best['dropout']}")

    print(f"\nPerformance:")
    print(f"  R²: {best['r2']:.4f}")
    print(f"  RMSE: {best['rmse']:.6f}")
    print(f"  Accuracy (±0.005): {best['accuracy']:.2f}%")
    print(f"  Directional Accuracy: {best['dir_accuracy']:.2f}%")

    print("\n" + "=" * 80)
    print(f"Results saved to: {csv_path}")
    print("=" * 80)

    return best, csv_path


def retrain_best_model(best_params):
    """Retrain the best model with more epochs and save it."""

    print("\n" + "=" * 80)
    print("RETRAINING BEST MODEL WITH MORE EPOCHS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    preprocessor = StockDataPreprocessor(
        lookback_window=int(best_params['window']),
        prediction_horizon=1
    )

    data = preprocessor.preprocess_pipeline(
        train_path='datasets/train_2000.csv',
        missing_strategy='mean',
        test_size=0.1,
        val_size=100
    )

    # Create datasets
    train_dataset = StockDataset(data['X_train'], data['y_train'])
    val_dataset = StockDataset(data['X_val'], data['y_val'])
    test_dataset = StockDataset(data['X_test'], data['y_test'])

    batch = int(best_params['batch'])
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Create model
    hidden = int(best_params['hidden'])
    model = create_model(
        input_dim=data['n_features'],
        d_model=hidden,
        nhead=4 if hidden >= 64 else 2,
        num_layers=4,
        dim_feedforward=hidden * 4,
        dropout=best_params['dropout']
    )

    print(f"Training with best hyperparameters for 50 epochs...")
    print(f"  window: {int(best_params['window'])}")
    print(f"  batch: {int(best_params['batch'])}")
    print(f"  lr: {best_params['lr']}")
    print(f"  hidden: {int(best_params['hidden'])}")
    print(f"  dropout: {best_params['dropout']}")

    # Train for more epochs
    model = train_model_quick(model, train_loader, val_loader, device, best_params['lr'], num_epochs=50)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    print(f"\nFinal Performance:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Accuracy (±0.005): {metrics['accuracy']:.2f}%")
    print(f"  Directional Accuracy: {metrics['dir_accuracy']:.2f}%")

    # Save model
    model_path = 'best_transformer_from_grid_search.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': best_params,
        'metrics': metrics
    }, model_path)

    print(f"\nBest model saved to: {model_path}")
    print("=" * 80)


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Run grid search
    results, param_grid = grid_search()

    # Save and display results
    best_params, csv_path = save_and_display_results(results, param_grid)

    # Optionally retrain best model with more epochs
    print("\n" + "=" * 80)
    response = input("Retrain best model with 50 epochs? (y/n): ")
    if response.lower() == 'y':
        retrain_best_model(best_params)
    else:
        print("Skipping retraining. You can retrain later using the saved results.")

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE!")
    print("=" * 80)
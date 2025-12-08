"""
Data Preprocessing
This module handles:
1. Loading data
2. Handling missing values
3. Creating sliding window sequences
4. Train/Val/Test split
5. Normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class StockDataPreprocessor:
    """
    Preprocessor for stock market data with temporal sequence generation.
    """
    
    def __init__(self, lookback_window: int = 30, prediction_horizon: int = 1):
        """
        Args:
            lookback_window: Number of past timesteps to use for prediction (default: 30)
            prediction_horizon: Number of days ahead to predict (default: 1)
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'forward_returns'  # predict returns, then convert to position
        
    def load_data(self, train_path: str) -> pd.DataFrame:
        """
        Load training data from CSV.
        
        Args:
            train_path: Path to train.csv
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {train_path}...")
        df = pd.read_csv(train_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:10]}... (showing first 10)")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'forward_fill', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        print(f"\nHandling missing values using strategy: {strategy}")
        print(f"Missing values before: {df.isnull().sum().sum()}")
        
        if strategy == 'mean':
            # Use mean imputation for each column
            df = df.fillna(df.mean())
        elif strategy == 'forward_fill':
            # Forward fill then backward fill for any remaining NaNs
            df = df.ffill().bfill()  # Updated pandas syntax
        elif strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Missing values after: {df.isnull().sum().sum()}")
        print(f"Rows remaining: {len(df)}")
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Separate features and target variable.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features_df, target_series, split_series)
        """
        print("\nPreparing features and target...")

        # Create t-1 (lagged) features for forward_returns, risk_free_rate, market_forward_excess_returns
        df['forward_returns_lag1'] = df['forward_returns'].shift(1)
        df['risk_free_rate_lag1'] = df['risk_free_rate'].shift(1)
        df['market_forward_excess_returns_lag1'] = df['market_forward_excess_returns'].shift(1)

        # Use ALL features except date_id, forward_returns, and split
        # Exclude non-feature columns
        exclude_columns = ['date_id', 'forward_returns', 'split', 'risk_free_rate', 'market_forward_excess_returns']

        # Get all column names except excluded ones
        all_columns = df.columns.tolist()
        feature_subset = [col for col in all_columns if col not in exclude_columns]

        # Add back the lagged features we created
        self.feature_columns = feature_subset

        # Drop rows with NaN created by lagging (first row will have NaN)
        print(f"Rows before dropping NaN from lagging: {len(df)}")
        df = df.dropna(subset=self.feature_columns + [self.target_column])
        print(f"Rows after dropping NaN: {len(df)}")

        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Feature columns (first 20): {self.feature_columns[:20]}...")

        X = df[self.feature_columns]
        y = df[self.target_column]
        split = df['split'] if 'split' in df.columns else None

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y, split
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, split: pd.Series = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time series prediction.

        Args:
            X: Feature DataFrame
            y: Target Series
            split: Split indicator Series (optional)

        Returns:
            Tuple of (X_sequences, y_sequences, split_sequences)
            X_sequences shape: (n_samples, lookback_window, n_features)
            y_sequences shape: (n_samples,)
            split_sequences shape: (n_samples,) - split indicator for each sequence
        """
        print(f"\nCreating sequences with lookback_window={self.lookback_window}...")

        X_np = X.values
        y_np = y.values
        split_np = split.values if split is not None else None

        X_sequences = []
        y_sequences = []
        split_sequences = []

        # Create sliding windows
        for i in range(len(X_np) - self.lookback_window - self.prediction_horizon + 1):
            # Input: past lookback_window days
            X_seq = X_np[i:i + self.lookback_window]
            # Target: forward_returns at day (i + lookback_window)
            target_idx = i + self.lookback_window + self.prediction_horizon - 1
            y_seq = y_np[target_idx]

            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

            # Split is determined by the target row
            if split_np is not None:
                split_sequences.append(split_np[target_idx])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        split_sequences = np.array(split_sequences) if split_np is not None else None

        print(f"Sequences created:")
        print(f"  X_sequences shape: {X_sequences.shape}")
        print(f"  y_sequences shape: {y_sequences.shape}")
        if split_sequences is not None:
            print(f"  split_sequences shape: {split_sequences.shape}")

        return X_sequences, y_sequences, split_sequences
    
    def train_val_test_split(self, X_seq: np.ndarray, y_seq: np.ndarray,
                            split_seq: np.ndarray = None,
                            test_size: float = 0.1, val_size: int = 100) -> dict:
        """
        Split data into train/val/test sets.
        If split_seq is provided, use it (0=train, 1=test, 2=val).
        Otherwise, use temporal ordering.

        Args:
            X_seq: Sequence features
            y_seq: Sequence targets
            split_seq: Split indicator array (0=train, 1=test, 2=val)
            test_size: Proportion of data for test set (only used if split_seq is None)
            val_size: Number of samples for validation set (only used if split_seq is None)

        Returns:
            Dictionary containing train/val/test splits
        """
        print(f"\nSplitting data...")

        if split_seq is not None:
            # Use provided split column: 0=train, 1=test, 2=val
            print("Using provided split column (0=train, 1=test, 2=val)...")

            train_mask = split_seq == 0
            test_mask = split_seq == 1
            val_mask = split_seq == 2

            X_train = X_seq[train_mask]
            y_train = y_seq[train_mask]

            X_val = X_seq[val_mask]
            y_val = y_seq[val_mask]

            X_test = X_seq[test_mask]
            y_test = y_seq[test_mask]

        else:
            # Use temporal ordering
            print(f"Using temporal split (test_size={test_size}, val_size={val_size})...")

            n_samples = len(X_seq)
            test_samples = int(n_samples * test_size)

            # Split: [train | val | test]
            test_start = n_samples - test_samples
            val_start = test_start - val_size

            X_train = X_seq[:val_start]
            y_train = y_seq[:val_start]

            X_val = X_seq[val_start:test_start]
            y_val = y_seq[val_start:test_start]

            X_test = X_seq[test_start:]
            y_test = y_seq[test_start:]

        print(f"Split completed:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def normalize_data(self, data_dict: dict) -> dict:
        """
        Normalize features using StandardScaler fitted on training data only.
        
        Args:
            data_dict: Dictionary containing train/val/test splits
            
        Returns:
            Dictionary with normalized data
        """
        print("\nNormalizing data...")
        
        X_train = data_dict['X_train']
        X_val = data_dict['X_val']
        X_test = data_dict['X_test']
        
        # Reshape to 2D for scaling: (n_samples * lookback_window, n_features)
        n_train, seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # Fit scaler on training data only
        print(f"  Fitting scaler on {X_train_2d.shape[0]} training samples...")
        self.scaler.fit(X_train_2d)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train_2d).reshape(n_train, seq_len, n_features)
        X_val_scaled = self.scaler.transform(X_val_2d).reshape(n_val, seq_len, n_features)
        X_test_scaled = self.scaler.transform(X_test_2d).reshape(n_test, seq_len, n_features)
        
        print(f"  Normalization complete")
        print(f"  Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        print(f"  Val mean:   {X_val_scaled.mean():.6f}, std: {X_val_scaled.std():.6f}")
        print(f"  Test mean:  {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")
        
        return {
            'X_train': X_train_scaled,
            'y_train': data_dict['y_train'],
            'X_val': X_val_scaled,
            'y_val': data_dict['y_val'],
            'X_test': X_test_scaled,
            'y_test': data_dict['y_test']
        }
    
    def preprocess_pipeline(self, train_path: str,
                           missing_strategy: str = 'mean',
                           test_size: float = 0.1,
                           val_size: int = 100) -> dict:
        """
        Complete preprocessing pipeline.

        Args:
            train_path: Path to train.csv
            missing_strategy: Strategy for handling missing values
            test_size: Proportion for test set (only used if no split column)
            val_size: Number of samples for validation set (only used if no split column)

        Returns:
            Dictionary with preprocessed train/val/test data
        """
        print("="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)

        # Step 1: Load data
        df = self.load_data(train_path)

        # Check for split column
        has_split_column = 'split' in df.columns
        if has_split_column:
            print("\n✓ Found 'split' column - will use for train/val/test split")
            split_counts = df['split'].value_counts().sort_index()
            print(f"  Split distribution: {dict(split_counts)}")
        else:
            print("\n⚠ No 'split' column found - will use temporal split")

        # Step 2: Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)

        # Step 3: Prepare features and target
        X, y, split = self.prepare_features_and_target(df)

        # Step 4: Create sequences
        X_seq, y_seq, split_seq = self.create_sequences(X, y, split)

        # Step 5: Train/val/test split
        data_dict = self.train_val_test_split(X_seq, y_seq, split_seq, test_size, val_size)

        # Step 6: Normalize data
        data_dict = self.normalize_data(data_dict)

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)

        # Add metadata
        data_dict['n_features'] = X_seq.shape[2]
        data_dict['lookback_window'] = self.lookback_window
        data_dict['feature_columns'] = self.feature_columns

        return data_dict


def convert_returns_to_position(returns: np.ndarray, method: str = 'simple') -> np.ndarray:
    """
    Convert predicted returns to position [0, 2].
    
    Args:
        returns: Predicted returns array
        method: Conversion method ('simple', 'sigmoid', 'tanh')
        
    Returns:
        Position array in range [0, 2]
    """
    if method == 'simple':
        # Simple linear mapping: positive returns -> position > 1, negative -> position < 1
        # Clip extreme values
        positions = 1 + np.clip(returns * 50, -1, 1)  # Scale factor of 50 is tunable
        
    elif method == 'sigmoid':
        # Use sigmoid to map to [0, 1], then scale to [0, 2]
        positions = 2 / (1 + np.exp(-returns * 50))
        
    elif method == 'tanh':
        # Use tanh to map to [-1, 1], then shift and scale to [0, 2]
        positions = 1 + np.tanh(returns * 50)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure positions are in valid range
    positions = np.clip(positions, 0, 2)
    
    return positions


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(lookback_window=30, prediction_horizon=1)
    
    # Run preprocessing pipeline
    data = preprocessor.preprocess_pipeline(
        train_path = 'train_2000.csv',
        missing_strategy='mean',
        test_size=0.1,
        val_size=100
    )
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Number of features: {data['n_features']}")
    print(f"Lookback window: {data['lookback_window']}")
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Validation samples: {data['X_val'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"\nInput shape: (batch_size, {data['lookback_window']}, {data['n_features']})")
    print(f"Output shape: (batch_size,)")
    
    # Test position conversion
    print("\n" + "="*80)
    print("TESTING POSITION CONVERSION")
    print("="*80)
    sample_returns = np.array([-0.02, -0.01, 0, 0.01, 0.02])
    positions = convert_returns_to_position(sample_returns, method='simple')
    print(f"Sample returns: {sample_returns}")
    print(f"Converted positions: {positions}")
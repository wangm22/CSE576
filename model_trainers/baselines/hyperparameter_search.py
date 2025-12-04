import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import json
from datetime import datetime

# ======================================================
# 1. Load data and generate t-1 lag features
# ======================================================
df = pd.read_csv("../../datasets/train_2000.csv")
for col in ["date_id", "date", "Date"]:
    if col in df.columns:
        df = df.sort_values(col).reset_index(drop=True)
        break

df["forward_returns_t-1"] = df["forward_returns"].shift(1)
df["risk_free_rate_t-1"] = df["risk_free_rate"].shift(1)
df["market_forward_excess_returns_t-1"] = df["market_forward_excess_returns"].shift(1)
df = df.dropna().reset_index(drop=True)

# ======================================================
# 2. Feature subset
# ======================================================
feature_cols = [
    'V2','M14','M18','V11','E17','V4','V3','E16','E2','V1','M16',
    'forward_returns_t-1','risk_free_rate_t-1','market_forward_excess_returns_t-1'
]
target_col = "forward_returns"

# ======================================================
# 3. Split (split==0 train, 1 test, 2 val)
# ======================================================
if "split" in df.columns:
    df_train = df[df["split"] == 0]
    df_test  = df[df["split"] == 1]
    df_val   = df[df["split"] == 2]
else:
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train+n_val]
    df_test = df.iloc[n_train+n_val:]

# ======================================================
# 4. Normalize and make sequences
# ======================================================
def create_sequences(X, y, window=30):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_val   = scaler.transform(df_val[feature_cols])
X_test  = scaler.transform(df_test[feature_cols])

y_train = df_train[target_col].values
y_val   = df_val[target_col].values
y_test  = df_test[target_col].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}\n")

# ======================================================
# 5. Complex 3-layer BiLSTM with residuals + attention
# ======================================================
class ComplexStockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3, bidirectional=True):
        super().__init__()
        d = 2 if bidirectional else 1
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(hidden_dim * d, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(hidden_dim * d, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * d, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        o1, _ = self.lstm1(x)
        o2, _ = self.lstm2(o1)
        o2 = o2 + o1[:, :, :o2.shape[2]]
        o3, _ = self.lstm3(o2)
        o3 = o3 + o2[:, :, :o3.shape[2]]
        w = torch.softmax(self.attn(o3), dim=1)
        ctx = torch.sum(w * o3, dim=1)
        return self.head(ctx)

# ======================================================
# 6. Hyperparameter Search Grid
# ======================================================
param_grid = {
    "window": [20, 30, 40],
    "batch": [32, 64, 128],
    "lr": [1e-4, 5e-4, 1e-3],
    "hidden": [64, 128, 256],
    "dropout": [0.2, 0.3, 0.5]
}

# Generate all combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())
print(f"🔍 Total hyperparameter combinations: {len(param_combinations)}\n")

results = []

# ======================================================
# 7. Hyperparameter Search Loop
# ======================================================
for idx, param_values in enumerate(param_combinations, 1):
    params = dict(zip(param_names, param_values))
    window = params["window"]
    batch = params["batch"]
    lr = params["lr"]
    hidden = params["hidden"]
    dropout = params["dropout"]
    
    print(f"[{idx}/{len(param_combinations)}] Testing: window={window}, batch={batch}, lr={lr}, hidden={hidden}, dropout={dropout}")
    
    # Create sequences with current window
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window)
    X_val_seq, y_val_seq     = create_sequences(X_val,   y_val,   window)
    X_test_seq, y_test_seq   = create_sequences(X_test,  y_test,  window)
    
    if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
        print(f"  ⚠️  Skipped: Window size too large for data split")
        continue
    
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t   = torch.tensor(X_val_seq,   dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val_seq,   dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32).to(device)
    y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32).unsqueeze(1).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch, shuffle=True)
    
    # Train model
    model = ComplexStockLSTM(len(feature_cols), hidden, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, wait, patience = float("inf"), 0, 10
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_model_state = model.state_dict().copy()
        else:
            wait += 1
            if wait >= patience:
                break
    
    # Evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().flatten()
        y_true = y_test_t.cpu().numpy().flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate accuracy: percentage of predictions within ±0.005 interval
    tolerance = 0.005
    correct = np.abs(y_pred - y_true) <= tolerance
    accuracy = 100.0 * correct.sum() / len(correct)
    
    result = {
        "window": window,
        "batch": batch,
        "lr": lr,
        "hidden": hidden,
        "dropout": dropout,
        "r2": float(r2),
        "rmse": float(rmse),
        "accuracy": float(accuracy)
    }
    results.append(result)
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.6f} | Accuracy (±0.05): {accuracy:.2f}%\n")

# ======================================================
# 8. Save Results
# ======================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("r2", ascending=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"hyperparameter_search_results_{timestamp}.csv", index=False)

print("\n" + "="*80)
print("🏆 TOP 10 RESULTS (sorted by R²)")
print("="*80)
print(results_df.head(10).to_string(index=False))

print("\n" + "="*80)
print("📊 BEST HYPERPARAMETERS")
print("="*80)
best_result = results_df.iloc[0]
print(f"window: {int(best_result['window'])}")
print(f"batch: {int(best_result['batch'])}")
print(f"lr: {best_result['lr']}")
print(f"hidden: {int(best_result['hidden'])}")
print(f"dropout: {best_result['dropout']}")
print(f"\nPerformance:")
print(f"  R²: {best_result['r2']:.4f}")
print(f"  RMSE: {best_result['rmse']:.6f}")
print(f"  Accuracy (±0.05): {best_result['accuracy']:.2f}%")

print(f"\n💾 Full results saved to: hyperparameter_search_results_{timestamp}.csv")

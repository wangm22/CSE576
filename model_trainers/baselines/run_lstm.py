import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ======================================================
# 1. Parse arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="../../datasets/train_2000.csv", help="Dataset CSV path")
parser.add_argument("--window", type=int, default=30)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--tol", type=float, default=0.01, help="Absolute tolerance for accuracy check")
parser.add_argument("--rel_tol", type=float, default=None,
                    help="Optional relative tolerance (% of |y_true|); overrides tol if given, e.g. 0.05 for ±5%")
args = parser.parse_args()

# ======================================================
# 2. Load data and generate t-1 lag features
# ======================================================
df = pd.read_csv(args.csv)
for col in ["date_id", "date", "Date"]:
    if col in df.columns:
        df = df.sort_values(col).reset_index(drop=True)
        break

df["forward_returns_t-1"] = df["forward_returns"].shift(1)
df["risk_free_rate_t-1"] = df["risk_free_rate"].shift(1)
df["market_forward_excess_returns_t-1"] = df["market_forward_excess_returns"].shift(1)
df = df.dropna().reset_index(drop=True)

# ======================================================
# 3. Feature subset
# ======================================================
feature_cols = [
    'V2','M14','M18','V11','E17','V4','V3','E16','E2','V1','M16',
    'forward_returns_t-1','risk_free_rate_t-1','market_forward_excess_returns_t-1'
]
target_col = "forward_returns"

# ======================================================
# 4. Split (split==0 train, 1 test, 2 val)
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
# 5. Normalize and make sequences
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

X_train_seq, y_train_seq = create_sequences(X_train, y_train, args.window)
X_val_seq, y_val_seq     = create_sequences(X_val,   y_val,   args.window)
X_test_seq, y_test_seq   = create_sequences(X_test,  y_test,  args.window)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val_seq,   dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val_seq,   dtype=torch.float32).unsqueeze(1).to(device)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32).unsqueeze(1).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch, shuffle=True)

print(f"✅ Using device: {device}")
print(f"Train/Val/Test shapes: {X_train_t.shape}, {X_val_t.shape}, {X_test_t.shape}")

# ======================================================
# 6. Complex 3-layer BiLSTM with residuals + attention
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
# 7. Train
# ======================================================
model = ComplexStockLSTM(len(feature_cols), args.hidden, args.dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_val, wait, patience = float("inf"), 0, 10

for epoch in range(1, args.epochs + 1):
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

    print(f"Epoch {epoch:03d} | Train {run_loss/len(train_loader):.6f} | Val {val_loss:.6f}")
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), "../../saved_models/best_lstm.pt")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping.")
            break

# ======================================================
# 8. Evaluate (R², RMSE, Interval Accuracy)
# ======================================================
model.load_state_dict(torch.load("../../saved_models/best_lstm.pt"))
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).cpu().numpy().flatten()
    y_true = y_test_t.cpu().numpy().flatten()

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# --- interval accuracy ---
if args.rel_tol is not None:
    tol_arr = np.abs(y_true) * args.rel_tol
else:
    tol_arr = np.full_like(y_true, args.tol)
correct = np.abs(y_pred - y_true) <= tol_arr
accuracy = 100.0 * correct.sum() / len(correct)

print(f"\n✅ Complex LSTM (t-1 features)")
print(f"R²: {r2:.4f} | RMSE: {rmse:.6f} | Interval Accuracy: {accuracy:.2f}% "
      f"(tolerance={'±{:.4f}'.format(args.tol) if args.rel_tol is None else f'±{args.rel_tol*100:.1f}% of |y|'} )")

pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "correct": correct}).to_csv(
    "lstm_predictions.csv", index=False)
print("💾 Saved predictions to lstm_predictions.csv")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt

# ==========================================================
# 1. Load and preprocess dataset
# ==========================================================
df = pd.read_csv("../datasets/train_2000.csv")
target_col = "forward_returns"
feature_cols = [
    c for c in df.columns
    if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
]

X = df[feature_cols].values
y = df[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, window=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

WINDOW = 30
X_seq, y_seq = create_sequences(X_scaled, y, window=WINDOW)
split_idx = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

print(f"Using device: {device}")
print(f"Training shape: {X_train_t.shape}, Testing shape: {X_test_t.shape}")

# ==========================================================
# 2. Complex 3-layer Bidirectional Residual LSTM with Attention
# ==========================================================
class ComplexStockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(hidden_dim * self.directions, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(hidden_dim * self.directions, hidden_dim, batch_first=True, bidirectional=bidirectional)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out2 = out2 + out1[:, :, :out2.shape[2]]  # residual connection
        out3, _ = self.lstm3(out2)
        out3 = out3 + out2[:, :, :out3.shape[2]]

        attn_weights = torch.softmax(self.attention(out3), dim=1)
        context = torch.sum(attn_weights * out3, dim=1)
        output = self.fc(context)
        return output

# ==========================================================
# 3. Train the full model
# ==========================================================
model = ComplexStockLSTM(input_dim=X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
best_val = np.inf
patience, wait = 10, 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test_t), y_test_t).item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {running_loss/len(train_loader):.6f} - Val Loss: {val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), "best_complex_lstm.pt")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping triggered.")
            break

# ==========================================================
# 4. Evaluate full model
# ==========================================================
model.load_state_dict(torch.load("best_complex_lstm.pt"))
model.eval()
with torch.no_grad():
    preds = model(X_test_t).cpu().numpy().flatten()
    true = y_test_t.cpu().numpy().flatten()

r2 = r2_score(true, preds)
rmse = np.sqrt(mean_squared_error(true, preds))
print(f"\n✅ Full Complex LSTM Performance: R²={r2:.4f}, RMSE={rmse:.6f}")

# ==========================================================
# 5. SHAP computation
# ==========================================================
print("\n🔍 Computing SHAP values (small sample for speed)...")

background = X_train_t[:30].detach().to(device)
test_sample = X_test_t[:5].detach().to(device)

torch.backends.cudnn.enabled = False
model.train()  # needed for SHAP GradientExplainer on RNN

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

torch.backends.cudnn.enabled = True
model.eval()

if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.squeeze(np.array(shap_values))  # shape: (samples, timesteps, features)

# ==========================================================
# 6. Compute feature importances
# ==========================================================
mean_abs = np.mean(np.abs(shap_values), axis=(0, 1)).squeeze()
feature_importance = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)

# Save top-20 features
TOP_K = 20
feature_importance_df = feature_importance.reset_index()
feature_importance_df.columns = ["feature", "mean_abs_shap_value"]
top_features_df = feature_importance_df.head(TOP_K)
top_features_df.to_csv("top20_features.csv", index=False)
print(f"💾 Saved top {TOP_K} features to top20_features.csv")

# Save all SHAP values for all features
n_samples, n_steps, n_feats = shap_values.shape
shap_flat = shap_values.reshape(n_samples * n_steps, n_feats)
shap_df = pd.DataFrame(shap_flat, columns=feature_cols)
shap_df.to_csv("shap_values.csv", index=False)
print(f"💾 Saved SHAP values for all features to shap_values.csv (shape={shap_df.shape})")

# Plot top features
# plt.figure(figsize=(6, 8))
# feature_importance.head(20).plot(kind="barh", title="Top 20 SHAP Feature Importance (Complex LSTM)")
# plt.tight_layout()
# plt.show()

# ==========================================================
# 7. Retrain with top-20 features (optional)
# ==========================================================
# important_features = top_features_df["feature"].tolist()
# print(f"\n📊 Retraining with top {len(important_features)} SHAP-selected features...")
# X_small = df[important_features].values
# y_small = df[target_col].values

# X_small_scaled = StandardScaler().fit_transform(X_small)
# X_seq_small, y_seq_small = create_sequences(X_small_scaled, y_small, window=WINDOW)
# split_idx = int(len(X_seq_small) * 0.8)
# X_train_s, X_test_s = X_seq_small[:split_idx], X_seq_small[split_idx:]
# y_train_s, y_test_s = y_seq_small[:split_idx], y_seq_small[split_idx:]

# X_train_s_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
# y_train_s_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1).to(device)
# X_test_s_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
# y_test_s_t = torch.tensor(y_test_s, dtype=torch.float32).unsqueeze(1).to(device)

# train_loader_s = DataLoader(TensorDataset(X_train_s_t, y_train_s_t), batch_size=64, shuffle=True)

# model_small = ComplexStockLSTM(input_dim=X_train_s.shape[2], hidden_dim=64, num_layers=2).to(device)
# optimizer = torch.optim.Adam(model_small.parameters(), lr=1e-3)

# best_val, wait = np.inf, 0
# for epoch in range(EPOCHS):
#     model_small.train()
#     running_loss = 0.0
#     for Xb, yb in train_loader_s:
#         optimizer.zero_grad()
#         preds = model_small(Xb)
#         loss = criterion(preds, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     model_small.eval()
#     with torch.no_grad():
#         val_loss = criterion(model_small(X_test_s_t), y_test_s_t).item()

#     print(f"[Reduced] Epoch {epoch+1}/{EPOCHS} - Train Loss: {running_loss/len(train_loader_s):.6f} - Val Loss: {val_loss:.6f}")

#     if val_loss < best_val:
#         best_val = val_loss
#         wait = 0
#         torch.save(model_small.state_dict(), "best_lstm_reduced.pt")
#     else:
#         wait += 1
#         if wait >= patience:
#             print("⏹ Early stopping (Reduced).")
#             break

# model_small.load_state_dict(torch.load("best_lstm_reduced.pt"))
# model_small.eval()
# with torch.no_grad():
#     preds_small = model_small(X_test_s_t).cpu().numpy().flatten()
#     true_small = y_test_s_t.cpu().numpy().flatten()

# r2_small = r2_score(true_small, preds_small)
# rmse_small = np.sqrt(mean_squared_error(true_small, preds_small))

# print(f"\n✅ Reduced LSTM Performance: R²={r2_small:.4f}, RMSE={rmse_small:.6f}")
# print(f"\n📈 Comparison:\nFull Model -> R²={r2:.4f}, RMSE={rmse:.6f}\nReduced Model -> R²={r2_small:.4f}, RMSE={rmse_small:.6f}")
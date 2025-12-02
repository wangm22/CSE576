import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import PatchTSTForPrediction
import os
from scipy.stats import spearmanr

# ====== Dataset Class (same as training) ======
class StockDataset(Dataset):
    def __init__(self, data, split_range, lookback=128, horizon=100):
        self.data = data.reset_index(drop=True)
        self.lookback = lookback
        self.horizon = horizon
        self.split_start, self.split_end = split_range

        self.valid_end_indices = []
        for end in range(self.split_start, self.split_end + 1):
            start = end - self.lookback + 1
            future_end = end + self.horizon
            if start >= 0 and future_end < len(self.data):
                self.valid_end_indices.append(end)

        if len(self.valid_end_indices) == 0:
            raise ValueError("No valid windows in test split.")

    def __len__(self):
        return len(self.valid_end_indices)

    def __getitem__(self, idx):
        end = self.valid_end_indices[idx]
        start = end - self.lookback + 1

        x = self.data.iloc[start : end + 1].values
        y = self.data["forward_returns"].iloc[end + 1 : end + 1 + self.horizon].values

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ====== MLP (same as training) ======
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self._mlp(x)


# ===========================
#         MAIN TESTING
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--patch_length", default=14, type=int)
    args = parser.parse_args()

    # ------------ Load data ------------
    df_raw = pd.read_csv("datasets/train_final.csv")

    train_df = df_raw[df_raw["split"] == 0]
    val_df   = df_raw[df_raw["split"] == 2]
    test_df  = df_raw[df_raw["split"] == 1]

    train_end = len(train_df) - 1
    val_start = train_end + 1
    val_end = val_start + len(val_df) - 1
    test_start = val_end + 1
    test_end = test_start + len(test_df) - 1

    df = df_raw.iloc[:, 2:-1]

    # ------------ Create test dataset ------------
    test_dataset = StockDataset(df, (test_start, test_end))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------ Load PatchTST (frozen) ------------
    save_dir = f"saved_models/patchtst/patch_{args.patch_length}/finals"
    patchtst = PatchTSTForPrediction.from_pretrained(save_dir).to(device)
    patchtst.eval()
    for p in patchtst.parameters():
        p.requires_grad = False

    # ------------ Load MLP ------------
    input_dim = df.shape[1]
    mlp = MLP(input_dim).to(device)
    mlp.load_state_dict(torch.load(f"mlp_return_predictor_{args.patch_length}_{args.epochs}.pt", map_location=device))
    mlp.eval()

    criterion = nn.MSELoss()

    preds_all = []
    targets_all = []

    # ============ TEST LOOP ============
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            patch_features = patchtst(x).prediction_outputs
            pred = mlp(patch_features).squeeze(-1)

            preds_all.append(pred.cpu())
            targets_all.append(y.cpu())

    preds_all = torch.cat(preds_all, dim=0).numpy()
    targets_all = torch.cat(targets_all, dim=0).numpy()

    # Metrics
    mse = np.mean((preds_all - targets_all)**2)
    mae = np.mean(np.abs(preds_all - targets_all))

    # ===== Additional Metrics: RMSE & R² =====
    rmse = np.sqrt(mse)

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum((preds_all - targets_all)**2)
    ss_tot = np.sum((targets_all - np.mean(targets_all))**2)
    r2 = 1 - ss_res / ss_tot

    # ===== Accuracy metric: |pred - true| <= 0.005 =====
    threshold = 0.005

    # Boolean mask for hits
    hits = np.abs(preds_all - targets_all) <= threshold

    # Overall accuracy
    accuracy = hits.mean()

    # ===== Spearman correlation (IC) =====
    pred_flat = preds_all.reshape(-1)
    true_flat = targets_all.reshape(-1)
    spearman_corr, spearman_p = spearmanr(pred_flat, true_flat)

    # Per-horizon Spearman
    horizon_spearman = []
    for h in range(targets_all.shape[1]):
        corr, _ = spearmanr(preds_all[:, h], targets_all[:, h])
        horizon_spearman.append(corr)
    horizon_spearman = np.array(horizon_spearman)

    print("========== TEST RESULTS ==========")
    print(f"Test R²: {r2:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Accuracy (|pred-true| <= ±0.005): {accuracy:.4f}")
    print(f"Spearman Corr (IC): {spearman_corr:.6f}   p={spearman_p:.3g}")

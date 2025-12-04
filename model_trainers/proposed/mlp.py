import pandas as pd
import numpy as np
import torch
import math
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import PatchTSTForPrediction

class StockDataset(Dataset):
    def __init__(self, data, split_range, lookback=128, horizon=1):
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
            raise ValueError(
                f"No valid windows found for split_range={split_range}. "
                f"Try reducing lookback/horizon or adjusting splits."
            )
        
        tft_data = pd.read_csv("datasets/predictions_with_quantiles.csv")
        self.tft_data = {row["date_id"]: 
                         row[["predicted_q02", "predicted_q10", "predicted_q25", "predicted_q50",
                              "predicted_q75", "predicted_q90", "predicted_q98"]].to_numpy(dtype=np.float32)
                         for _, row in tft_data.iterrows()}

        self.date_ids = list(self.data["date_id"])

    def __len__(self):
        return len(self.valid_end_indices)

    def __getitem__(self, idx):
        end = self.valid_end_indices[idx]
        start = end - self.lookback + 1

        date_id = self.date_ids[end]

        x = self.data.iloc[start : end + 1, 2 :].values
        tft_x = self.tft_data[date_id]
        y = self.data["forward_returns"].iloc[
            end + 1 : end + 1 + self.horizon
        ].values

        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(tft_x, dtype=torch.float32).unsqueeze(0),
                torch.tensor(y, dtype=torch.float32))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self._mlp(x)

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = math.inf

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patch_length", default=14, type=int)
    parser.add_argument("--lr_patchtst", default=1e-5, type=float)
    parser.add_argument("--lr_mlp", default=1e-3, type=float)
    parser.add_argument("--patchtst", default=True, type=bool)
    parser.add_argument("--tft", default=True, type=bool)
    args = parser.parse_args()

    df = pd.read_csv("datasets/train_final.csv")

    train_df = df[df["split"] == 0]
    val_df = df[df["split"] == 2]
    test_df = df[df["split"] == 1]
    
    train_end = len(train_df) - 1
    val_start = train_end + 1
    val_end = val_start + len(val_df) - 1
    test_start = val_end + 1
    test_end = test_start + len(test_df) - 1

    df = df.iloc[:, :-1]

    train_dataset = StockDataset(df, (0, train_end))
    val_dataset = StockDataset(df, (val_start, val_end))

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = f"saved_models/patchtst/patch_{args.patch_length}_{args.lr_patchtst}/finals"
    _patchtst = PatchTSTForPrediction.from_pretrained(save_dir).to(device)
    _patchtst.eval()
    for p in _patchtst.parameters():
        p.requires_grad = False

    if args.patchtst and args.tft:
        input_dim = 21
    elif args.patchtst:
        input_dim = 14
    elif args.tft:
        input_dim = 7

    _mlp = MLP(input_dim).to(device)

    optimizer = torch.optim.Adam(_mlp.parameters(), lr=args.lr_mlp)
    criterion = nn.MSELoss()

    early_stopper = EarlyStopper(patience=7, min_delta=1e-5)

    for epoch in range(args.epochs):
        _mlp.train()
        train_losses = []

        for x, tft_x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            tft_x = tft_x.to(device)
            
            with torch.no_grad():
                patch_features = _patchtst(x).prediction_outputs

            if args.patchtst and args.tft:
                mlp_features = torch.cat([patch_features, tft_x], dim=2)
            elif args.patchtst:
                mlp_features = patch_features
            elif args.tft:
                mlp_features = tft_x

            pred = _mlp(mlp_features).squeeze(-1)

            loss = criterion(pred, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _mlp.eval()
        val_losses = []

        with torch.no_grad():
            for x, tft_x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                tft_x = tft_x.to(device)

                patch_features = _patchtst(x).prediction_outputs

                if args.patchtst and args.tft:
                    mlp_features = torch.cat([patch_features, tft_x], dim=2)
                elif args.patchtst:
                    mlp_features = patch_features
                elif args.tft:
                    mlp_features = tft_x

                pred = _mlp(mlp_features).squeeze(-1)

                loss = criterion(pred, y)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if early_stopper.should_stop(avg_val_loss):
            print("🚨 Early stopping triggered!")
            break

    torch.save(_mlp.state_dict(), f"mlp_return_predictor.pt")
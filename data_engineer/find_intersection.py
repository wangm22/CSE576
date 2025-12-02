import pandas as pd

pca_df = pd.read_csv("data_engineer/pca_selection.csv")
shap_df = pd.read_csv("data_engineer/shap_selection.csv")

pca_set = set(pca_df["feature"])
shap_set = set(shap_df["feature"])

intersection = sorted(list(pca_set & shap_set))
print(intersection)
intersection = ["date_id"] + intersection + ["forward_returns", "risk_free_rate", "market_forward_excess_returns", "split"]

df = pd.read_csv("datasets/train_2000.csv")
intersect_df = df[intersection]
intersect_df.to_csv("datasets/train_intersect.csv", index=False)
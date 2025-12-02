import pandas as pd

final_df = pd.read_csv("datasets/train_intersect.csv")

final_df["forward_returns"] = final_df["forward_returns"].shift(1)
final_df["risk_free_rate"] = final_df["risk_free_rate"].shift(1)
final_df["market_forward_excess_returns"] = final_df["market_forward_excess_returns"].shift(1)
final_df = final_df.drop(index=0)

date = pd.to_datetime("2000-01-01") + pd.to_timedelta(final_df["date_id"], unit="d")
final_df.insert(1, "date", date)

final_df.to_csv("datasets/train_final.csv", index=False)
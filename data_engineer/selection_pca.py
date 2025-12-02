import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

class _PCA:
    def __init__(self, df):
        self.df = df
    
    def construct_PCA(self, cap_ratio=0.9):
        """
        Must call 
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df)

        self.pca = PCA(n_components=cap_ratio)
        X_pca = self.pca.fit_transform(X_scaled)

        self.pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(self.pca.n_components_)])

        return self.pca_df
    
    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_

    def loadings(self):
        return pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.df.columns
        )
    
    def num_PCs_over_explained_variance_ratio(self, cap_ratio=0.9):
        num_PCs = 0
        sum_ratio = 0

        for evr in self.explained_variance_ratio():
            num_PCs += 1
            sum_ratio += evr
            if sum_ratio >= cap_ratio:
                break
        
        self.num_features_select = num_PCs

        return num_PCs, sum_ratio
    
    def select_features(self, loading_threshold=0.1, cap_ratio=0.9):
        """
        Call after construct_PCA()
        Returns column names
        """
        self.num_PCs_over_explained_variance_ratio(cap_ratio=cap_ratio)
        loadings_df = self.loadings()

        col_dict = {col: 0 for col in self.df.columns}

        for i in range(self.num_features_select):
            indices = loadings_df[loadings_df[f"PC{i+1}"] >= loading_threshold].index
            for col in indices:
                col_dict[col] += loadings_df[f"PC{i+1}"].loc[col] * self.num_features_select / (i+1)

        sorted_dict = dict(sorted(col_dict.items(), key=lambda x: x[1], reverse=True))
        # return list(sorted_dict.items())[:self.num_features_select]
        return list(sorted_dict.keys())[:self.num_features_select]

if __name__ == "__main__":
    df = pd.read_csv("datasets/train_2000.csv")
    df = df[df["split"] == 0]

    pca_class = _PCA(df)
    pca_df = pca_class.construct_PCA()
    print("Total explained variance ratio:\n", sum(pca_class.explained_variance_ratio()), "\n")
    print("Loadings:\n", pca_class.loadings(), "\n")
    print("Selected features:\n", pca_class.select_features(), "\n")

    feature_df = pd.DataFrame(pca_class.select_features(), columns=["feature"])
    feature_df.to_csv("data_engineer/pca_selection.csv", index=False)
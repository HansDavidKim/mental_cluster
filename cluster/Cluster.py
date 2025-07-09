### PCA-family algorithms had the highest trustworthiness scores.
### Thus, we adopt PCA as the default algorithm for dimensionality reduction.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from cluster.DimReductionOption import DimReductionOptions
from ast import literal_eval


class Cluster:
    def __init__(self, 
                 n_components: int = 15
                 ):
        
        ### According to the experiment we conducted, PCA_family algorithms had the highest trustworthiness scores.
        ### Thus, we adopt PCA as the default algorithm for dimensionality reduction.
        self.pca = PCA(n_components=n_components)
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the PCA model and transform the data.

        Parameters:
        X (pd.DataFrame): The input data to be transformed.

        Returns:
        pd.DataFrame: The transformed data after PCA.
        """
        # NumPy 포맷의 문자열 벡터를 float 배열로 변환
        if isinstance(X['prob_logits'].iloc[0], str):
            X['prob_logits'] = X['prob_logits'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

        logits_matrix = np.vstack(X['prob_logits'].to_numpy())
        reduced = self.pca.fit_transform(logits_matrix)

        X = X.copy()
        X['reduced_logits'] = list(reduced)
        return X
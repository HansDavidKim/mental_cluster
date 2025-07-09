### PCA-family algorithms had the highest trustworthiness scores.
### Thus, we adopt PCA as the default algorithm for dimensionality reduction.
import pandas as pd
import numpy as np

from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from cluster.DimReductionOption import DimReductionOptions
from ast import literal_eval


class Cluster:
    def __init__(self, 
                 n_components: int = 10
                 ):
        
        self.se = SpectralEmbedding(
            n_components=n_components, 
            random_state=42, 
            n_neighbors=10, 
            eigen_solver='arpack'
        )
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the PCA model and transform the data.

        Parameters:
        X (pd.DataFrame): The input data to be transformed.

        Returns:
        pd.DataFrame: The transformed data after PCA.
        """
        if isinstance(X['prob_logits'].iloc[0], str):
            X['prob_logits'] = X['prob_logits'].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

        logits_matrix = np.vstack(X['prob_logits'].to_numpy())
        # Standardize the data
        scaler = StandardScaler()
        logits_matrix = scaler.fit_transform(logits_matrix)
        reduced = self.se.fit_transform(logits_matrix)

        X = X.copy()
        X['reduced_logits'] = list(reduced)
        return X
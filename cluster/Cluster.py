### PCA-family algorithms had the highest trustworthiness scores.
### Thus, we adopt PCA as the default algorithm for dimensionality reduction.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from cluster.DimReductionOption import DimReductionOption, DimReductionOptionType

class Cluster:
    def __init__(self, 
                 reduction_option: DimReductionOptionType = DimReductionOptionType.PCA,
                 n_components: int = 15
                 ):
        
        ### According to the experiment we conducted, PCA_family algorithms had the highest trustworthiness scores.
        ### Thus, we adopt PCA as the default algorithm for dimensionality reduction.
        self.reduction_option = reduction_option
        self.pca = PCA(n_components=n_components)
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the PCA model and transform the data.
        
        Parameters:
        X (pd.DataFrame): The input data to be transformed.
        
        Returns:
        pd.DataFrame: The transformed data after PCA.
        """
        logits_matrix = np.vstack(X['prob_logits'].to_numpy())
        reduced = self.pca.fit_transform(logits_matrix)

        X['reduced_logits'] = list(reduced)
        return X
### PCA-family algorithms had the highest trustworthiness scores.
### Thus, we adopt PCA as the default algorithm for dimensionality reduction.

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
        
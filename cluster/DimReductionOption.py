from enum import Enum, auto
from sklearn.decomposition import *
from sklearn.manifold import *

class DimReductionOptions(Enum):
    PCA = auto()
    INCREMENTAL_PCA = auto()
    SPARSE_PCA = auto()
    TRUNCATED_SVD = auto()
    KERNEL_PCA = auto()

    ISOMAP = auto()
    LOCALLY_LINEAR_EMBEDDING = auto()  
    MODIFIED_LLE = auto()
    HESSIAN_LLE = auto()
    LTSA = auto()
    SPECTRAL_EMBEDDING = auto()
    TSNE = auto()

    FAST_ICA = auto()
    FACTOR_ANALYSIS = auto()
    DICTIONARY_LEARNING = auto()
    MINI_BATCH_DICTIONARY_LEARNING = auto()
    NMF = auto()

DIM_REDUCTION_CONSTRUCTORS = {
    DimReductionOptions.PCA: PCA,
    DimReductionOptions.INCREMENTAL_PCA: IncrementalPCA,
    DimReductionOptions.SPARSE_PCA: SparsePCA,
    DimReductionOptions.TRUNCATED_SVD: TruncatedSVD,
    DimReductionOptions.KERNEL_PCA: KernelPCA,

    DimReductionOptions.ISOMAP: Isomap,
    DimReductionOptions.LOCALLY_LINEAR_EMBEDDING: lambda **kwargs: LocallyLinearEmbedding(method='standard', **kwargs),
    DimReductionOptions.MODIFIED_LLE: lambda **kwargs: LocallyLinearEmbedding(method='modified', **kwargs),
    DimReductionOptions.HESSIAN_LLE: lambda **kwargs: LocallyLinearEmbedding(method='hessian', **kwargs),
    DimReductionOptions.LTSA: lambda **kwargs: LocallyLinearEmbedding(method='ltsa', **kwargs),
    DimReductionOptions.SPECTRAL_EMBEDDING: SpectralEmbedding,
    DimReductionOptions.TSNE: TSNE,

    DimReductionOptions.FAST_ICA: FastICA,
    DimReductionOptions.FACTOR_ANALYSIS: FactorAnalysis,
    DimReductionOptions.DICTIONARY_LEARNING: DictionaryLearning,
    DimReductionOptions.MINI_BATCH_DICTIONARY_LEARNING: MiniBatchDictionaryLearning,
    DimReductionOptions.NMF: NMF,
}
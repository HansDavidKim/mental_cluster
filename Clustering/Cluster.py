### Clustering Module : Based on KMeans, Agglomerative Clustering
### Model : snunlp/KR-SBERT-V40K-klueNLI-augSTS

import torch
import numpy as np
import pandas as pd
import itertools

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Cluster:
    def __init__(self, model_name: str = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'):
        """
        Initialize the Cluster with a pre-trained SentenceTransformer model.
        
        :param model_name: Name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, data: pd.DataFrame, text_column: str = 'title_comment') -> np.ndarray:
        """
        Encode the text data into embeddings.
        
        :param data: DataFrame containing the text data.
        :param text_column: Column name containing the text to encode.
        :return: Numpy array of embeddings.
        """
        if text_column not in data.columns:
            raise ValueError(f"DataFrame must contain a '{text_column}' column.")
        
        texts = data[text_column].tolist()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            model_output = self.model(**inputs)
            embeddings = mean_pooling(model_output, inputs['attention_mask'])
        
        return embeddings.cpu().numpy()
    
    def cluster(self, embeddings: np.ndarray, method: str = 'kmeans') -> np.ndarray:
        """
        Perfrom clustering on the embeddings.
        :param embeddings: Numpy array of embeddings.
        :param method: Clustering method ('kmeans' or 'agglomerative').
        :return: Numpy array of cluster labels.
        """
        # Find the optimal number of clusters using grid search
        if method == 'kmeans':
            grid_search_params = {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20],
                'max_iter': [300, 500]
            }
            best_labels, best_score = self.grid_search_cluster(embeddings, method='kmeans', param_grid=grid_search_params)
        elif method == 'agglomerative':
            grid_search_params = {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'linkage': ['ward', 'complete', 'average', 'single']
            }
            best_labels, best_score = self.grid_search_cluster(embeddings, method='agglomerative', param_grid=grid_search_params)
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'agglomerative'.")
        
        return best_labels
    
    from tqdm import tqdm  # 맨 위에 추가

    def grid_search_cluster(self, embeddings: np.ndarray, method: str = 'kmeans', param_grid: dict = None) -> tuple:
        """
        Perform manual grid search for optimal clustering parameters using silhouette score.
        """
        best_score = -1
        best_labels = None

        # Generate all combinations of parameters
        keys = list(param_grid.keys())
        param_combinations = list(itertools.product(*[param_grid[k] for k in keys]))

        for values in tqdm(param_combinations, desc=f"Grid Search ({method})"):
            params = dict(zip(keys, values))

            if method == 'kmeans':
                model = KMeans(**params, random_state=42)
            elif method == 'agglomerative':
                if params.get('linkage') == 'ward':
                    model = AgglomerativeClustering(**params, metric='euclidean')
                else:
                    model = AgglomerativeClustering(**params)
            else:
                raise ValueError("Unsupported clustering method")

            try:
                labels = model.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
            except Exception as e:
                print(f"Skipping config {params} due to error: {e}")

        return best_labels, best_score
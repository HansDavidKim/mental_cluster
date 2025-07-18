import torch
import numpy as np
import pandas as pd
import itertools

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Cluster:
    def __init__(self, model_name: str = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'):
        """
        Initialize the Cluster with a pre-trained SentenceTransformer model.
        Automatically assigns GPU if available.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, data: pd.DataFrame, text_column: str = 'title_comment', batch_size: int = 128) -> np.ndarray:
        """
        Encode the text data into embeddings with optional batching and GPU support.

        :param data: DataFrame containing the text data.
        :param text_column: Column name containing the text to encode.
        :param batch_size: Batch size for encoding.
        :return: Numpy array of embeddings.
        """
        from tqdm import tqdm

        if text_column not in data.columns:
            raise ValueError(f"DataFrame must contain a '{text_column}' column.")

        texts = data[text_column].tolist()
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with SBERT"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                model_output = self.model(**inputs)
                embeddings = mean_pooling(model_output, inputs['attention_mask'])

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def cluster(self, embeddings: np.ndarray, method: str = 'kmeans') -> np.ndarray:
        """
        Perform clustering on the embeddings using the specified method.

        :param embeddings: Numpy array of embeddings.
        :param method: Clustering method ('kmeans' or 'agglomerative').
        :return: Numpy array of cluster labels.
        """
        if method == 'kmeans':
            grid_search_params = {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20],
                'max_iter': [300, 500]
            }
        elif method == 'agglomerative':
            grid_search_params = {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'linkage': ['ward', 'complete', 'average', 'single']
            }
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'agglomerative'.")

        best_labels, best_score = self.grid_search_cluster(embeddings, method=method, param_grid=grid_search_params)
        return best_labels

    def grid_search_cluster(self, embeddings: np.ndarray, method: str = 'kmeans', param_grid: dict = None) -> tuple:
        """
        Perform manual grid search for optimal clustering parameters using silhouette score.

        :param embeddings: Embedding vectors.
        :param method: Clustering method.
        :param param_grid: Dictionary of hyperparameter search space.
        :return: Tuple of (best_labels, best_score)
        """
        from tqdm import tqdm

        best_score = -1
        best_labels = None

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

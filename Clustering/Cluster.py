import torch
import numpy as np
import pandas as pd
import itertools

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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
        Supported methods: 'kmeans', 'agglomerative', 'dbscan'
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
        elif method == 'dbscan':
            grid_search_params = {
                'eps': [0.3, 0.5, 0.7, 1.0, 1.2],
                'min_samples': [3, 5, 10]
            }
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans', 'agglomerative', or 'dbscan'.")

        best_labels, best_score = self.grid_search_cluster(embeddings, method=method, param_grid=grid_search_params)
        print(f"✅ Best Silhouette Score: {best_score:.4f}")
        return best_labels

    def grid_search_cluster(self, embeddings: np.ndarray, method: str = 'kmeans', param_grid: dict = None) -> tuple:
        """
        Perform manual grid search for optimal clustering parameters using silhouette score.
        For DBSCAN, skip configs where only one cluster or all noise points.
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
            elif method == 'dbscan':
                model = DBSCAN(**params, metric='euclidean')
            else:
                raise ValueError("Unsupported clustering method")

            try:
                labels = model.fit_predict(embeddings)

                # For DBSCAN, skip cases where only one cluster or all points are noise
                if method == 'dbscan':
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters <= 1:
                        continue

                score = silhouette_score(embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
            except Exception as e:
                print(f"Skipping config {params} due to error: {e}")

        return best_labels, best_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def reduce_dimension(matrix: np.ndarray) -> np.ndarray:
    embedded_matrix = TSNE(
            n_components=2, 
            learning_rate='auto', 
            init='random', 
            perplexity=3, 
            random_state=42
        ).fit_transform(matrix)
    
    return embedded_matrix

def plot_clusters(embedded: np.ndarray, labels: np.ndarray, save_path: str = "tsne_clusters.png"):
    plt.figure(figsize=(10, 8))

    num_clusters = len(set(labels))
    for cluster_id in range(num_clusters):
        points = embedded[labels == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.6)

    plt.title("t-SNE Clustering Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./result/{save_path}')
    print(f"[✔] 시각화 저장 완료: {save_path}")
    plt.close()
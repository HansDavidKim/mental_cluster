import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 3D 지원

def reduce_dimension(matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    embedded_matrix = TSNE(
        n_components=n_components, 
        learning_rate='auto', 
        init='random', 
        perplexity=3, 
        random_state=42
    ).fit_transform(matrix)
    
    return embedded_matrix

def plot_clusters(
    embedded: np.ndarray, 
    labels: np.ndarray, 
    save_path: str = "tsne_clusters.png", 
    n_components: int = 3
):
    fig = plt.figure(figsize=(10, 8))

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id in range(len(set(labels))):
            points = embedded[labels == cluster_id]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Cluster {cluster_id}", alpha=0.6)
        ax.set_title("3D t-SNE Clustering Visualization")
    else:
        for cluster_id in range(len(set(labels))):
            points = embedded[labels == cluster_id]
            plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.6)
        plt.title("2D t-SNE Clustering Visualization")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./result/{save_path}')
    print(f"[✔] Visualization Finished: {save_path}")
    plt.close()

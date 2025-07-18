import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위해 필요

def reduce_dimension(matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Reduce the dimensionality of the given matrix using PCA.

    :param matrix: Input high-dimensional matrix.
    :param n_components: Target dimensionality (2 or 3).
    :return: PCA-reduced matrix.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(matrix)

def plot_clusters(
    embedded: np.ndarray,
    labels: np.ndarray,
    save_path: str = "pca_clusters.png",
    n_components: int = 3
):
    """
    Plot 2D or 3D scatter plot of clustered embeddings and save as image.

    :param embedded: Reduced embedding matrix (PCA).
    :param labels: Cluster labels for each point.
    :param save_path: Filename to save the plot.
    :param n_components: Dimension of the embedding (2 or 3).
    """
    os.makedirs("./result", exist_ok=True)
    fig = plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id in unique_labels:
            points = embedded[labels == cluster_id]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Cluster {cluster_id}", alpha=0.6)
        ax.set_title("3D PCA Clustering Visualization")
    else:
        for cluster_id in unique_labels:
            points = embedded[labels == cluster_id]
            plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", alpha=0.6)
        plt.title("2D PCA Clustering Visualization")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./result/{save_path}')
    print(f"[✔] Visualization Finished: {save_path}")
    plt.close()

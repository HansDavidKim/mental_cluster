import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cluster.Cluster as Cluster

def cluster_and_save(data: pd.DataFrame,
                     k_range=range(2, 11),
                     reduced_col: str = "reduced_logits",
                     save_csv_path: str = "clustered_data.csv",
                     save_img_path: str = "clustered_plot.png") -> None:
    """
    KMeans로 최적 군집 개수를 찾고, cluster 열을 추가한 후 CSV 및 시각화 이미지로 저장합니다.

    Args:
        data (pd.DataFrame): reduced_logits 열이 포함된 입력 데이터
        k_range (range): 시도할 클러스터 개수 범위
        reduced_col (str): 차원 축소 벡터가 들어있는 컬럼 이름
        save_csv_path (str): cluster 열이 추가된 CSV 저장 경로
        save_img_path (str): 시각화 이미지 저장 경로
    """
    df = data.copy()

    # 문자열 형태라면 파싱
    if isinstance(df[reduced_col].iloc[0], str):
        df[reduced_col] = df[reduced_col].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

    X = np.vstack(df[reduced_col].to_numpy())

    # 최적 k 찾기
    best_k = None
    best_score = -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    df["cluster"] = best_labels

    # CSV 저장
    df.to_csv(save_csv_path, index=False)
    print(f"[✓] Clustered CSV saved to: {save_csv_path}")

    # 시각화 저장
    plt.figure(figsize=(10, 8))
    num_clusters = best_k
    cmap = plt.get_cmap("tab10" if num_clusters <= 10 else "tab20")
    colors = cmap(np.linspace(0, 1, num_clusters))

    for i in range(num_clusters):
        cluster_points = X[df["cluster"] == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    label=f"Cluster {i}", color=colors[i], alpha=0.7)

    plt.title(f"KMeans Clustering (Best k = {best_k}, Silhouette = {best_score:.3f})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_img_path)
    print(f"[✓] Clustering plot saved to: {save_img_path}")

if __name__ == "__main__":
    data_path = './data/filtered_sentiment_data_분노.csv'
    n_components = 2
    perplexity = 30
    cluster_object = Cluster.Cluster(n_components=15)
    data = pd.read_csv(data_path)
    data = cluster_object.fit_transform(data)

    cluster_and_save(data, k_range=range(2, 11),
                     reduced_col="reduced_logits",
                     save_csv_path="clustered_data.csv",
                     save_img_path="clustered_plot.png")
import pandas as pd
import numpy as np
from cluster.DimReduction import train_reduction
from cluster.DimReductionOption import DimReductionOptions

def test_reduction(csv_path: str = "../data/labeled_sentiment_data.csv", n_trials: int = 250):
    """
    각 차원 축소 알고리즘에 대해 trustworthiness를 측정하고 출력합니다.

    Args:
        csv_path (str): CSV 경로 (prob_logits 열 필요)
        n_trials (int): Optuna 튜닝 횟수
    """
    # 1. CSV 로딩
    df = pd.read_csv(csv_path)
    if "prob_logits" not in df.columns:
        raise ValueError("❌ 'prob_logits' column not found in CSV")

    # 2. 알고리즘 순회 실험
    for algo in DimReductionOptions:
        print(f"\n[🔎 {algo.name}]")
        try:
            model, score, params = train_reduction(algo, df, n_trials=n_trials)
            n_components = params.get("n_components", "?")
            print(f"✅ Trustworthiness: {score:.4f}")
            print(f"📉 Best dim (n_components): {n_components}")
            print(f"⚙️  Best params: {params}")

        except Exception as e:
            print(f"❌ Failed for {algo.name}: {str(e)}")

# 모듈 실행 시 직접 동작
if __name__ == "__main__":
    test_reduction()

import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from .DimReductionOption import DimReductionOptions, DIM_REDUCTION_CONSTRUCTORS

def train_reduction(algorithm: DimReductionOptions, 
                    data: pd.DataFrame, 
                    n_trials: int = 300) -> tuple:
    """
    주어진 차원 축소 알고리즘에 대해 Optuna를 사용하여
    trustworthiness score를 최대화하는 최적 파라미터를 탐색합니다.

    Args:
        algorithm (DimReductionOptions): 차원 축소 알고리즘 Enum
        data (pd.DataFrame): 'prob_logits' 열이 포함된 입력 데이터프레임
        n_trials (int): Optuna 최적화 시도 횟수

    Returns:
        tuple: (best_model, best_score, best_params)
    """
    # 1. prob_logits 추출 및 파싱
    X = data["prob_logits"].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))
    X = np.vstack(X.values)

    ModelFactory = DIM_REDUCTION_CONSTRUCTORS[algorithm]

    def objective(trial):
        n_components = trial.suggest_int("n_components", 2, min(X.shape[1], 15))
        use_scaler = trial.suggest_categorical("use_scaler", [True, False])
        X_scaled = StandardScaler().fit_transform(X) if use_scaler else X

        kwargs = {"n_components": n_components}

        # 알고리즘별 추가 하이퍼파라미터
        if algorithm == DimReductionOptions.TSNE:
            kwargs["perplexity"] = trial.suggest_float("perplexity", 5.0, 50.0)
            kwargs["init"] = "pca"
            kwargs["random_state"] = 42
        elif algorithm == DimReductionOptions.KERNEL_PCA:
            kwargs["kernel"] = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"])
        elif algorithm in [
            DimReductionOptions.ISOMAP,
            DimReductionOptions.LOCALLY_LINEAR_EMBEDDING,
            DimReductionOptions.MODIFIED_LLE,
            DimReductionOptions.HESSIAN_LLE,
            DimReductionOptions.LTSA,
        ]:
            kwargs["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)

        try:
            model = ModelFactory(**kwargs)
            X_reduced = model.fit_transform(X_scaled)
            score = trustworthiness(X_scaled, X_reduced, n_neighbors=5)
            return score
        except Exception:
            return 0.0

    # 2. Optuna 최적화 수행
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    use_scaler = best_params.pop("use_scaler")
    X_final = StandardScaler().fit_transform(X) if use_scaler else X

    best_model = ModelFactory(**best_params)
    best_model.fit(X_final)

    return best_model, study.best_value, study.best_params

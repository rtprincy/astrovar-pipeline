
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Any, Tuple, List
# from sklearn.manifold import TSNE
from openTSNE import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

def remove_correlated(df: pd.DataFrame, threshold: float=0.95) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=drop_cols), drop_cols

def optimize_tsne(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    best_score=-np.inf; best_emb=None; best_p=None
    for perp in params.get("perplexity_grid",[5]):
        for lr in params.get("learning_rate_grid",[200]):
            print("perplexity: ",perp)
            emb = TSNE(n_components=2, perplexity=perp, learning_rate=lr, n_iter=params.get("n_iter",1000),
                       metric=params.get("metric","euclidean"), random_state=params.get("random_state",42)).fit(X)
            # No labels yet; proxy quality via KNN preservation is expensive; use clusterability via GMM+BIC as heuristic
            gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=42).fit(emb)
            score = -gmm.bic(emb)
            if score > best_score:
                best_score=score; best_emb=emb; best_p={"perplexity":perp, "learning_rate":lr}
    return best_emb, best_p

def optimize_gmm(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], GaussianMixture]:
    best_bic=np.inf; best_labels=None; best_cfg=None; best_model=None
    for k in params.get("n_components_grid",[8,10,12]):
        gmm = GaussianMixture(n_components=k, covariance_type=params.get("covariance_type","full"),
                              n_init=params.get("n_init",5), random_state=params.get("random_state",42))
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic; best_labels = gmm.predict(X); best_cfg={"n_components":k}; best_model=gmm
    return best_labels, best_cfg, best_model

def rf_prune_features(X: pd.DataFrame, y: np.ndarray, top_k: int=60, random_state: int=42) -> Tuple[pd.DataFrame, pd.Series]:
    rf = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    keep = importances.head(top_k).index
    return X[keep], importances


from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def plot_tsne(emb: np.ndarray, labels: np.ndarray, path: str):
    plt.figure()
    sc = plt.scatter(emb[:,0], emb[:,1], c=labels, s=6)
    plt.title("t-SNE + GMM clusters")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_importances(importances: pd.Series, path: str):
    plt.figure()
    importances.head(30).plot(kind="barh")
    plt.title("Top feature importances (RF)")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

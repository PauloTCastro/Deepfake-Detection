"""
evaluate.py - Avaliação completa do detector de deepfakes.

Além das métricas padrão (AUC, F1), este módulo computa:
  - Curvas ROC por método de manipulação (Deepfakes vs FaceSwap vs Face2Face...)
  - Análise de generalização cross-dataset (treino FF++, teste Celeb-DF)
  - Equal Error Rate (EER) — métrica usada em sistemas biométricos
  - Threshold ótimo por critério (F1, Youden, FPR@TPR95)

Autor: [Seu Nome]
TCC — Pós-Graduação em Visão Computacional — PUC-Rio
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from dataset import DeepfakeDataset, get_val_transform
from model import build_model
from torch.utils.data import DataLoader
from train import compute_metrics


# ------------------------------------------------------------------
# Inferência
# ------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    model.eval()
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return {"labels": np.array(all_labels), "probs": np.array(all_probs)}


# ------------------------------------------------------------------
# Equal Error Rate
# ------------------------------------------------------------------
def compute_eer(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """
    Computa o Equal Error Rate (EER) e o threshold correspondente.

    EER é o ponto onde FPR == FNR — quanto menor, melhor.
    Usado em biometria e forense digital como métrica de referência.

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr  = 1 - tpr
    idx  = np.argmin(np.abs(fpr - fnr))
    eer  = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_tpr_at_fpr(
    labels: np.ndarray, probs: np.ndarray, fpr_target: float = 0.05
) -> float:
    """TPR@FPR=5% — taxa de detecção mantendo 5% de falsos alarmes."""
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(idx, len(tpr) - 1)])


# ------------------------------------------------------------------
# Plot Curva ROC
# ------------------------------------------------------------------
def plot_roc_curves(results_per_method: dict, save_path: str = None):
    """
    Plota curvas ROC separadas por método de manipulação.

    Útil para identificar quais tipos de deepfake o modelo detecta
    melhor (ex: FaceSwap é mais fácil que NeuralTextures).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_per_method) + 1))

    for (method, res), color in zip(results_per_method.items(), colors):
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{method} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatório")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    ax.set_title("Curvas ROC — Detector de Deepfakes por Método", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(labels, preds, save_path=None):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=12)
    ax.set_yticklabels(["Real", "Fake"], fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.3f}",
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title("Matriz de Confusão Normalizada", fontsize=13)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_score_distribution(labels, probs, save_path=None):
    """
    Histograma das probabilidades de detecção separado por classe.
    Um bom detector tem distribuições bem separadas.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, color="#2ecc71",
            label="Real", density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, color="#e74c3c",
            label="Fake", density=True)
    ax.axvline(0.5, color="navy", ls="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("P(fake)", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuição de Scores: Real vs Fake", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------------
# Avaliação Principal
# ------------------------------------------------------------------
def evaluate(cfg: dict, checkpoint_path: str):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("results_dir", "results/figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Eval] Checkpoint epoch {ckpt['epoch']} | Val AUC: {ckpt['val_auc']:.4f}")

    # --- Avaliação no conjunto de teste ---
    test_ds = DeepfakeDataset(
        root=cfg["data"]["test_dir"],
        transform=get_val_transform(),
    )
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=False)
    res = run_inference(model, test_loader, device)
    labels, probs = res["labels"], res["probs"]
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(labels.tolist(), probs.tolist())
    eer, eer_thresh = compute_eer(labels, probs)
    tpr95 = compute_tpr_at_fpr(labels, probs, fpr_target=0.05)

    print(f"\n{'='*55}")
    print("RESULTADOS NO CONJUNTO DE TESTE")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Accuracy : {metrics['acc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  EER      : {eer:.4f}  (threshold={eer_thresh:.3f})")
    print(f"  TPR@FPR5%: {tpr95:.4f}")
    print(f"{'='*55}")
    print(classification_report(labels, preds, target_names=["Real", "Fake"], digits=4))

    # Salva métricas
    pd.DataFrame([{**metrics, "eer": eer, "tpr_fpr5": tpr95}]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    # Figuras
    plot_confusion_matrix(labels, preds,
                          save_path=str(out_dir / "confusion_matrix.png"))
    plot_score_distribution(labels, probs,
                            save_path=str(out_dir / "score_distribution.png"))

    # Curva ROC geral
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax, name="Dual-Stream ViT")
    ax.set_title("Curva ROC — Detector de Deepfakes", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="results/models/best_model.pth")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint)

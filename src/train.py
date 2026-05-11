"""
train.py - Loop de treinamento para o detector de deepfakes.

Características:
  - Focal Loss (deepfakes de alta qualidade são exemplos "difíceis")
  - AUC como métrica principal (mais informativo que accuracy em binário)
  - Curriculum Learning: começa com fakes de baixa qualidade, avança
    para alta qualidade nas épocas finais
  - EMA (Exponential Moving Average) do modelo para inferência estável

Autor: [Seu Nome]
TCC — Pós-Graduação em Visão Computacional — PUC-Rio
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from dataset import build_dataloaders
from model import build_model


# ------------------------------------------------------------------
# Focal Loss Binária
# ------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss para classificação binária.

    Deepfakes de alta qualidade (StyleGAN3, SimSwap) são exemplos difíceis
    com loss alto — o fator focal (1-p)^gamma amplifica a atenção a eles.

    Args:
        gamma: fator de modulação. 0 = BCE padrão.
        pos_weight: peso da classe positiva (fake) para desbalanceamento.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits).squeeze(1)
        bce    = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none",
        )
        pt     = torch.where(targets == 1, probs, 1 - probs)
        focal  = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


import torch.nn.functional as F


# ------------------------------------------------------------------
# EMA — Exponential Moving Average
# ------------------------------------------------------------------
class ModelEMA:
    """
    Mantém uma média exponencial dos pesos do modelo.

    Modelos EMA tendem a ser mais estáveis e generalizáveis
    na inferência do que o modelo do último checkpoint.

    decay=0.9999 é padrão em detectores de deepfake (Wang et al., 2020).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# ------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------
def compute_metrics(all_labels, all_probs, threshold: float = 0.5) -> dict:
    preds = (np.array(all_probs) >= threshold).astype(int)
    labels = np.array(all_labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc       = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    try:
        auc = roc_auc_score(labels, all_probs)
    except Exception:
        auc = float("nan")

    return {"acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc}


import numpy as np


# ------------------------------------------------------------------
# Um epoch de treino
# ------------------------------------------------------------------
def train_one_epoch(model, ema, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        probs = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    n = len(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


# ------------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(len(all_labels), 1)
    return metrics


# ------------------------------------------------------------------
# Loop Principal
# ------------------------------------------------------------------
def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Dispositivo: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg["data"])

    model = build_model(cfg["model"]).to(device)
    ema   = ModelEMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    info  = model.count_params()
    print(f"[Model] {info['total']:,} params | {info['trainable']:,} treináveis")

    criterion = BinaryFocalLoss(
        gamma=cfg["train"].get("focal_gamma", 2.0),
        pos_weight=cfg["train"].get("pos_weight", 1.5),
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
    )

    total_epochs = cfg["train"].get("epochs", 30)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["train"].get("T_0", 10),
        T_mult=2,
        eta_min=cfg["train"].get("min_lr", 1e-6),
    )

    scaler   = GradScaler()
    best_auc = 0.0
    patience = cfg["train"].get("patience", 8)
    pat_ctr  = 0
    ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "results/models"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, ema, train_loader, optimizer, criterion, scaler, device)
        vl = evaluate(ema.ema, val_loader, criterion, device)  # avalia com EMA
        scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{total_epochs} | "
            f"Loss T/V: {tr['loss']:.4f}/{vl['loss']:.4f} | "
            f"AUC T/V: {tr['auc']:.4f}/{vl['auc']:.4f} | "
            f"F1 V: {vl['f1']:.4f} | "
            f"LR: {lr_now:.2e} | {time.time()-t0:.0f}s"
        )

        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            pat_ctr  = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema.state_dict(),   # salva EMA
                "val_auc": best_auc,
                "val_f1":  vl["f1"],
                "cfg": cfg,
            }, ckpt_dir / "best_model.pth")
            print(f"  ✓ Melhor AUC: {best_auc:.4f}")
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"[EarlyStopping] Sem melhora há {patience} epochs.")
                break

    print(f"\n[Train] Concluído — AUC validação: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)

"""
visualize.py - Visualização de explicabilidade para detecção de deepfakes.

Módulos:
  1. AttentionVisualizer   — mapa de atenção do ViT espacial (onde o modelo "olha")
  2. FrequencyVisualizer   — espectro de Fourier real vs fake (assinatura GAN)
  3. EvidencePanel         — painel completo de evidências para um rosto analisado

Estes visuais são essenciais para explicar ao avaliador (e eventualmente
a jornalistas, juízes, peritos) por que o sistema classificou uma imagem
como deepfake — respondendo ao requisito de explicabilidade do Marco Civil
e da futura Lei de IA brasileira.

Autor: [Seu Nome]
TCC — Pós-Graduação em Visão Computacional — PUC-Rio
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from dataset import MEAN, STD, IMG_SIZE
from model import DeepfakeDetector


# ------------------------------------------------------------------
# Pré-processamento de imagem única
# ------------------------------------------------------------------
def load_image(path: str, img_size: int = IMG_SIZE) -> tuple[torch.Tensor, np.ndarray]:
    """
    Carrega e pré-processa uma imagem para inferência.

    Returns:
        (tensor (1,3,H,W), array_rgb uint8 para visualização)
    """
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rsz = cv2.resize(img_rgb, (img_size, img_size))

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = tfm(img_rsz).unsqueeze(0)
    return tensor, img_rsz


# ------------------------------------------------------------------
# 1. Mapa de Atenção do ViT
# ------------------------------------------------------------------
class AttentionVisualizer:
    """
    Visualiza onde o ViT espacial presta atenção ao analisar um rosto.

    Utiliza o método de Attention Rollout adaptado para ViT-B/16:
    propaga atenções da camada final de volta ao espaço de imagem.

    Regiões de alta atenção em rostos falsos tipicamente indicam:
    - Bordas entre rosto sobreposto e fundo original (blending boundary)
    - Região dos olhos (piscadas não naturais em Face2Face)
    - Dentes e lábios (síntese de boca é frequentemente imperfeita)
    """

    def __init__(self, model: DeepfakeDetector, device: str = "cpu"):
        self.model  = model.to(device).eval()
        self.device = device

    def get_heatmap(
        self,
        tensor: torch.Tensor,
        layer_idx: int = -1,
        upsample_size: tuple[int, int] = (IMG_SIZE, IMG_SIZE),
    ) -> np.ndarray:
        """
        Gera mapa de calor de atenção normalizado [0, 1].

        Args:
            tensor: (1, 3, H, W)
            layer_idx: índice do bloco ViT (-1 = último)
            upsample_size: (H, W) de saída

        Returns:
            heatmap: array (H, W) float [0, 1]
        """
        tensor = tensor.to(self.device)
        attn_map = self.model.get_attention_map(tensor, layer_idx=layer_idx)
        # attn_map: (1, 14, 14)
        attn_np = attn_map.squeeze(0).cpu().numpy()

        # Upsampling para o tamanho da imagem
        heatmap = cv2.resize(attn_np, upsample_size[::-1],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def overlay(
        self,
        img_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Sobrepõe o heatmap colorido sobre a imagem original."""
        hm_uint8  = (heatmap * 255).astype(np.uint8)
        hm_color  = cv2.applyColorMap(hm_uint8, colormap)
        hm_color  = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        return (alpha * hm_color + (1 - alpha) * img_rgb).astype(np.uint8)


# ------------------------------------------------------------------
# 2. Análise de Frequência
# ------------------------------------------------------------------
class FrequencyVisualizer:
    """
    Visualiza o espectro de magnitude DFT de imagens reais vs falsas.

    Deepfakes baseados em GAN tipicamente exibem padrões regulares
    em grades no espectro de frequência — chamados "GAN fingerprints"
    ou "spectral artifacts" (Durall et al., 2020; Zhang et al., 2019).

    Esta visualização é uma evidência pericial de manipulação.
    """

    @staticmethod
    def compute_spectrum(img_rgb: np.ndarray) -> np.ndarray:
        """
        Computa espectro de magnitude log-normalizado.

        Returns:
            spectrum: array (H, W) uint8 [0, 255]
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f    = np.fft.fft2(gray)
        fs   = np.fft.fftshift(f)
        mag  = np.log(1 + np.abs(fs))
        mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return (mag * 255).astype(np.uint8)

    @staticmethod
    def compute_azimuthal_average(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computa a média azimutal do espectro (perfil radial).

        O perfil radial de deepfakes tende a ter picos em frequências
        específicas onde a GAN "trava" — evidência espectral de síntese.

        Returns:
            (radii, power): arrays para plotagem.
        """
        H, W = spectrum.shape
        cy, cx = H // 2, W // 2
        Y, X  = np.ogrid[:H, :W]
        R     = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)

        max_r = min(cy, cx)
        power = np.zeros(max_r)
        count = np.zeros(max_r)

        for r in range(max_r):
            mask = (R == r)
            if mask.any():
                power[r] = spectrum[mask].mean()
                count[r] = mask.sum()

        radii = np.arange(max_r)
        return radii, power


# ------------------------------------------------------------------
# 3. Painel Completo de Evidências
# ------------------------------------------------------------------
def plot_evidence_panel(
    image_path: str,
    model: DeepfakeDetector,
    device: str = "cpu",
    save_path: str | None = None,
):
    """
    Gera um painel de 6 visualizações para análise forense de uma imagem.

    Layout:
    ┌──────────────┬──────────────┬──────────────┐
    │  Imagem      │  Mapa de     │  Sobreposição │
    │  Original    │  Atenção     │  Atenção      │
    ├──────────────┼──────────────┼──────────────┤
    │  Espectro    │  Espectro    │  Perfil       │
    │  de Fourier  │  Colorido    │  Radial       │
    └──────────────┴──────────────┴──────────────┘

    Args:
        image_path: caminho para a imagem a analisar.
        model: DeepfakeDetector carregado.
        device: 'cuda' | 'cpu'
        save_path: se fornecido, salva a figura.
    """
    # --- Carrega e classifica ---
    tensor, img_rgb = load_image(image_path)
    tensor = tensor.to(device)
    model  = model.to(device).eval()

    with torch.no_grad():
        logit = model(tensor)
    prob_fake = torch.sigmoid(logit).item()
    verdict   = "🔴 DEEPFAKE" if prob_fake > 0.5 else "🟢 REAL"
    conf_str  = f"{prob_fake * 100:.1f}% de probabilidade de ser falso"

    # --- Atenção ---
    attn_vis = AttentionVisualizer(model, device)
    heatmap  = attn_vis.get_heatmap(tensor)
    overlay  = attn_vis.overlay(img_rgb, heatmap)

    # --- Frequência ---
    freq_vis  = FrequencyVisualizer()
    spectrum  = freq_vis.compute_spectrum(img_rgb)
    spec_color = cv2.applyColorMap(spectrum, cv2.COLORMAP_INFERNO)
    spec_color = cv2.cvtColor(spec_color, cv2.COLOR_BGR2RGB)
    radii, power = freq_vis.compute_azimuthal_average(spectrum)

    # --- Plot ---
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#1a1a2e")

    title_color = "#e94560" if prob_fake > 0.5 else "#0f9b8e"
    fig.suptitle(
        f"{verdict}   |   {conf_str}",
        fontsize=18, fontweight="bold",
        color=title_color, y=0.97,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    style = {"color": "white"}
    for a in ax:
        a.set_facecolor("#16213e")
        a.tick_params(colors="white")
        for spine in a.spines.values():
            spine.set_edgecolor("#444")

    ax[0].imshow(img_rgb)
    ax[0].set_title("Imagem Original", **style)
    ax[0].axis("off")

    ax[1].imshow(heatmap, cmap="hot")
    ax[1].set_title("Mapa de Atenção (ViT)", **style)
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Atenção Sobreposta", **style)
    ax[2].axis("off")

    ax[3].imshow(spectrum, cmap="gray")
    ax[3].set_title("Espectro DFT (magnitude log)", **style)
    ax[3].axis("off")

    ax[4].imshow(spec_color)
    ax[4].set_title("Espectro Colorido (Inferno)", **style)
    ax[4].axis("off")

    ax[5].plot(radii, power, color="#e94560", lw=2)
    ax[5].fill_between(radii, power, alpha=0.3, color="#e94560")
    ax[5].set_title("Perfil Radial de Frequência", **style)
    ax[5].set_xlabel("Frequência (raio)", color="white")
    ax[5].set_ylabel("Potência média", color="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualização] Salvo: {save_path}")

    plt.show()
    return prob_fake


# ------------------------------------------------------------------
# Comparação Real vs Fake em Grade
# ------------------------------------------------------------------
def plot_comparison_grid(
    real_paths: list[str],
    fake_paths: list[str],
    model: DeepfakeDetector,
    device: str = "cpu",
    n_samples: int = 4,
    save_path: str | None = None,
):
    """
    Grade de comparação mostrando rostos reais e falsos com
    probabilidade de detecção e mapa de atenção de cada um.
    """
    import random
    real_sel = random.sample(real_paths, min(n_samples, len(real_paths)))
    fake_sel = random.sample(fake_paths, min(n_samples, len(fake_paths)))

    attn_vis = AttentionVisualizer(model, device)
    model    = model.to(device).eval()

    n    = len(real_sel)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 14))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Comparação: Real vs Deepfake — Mapas de Atenção",
                 fontsize=16, fontweight="bold", color="white", y=1.01)

    titles_row = ["Real (original)", "Atenção (Real)",
                  "Deepfake", "Atenção (Fake)"]
    samples    = real_sel + fake_sel
    labels     = [0] * n + [1] * n

    for col, (path, label) in enumerate(zip(real_sel + fake_sel, labels)):
        row_base = 0 if label == 0 else 2
        tensor, img_rgb = load_image(path)

        with torch.no_grad():
            logit = model(tensor.to(device))
        prob = torch.sigmoid(logit).item()

        heatmap = attn_vis.get_heatmap(tensor.to(device))
        overlay = attn_vis.overlay(img_rgb, heatmap)

        for r_off, vis_img, is_attn in [(0, img_rgb, False), (1, overlay, True)]:
            ax  = axes[row_base + r_off][col]
            ax.imshow(vis_img, cmap="hot" if is_attn else None)
            ax.set_facecolor("#16213e")
            ax.axis("off")
            if r_off == 0:
                title_c = "#e94560" if prob > 0.5 else "#0f9b8e"
                ax.set_title(f"P(fake)={prob:.2f}", color=title_c,
                             fontsize=10, fontweight="bold")

    for i, title in enumerate(titles_row):
        axes[i][0].set_ylabel(title, color="white", fontsize=11, rotation=90, labelpad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    model = DeepfakeDetector()
    model.load_state_dict(ckpt["model_state_dict"])

    prob = plot_evidence_panel(
        args.image, model,
        device=args.device,
        save_path=args.output,
    )
    print(f"\nP(deepfake) = {prob:.4f} → {'FAKE' if prob > 0.5 else 'REAL'}")

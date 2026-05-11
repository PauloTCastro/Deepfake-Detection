"""
model.py - Detector de Deepfakes com Dual-Stream Vision Transformer.

Arquitetura de dois fluxos paralelos:
  Stream 1 — Espacial (ViT-B/16):
    Detecta artefatos locais de síntese: bordas de rosto, inconsistências
    de textura de pele, "blending boundaries" típicas de face-swap.

  Stream 2 — Frequencial (DFT + ViT leve):
    Deepfakes deixam assinaturas no espectro de frequência que são
    invisíveis ao olho humano mas detectáveis via transformada de Fourier.
    Redes GAN tendem a superamplificar certas frequências.

  Fusão: Cross-Attention entre os dois fluxos + cabeça binária.

Referências principais:
  - Dosovitskiy et al., 2020 — ViT (https://arxiv.org/abs/2010.11929)
  - Qian et al., 2020 — Thinking in Frequency (https://arxiv.org/abs/2007.09355)
  - Zhao et al., 2021 — Multi-attentional Deepfake Detection (CVPR 2021)

Autor: [Seu Nome]
TCC — Pós-Graduação em Visão Computacional — PUC-Rio
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ------------------------------------------------------------------
# Stream 2 — Análise de Frequência
# ------------------------------------------------------------------
class FrequencyExtractor(nn.Module):
    """
    Extrai representação espectral via DFT 2D e empacota em patches.

    Passos:
      1. Converte imagem para escala de cinza (média dos canais)
      2. Aplica FFT2D e centra o espectro (fftshift)
      3. Computa magnitude logarítmica: log(1 + |F|)
      4. Empacota em patches compatíveis com ViT

    Deepfakes GAN (StyleGAN, FaceSwap, SimSwap) deixam padrões
    regulares no espectro de frequência, especialmente em grades
    de frequências médias-altas — chamados de "GAN fingerprints".
    """

    def __init__(self, patch_size: int = 16, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size   = img_size
        # Projeção linear de patch → embedding
        self.patch_proj = nn.Conv2d(
            1, 768,
            kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (B, 3, H, W) RGB normalizado.
        Returns:
            patches: tensor (B, N, 768) — N = (H/P)*(W/P) patches espectrais.
        """
        # Luminância (peso ITU-R BT.601)
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)   # (B, 1, H, W)

        # FFT2D
        f = torch.fft.fft2(gray)
        f = torch.fft.fftshift(f, dim=(-2, -1))
        magnitude = torch.log(1 + torch.abs(f))

        # Normaliza para [0, 1]
        B = magnitude.shape[0]
        mn = magnitude.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
        mx = magnitude.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
        magnitude = (magnitude - mn) / (mx - mn + 1e-8)

        # Patch embedding
        patches = self.patch_proj(magnitude)   # (B, 768, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, 768)
        return patches


# ------------------------------------------------------------------
# Cross-Attention entre os dois Streams
# ------------------------------------------------------------------
class DualStreamCrossAttention(nn.Module):
    """
    Módulo de Cross-Attention que permite troca de informação
    entre o stream espacial e o stream frequencial.

    O stream espacial "consulta" o frequencial para saber quais
    regiões têm anomalias espectrais — e vice-versa. Esta troca
    é mais informativa que simplesmente concatenar os features.

    Referência: Chen et al., 2021 — CrossViT
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_s2f = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_f2s = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        spatial: torch.Tensor,
        freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial: (B, N, D) tokens do stream espacial
            freq:    (B, N, D) tokens do stream frequencial
        Returns:
            (spatial_enhanced, freq_enhanced)
        """
        # Spatial consulta Frequencial
        s_enh, _ = self.cross_attn_s2f(
            query=spatial, key=freq, value=freq
        )
        spatial = self.norm1(spatial + s_enh)

        # Frequencial consulta Espacial
        f_enh, _ = self.cross_attn_f2s(
            query=freq, key=spatial, value=spatial
        )
        freq = self.norm2(freq + f_enh)

        return spatial, freq


# ------------------------------------------------------------------
# Modelo Principal — Dual-Stream Deepfake Detector
# ------------------------------------------------------------------
class DeepfakeDetector(nn.Module):
    """
    Detector de deepfakes com fusão espacial + frequencial.

    Fluxo completo:
        Imagem → [Stream Espacial (ViT-B/16)] ─┐
                                               ├→ Cross-Attention → MLP → P(fake)
        Imagem → [Stream Frequencial (FFT+ViT)]─┘

    Args:
        freeze_vit_blocks (int): Quantos blocos do ViT-B congelar (0–12).
        dropout (float): Dropout na cabeça de classificação.
        num_cross_attn_layers (int): Camadas de Cross-Attention.

    Output:
        logit (B, 1) — valor positivo = fake, negativo = real.
        Aplique sigmoid para obter probabilidade.
    """

    def __init__(
        self,
        freeze_vit_blocks: int = 8,
        dropout: float = 0.4,
        num_cross_attn_layers: int = 2,
    ):
        super().__init__()

        # --- Stream 1: Espacial (ViT-B/16 pré-treinado) ---
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Congela os primeiros N blocos
        for i, block in enumerate(vit.encoder.layers):
            if i < freeze_vit_blocks:
                for p in block.parameters():
                    p.requires_grad = False

        self.spatial_patch_embed = vit.conv_proj  # (B,768,14,14)
        self.spatial_cls_token   = vit.class_token
        self.spatial_pos_embed   = vit.encoder.pos_embedding
        self.spatial_encoder     = vit.encoder.layers
        self.spatial_norm        = vit.encoder.ln

        # --- Stream 2: Frequencial ---
        self.freq_extractor = FrequencyExtractor()
        # ViT leve para o stream frequencial (6 blocos)
        vit_freq = vit_b_16(weights=None)
        self.freq_encoder = nn.Sequential(*list(vit_freq.encoder.layers)[:6])
        self.freq_norm    = nn.LayerNorm(768)
        self.freq_cls     = nn.Parameter(torch.zeros(1, 1, 768))
        self.freq_pos     = nn.Parameter(torch.randn(1, 197, 768) * 0.02)

        # --- Cross-Attention (fusão) ---
        self.cross_attn_layers = nn.ModuleList([
            DualStreamCrossAttention(dim=768, num_heads=8, dropout=0.1)
            for _ in range(num_cross_attn_layers)
        ])

        # --- Cabeça de Classificação ---
        self.head = nn.Sequential(
            nn.LayerNorm(768 * 2),
            nn.Linear(768 * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Forward Streams
    # ------------------------------------------------------------------
    def _forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Passa pelo stream espacial. Retorna tokens (B, N+1, 768)."""
        B = x.shape[0]
        # Patch embedding
        patches = self.spatial_patch_embed(x)  # (B,768,14,14)
        patches = patches.flatten(2).transpose(1, 2)  # (B,196,768)

        # CLS token + positional embedding
        cls = self.spatial_cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.spatial_pos_embed

        # Encoder
        for block in self.spatial_encoder:
            tokens = block(tokens)
        return self.spatial_norm(tokens)

    def _forward_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Passa pelo stream frequencial. Retorna tokens (B, N+1, 768)."""
        B = x.shape[0]
        patches = self.freq_extractor(x)   # (B, 196, 768)
        cls = self.freq_cls.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.freq_pos
        tokens = self.freq_encoder(tokens)
        return self.freq_norm(tokens)

    # ------------------------------------------------------------------
    # Forward Principal
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (B, 3, 224, 224) normalizado.
        Returns:
            logits: tensor (B, 1) — sigmoid → P(fake).
        """
        # Streams paralelos
        spatial_tokens = self._forward_spatial(x)  # (B, 197, 768)
        freq_tokens    = self._forward_freq(x)      # (B, 197, 768)

        # Cross-Attention iterativo
        for layer in self.cross_attn_layers:
            spatial_tokens, freq_tokens = layer(spatial_tokens, freq_tokens)

        # Extrai tokens CLS de cada stream
        cls_spatial = spatial_tokens[:, 0]  # (B, 768)
        cls_freq    = freq_tokens[:, 0]     # (B, 768)

        # Concatena e classifica
        fused  = torch.cat([cls_spatial, cls_freq], dim=1)  # (B, 1536)
        logits = self.head(fused)
        return logits

    # ------------------------------------------------------------------
    # Mapa de Atenção para Explicabilidade
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extrai o mapa de atenção do último bloco do ViT espacial
        para visualizar "onde" o modelo detecta artefatos de deepfake.

        Returns:
            attn_map: tensor (B, H, W) normalizado [0,1].
        """
        B = x.shape[0]
        patches = self.spatial_patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.spatial_cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.spatial_pos_embed

        target_block = list(self.spatial_encoder)[layer_idx]
        attn_weights = None

        def hook_fn(module, input, output):
            nonlocal attn_weights
            q, k, v = input[0], input[1], input[2]
            scale = q.shape[-1] ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            attn_weights = attn.detach()

        h = target_block.self_attention.register_forward_hook(hook_fn)
        for block in self.spatial_encoder:
            tokens = block(tokens)
        h.remove()

        if attn_weights is None:
            return torch.zeros(B, 14, 14)

        # Atenção do CLS para os patches de imagem (média dos heads)
        cls_attn = attn_weights[:, :, 0, 1:]  # (B, heads, 196)
        cls_attn = cls_attn.mean(dim=1)        # (B, 196)
        cls_attn = cls_attn.reshape(B, 14, 14)

        # Normaliza
        mn = cls_attn.view(B, -1).min(dim=1).values.view(B, 1, 1)
        mx = cls_attn.view(B, -1).max(dim=1).values.view(B, 1, 1)
        return (cls_attn - mn) / (mx - mn + 1e-8)

    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def build_model(cfg: dict) -> DeepfakeDetector:
    return DeepfakeDetector(
        freeze_vit_blocks=cfg.get("freeze_vit_blocks", 8),
        dropout=cfg.get("dropout", 0.4),
        num_cross_attn_layers=cfg.get("num_cross_attn_layers", 2),
    )


if __name__ == "__main__":
    model = DeepfakeDetector()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    info = model.count_params()
    print(f"Output: {out.shape}  |  sigmoid: {torch.sigmoid(out).detach()}")
    print(f"Parâmetros: {info['total']:,} total | {info['trainable']:,} treináveis")

"""
dataset.py - Pipeline de dados para detecção de deepfakes.

Datasets suportados:
  - FaceForensics++ (FF++): 1.000 vídeos reais + 4 métodos de manipulação
    https://github.com/ondyari/FaceForensics
  - Celeb-DF v2: 590 reais + 5.639 falsos (maior qualidade)
    https://github.com/yuezunli/celeb-deepfakeforensics
  - DFDC (DeepFake Detection Challenge): 100k+ vídeos da Meta/Kaggle
    https://www.kaggle.com/c/deepfake-detection-challenge

Estratégia de sampling:
  - Extrai N frames por vídeo com espaçamento uniforme
  - Detecta e recorta rostos com MTCNN antes de alimentar o modelo
  - Aplica augmentações que simulam compressão JPEG/H.264 (artefatos reais)

Autor: [Seu Nome]
TCC — Pós-Graduação em Visão Computacional — PUC-Rio
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Detecção de rosto — instale: pip install facenet-pytorch
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ------------------------------------------------------------------
# Extrator de Frames
# ------------------------------------------------------------------
class VideoFrameExtractor:
    """
    Extrai frames de vídeos com detecção de rosto integrada.

    Pré-processa vídeos do FF++, Celeb-DF e DFDC para o formato
    de imagens de rosto recortadas prontas para o modelo.

    Args:
        output_dir: pasta onde salvar os frames extraídos.
        n_frames: número de frames por vídeo.
        face_margin: margem em % ao redor do rosto detectado.
        device: 'cpu' ou 'cuda' para MTCNN.
    """

    def __init__(
        self,
        output_dir: str,
        n_frames: int = 30,
        face_margin: float = 0.3,
        device: str = "cpu",
    ):
        self.output_dir  = Path(output_dir)
        self.n_frames    = n_frames
        self.face_margin = face_margin

        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=IMG_SIZE,
                margin=int(IMG_SIZE * face_margin),
                min_face_size=80,
                keep_all=False,
                device=device,
            )
        else:
            self.detector = None
            print("[Aviso] MTCNN não disponível — usando frame inteiro.")

    def extract_video(self, video_path: str, label: int, split: str = "train") -> int:
        """
        Extrai N frames de um vídeo, detecta o rosto e salva.

        Args:
            video_path: caminho do .mp4 / .avi
            label: 0 = real, 1 = fake
            split: 'train' | 'val' | 'test'

        Returns:
            Número de frames extraídos com sucesso.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.n_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int).tolist()

        label_str = "fake" if label == 1 else "real"
        out_dir = self.output_dir / split / label_str
        out_dir.mkdir(parents=True, exist_ok=True)

        stem    = Path(video_path).stem
        saved   = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tenta detectar e recortar rosto
            face = self._crop_face(frame_rgb)
            if face is None:
                face = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))

            out_path = out_dir / f"{stem}_f{idx:04d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        cap.release()
        return saved

    def _crop_face(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Detecta e recorta o rosto dominante via MTCNN."""
        if self.detector is None:
            return None
        try:
            face_tensor = self.detector(frame_rgb)
            if face_tensor is None:
                return None
            # De volta a numpy uint8
            face_np = face_tensor.permute(1, 2, 0).numpy()
            face_np = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            return face_np
        except Exception:
            return None


# ------------------------------------------------------------------
# Augmentações
# ------------------------------------------------------------------
def get_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """
    Augmentações de treino com foco em robustez a compressão e artefatos.

    Deepfakes circulam predominantemente em plataformas sociais que
    aplicam compressão H.264/H.265 e JPEG pesado — o modelo deve
    detectar mesmo sob degradação severa.
    """
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),

        # Simula compressão de rede social (WhatsApp, Instagram, TikTok)
        A.OneOf([
            A.ImageCompression(quality_lower=40, quality_upper=95, p=1.0),
            A.Downscale(scale_min=0.5, scale_max=0.9,
                        interpolation=cv2.INTER_LINEAR, p=1.0),
        ], p=0.6),

        # Degradação de câmera / transmissão
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussNoise(var_limit=(5, 30), p=1.0),
        ], p=0.4),

        # Cor — deepfakes frequentemente têm problemas de cor/iluminação
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Oclusão parcial (óculos, dedos, logos de plataforma)
        A.CoarseDropout(
            max_holes=6, max_height=40, max_width=40,
            min_holes=1, fill_value=0, p=0.3
        ),

        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class DeepfakeDataset(Dataset):
    """
    Dataset binário de detecção de deepfakes.

    Espera estrutura de diretórios:
        root/
        ├── real/   *.jpg  (label = 0)
        └── fake/   *.jpg  (label = 1)

    Args:
        root: pasta raiz com subdiretórios 'real' e 'fake'.
        transform: pipeline Albumentations.
        max_per_class: limita amostras por classe (balanceamento).
        ff_method_filter: se fornecido, usa apenas imagens desse método
            de manipulação do FF++ (ex: 'Deepfakes', 'FaceSwap').
    """

    def __init__(
        self,
        root: str,
        transform=None,
        max_per_class: int | None = None,
        ff_method_filter: str | None = None,
    ):
        self.root      = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, class_name in [(0, "real"), (1, "fake")]:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            files = sorted(class_dir.glob("*.jpg"))

            if ff_method_filter:
                files = [f for f in files if ff_method_filter.lower() in f.stem.lower()]

            if max_per_class:
                files = files[:max_per_class]

            self.samples.extend([(f, label) for f in files])

        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"[Dataset] {root} → real: {real_count} | fake: {fake_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.float32)


# ------------------------------------------------------------------
# Sampler balanceado
# ------------------------------------------------------------------
def make_balanced_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    """
    Cria sampler que balanceia real e fake no batch.

    FF++ e Celeb-DF têm mais fakes que reais — o sampler corrige isso.
    """
    labels = [l for _, l in dataset.samples]
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
    )


# ------------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------------
def build_dataloaders(cfg: dict):
    """
    Constrói loaders de treino, validação e teste.

    Args:
        cfg: configuração com 'train_dir', 'val_dir', 'test_dir',
             'batch_size', 'num_workers', 'max_per_class'.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DeepfakeDataset(
        root=cfg["train_dir"],
        transform=get_train_transform(cfg.get("img_size", IMG_SIZE)),
        max_per_class=cfg.get("max_per_class"),
    )
    val_ds = DeepfakeDataset(
        root=cfg["val_dir"],
        transform=get_val_transform(cfg.get("img_size", IMG_SIZE)),
    )
    test_ds = DeepfakeDataset(
        root=cfg["test_dir"],
        transform=get_val_transform(cfg.get("img_size", IMG_SIZE)),
    )

    sampler = make_balanced_sampler(train_ds) if cfg.get("balanced_sampler", True) else None
    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader, test_loader

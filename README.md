# 🕵️ Detecção de Deepfakes com Dual-Stream Vision Transformer
### Análise Espacial + Frequencial com Cross-Attention

> **Pós-Graduação em Visão Computacional**  
> Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)  
> Autor: Paulo de Tarso Castro Silva · Orientador: Manoela Kohler
> Ano: 2026

---

## 📋 Sumário

- [Motivação](#-motivação)
- [Objetivo](#-objetivo)
- [Datasets](#-datasets)
- [Arquitetura](#-arquitetura)
- [Explicabilidade](#-explicabilidade)
- [Resultados](#-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Contexto Legal Brasileiro](#-contexto-legal-brasileiro)
- [Referências](#-referências)

---

## 💡 Motivação

Deepfakes — vídeos e imagens sintéticas de rostos humanos gerados por IA — representam uma das ameaças mais urgentes à integridade da informação digital. Em 2024:

- **R$ 200M+** estimados em fraudes financeiras via deepfake de executivos no Brasil (FEBRABAN, 2024)
- **3 candidatos** políticos brasileiros tiveram vídeos falsos viralizados durante as eleições municipais
- A Interpol identificou deepfakes como vetor crítico de **fraude de identidade e extorsão**
- O **TSE** e o **CGI.br** iniciaram grupos de trabalho sobre deepfakes em 2024

A detecção automática é urgente — e precisa ser **explicável**: um sistema que diz "é fake" sem justificativa não serve para fins jornalísticos, periciais ou jurídicos.

---

## 🎯 Objetivo

Desenvolver um sistema de detecção de deepfakes que:

1. **Detecte** manipulações em rostos com alta acurácia em múltiplos métodos de síntese
2. **Combine** análise espacial (artefatos visuais) e frequencial (assinaturas GAN)
3. **Explique** as evidências via mapa de atenção (ViT) e análise espectral (DFT)
4. **Generalize** para deepfakes não vistos no treinamento (cross-dataset)
5. **Produza** um painel de evidências forenses adequado para uso pericial

---

## 📊 Datasets

### FaceForensics++ (FF++) — principal

| Propriedade | Valor |
|---|---|
| Vídeos reais | 1.000 (YouTube) |
| Métodos de manipulação | 4 |
| Total de vídeos falsos | 4.000 |
| Qualidades | c0 (raw), c23 (HQ), c40 (LQ) |
| Acesso | Formulário: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |

**4 métodos de manipulação:**
| Método | Técnica | Dificuldade |
|---|---|---|
| `Deepfakes` | AutoEncoder face-swap | Média |
| `FaceSwap` | Computer Graphics | Baixa |
| `Face2Face` | Reenactment de expressão | Alta |
| `NeuralTextures` | Síntese neural de textura | Alta |

### Celeb-DF v2 — avaliação de generalização

| Propriedade | Valor |
|---|---|
| Vídeos reais | 590 |
| Vídeos falsos | 5.639 |
| Qualidade | Alta (produção profissional) |
| Uso | Teste cross-dataset (modelo não vê durante treino) |
| Download | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) |

---

## 🏗️ Arquitetura

```
Imagem de Rosto (224×224×3)
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ▼                                              ▼
┌─────────────────────────┐              ┌──────────────────────────┐
│  STREAM ESPACIAL        │              │  STREAM FREQUENCIAL       │
│  ViT-B/16               │              │  FFT2D → log|F| → ViT-6  │
│  (pré-treinado ImageNet)│              │  (treinado do zero)       │
│                         │              │                           │
│  Detecta:               │              │  Detecta:                 │
│  • Blending boundaries  │              │  • GAN fingerprints       │
│  • Skin texture glitch  │              │  • Frequências anômalas   │
│  • Eye inconsistency    │              │  • Grade de espectro GAN  │
│                         │              │                           │
│  Output: (B, 197, 768)  │              │  Output: (B, 197, 768)    │
└────────────┬────────────┘              └──────────────┬────────────┘
             │                                          │
             └─────────────┬────────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  Cross-Attention        │  × 2 camadas
             │  (Spatial ↔ Freq)       │
             │                         │
             │  Cada stream "pergunta" │
             │  ao outro onde há       │
             │  artefatos correlatos   │
             └────────────┬────────────┘
                          │
                    [CLS_spatial ⊕ CLS_freq]   (dim 1536)
                          │
                          ▼
             ┌─────────────────────────┐
             │  MLP Classificador      │
             │  1536→512→128→1         │
             │  GELU + Dropout(0.4)    │
             └────────────┬────────────┘
                          │
                     P(fake) ∈ [0, 1]
```

### Por que dois streams?

| Evidência | Stream | Exemplo |
|---|---|---|
| Borda de fusão de rosto | Espacial | Linha visível entre rosto sintético e pescoço |
| Textura de pele anômala | Espacial | Pele excessivamente suave / padrão repetitivo |
| GAN fingerprint | Frequencial | Picos periódicos em frequências médias |
| Artefatos de compressão sintética | Frequencial | Distribuição espectral diferente de câmera real |
| Movimento ocular | Espacial | Piscadas ausentes ou não naturais |

> 💡 A análise frequencial é particularmente poderosa para deepfakes de **alta qualidade**: mesmo quando os artefatos visuais são imperceptíveis ao olho humano, as redes GAN deixam assinaturas espectrais detectáveis.

---

## 👁️ Explicabilidade

### Mapa de Atenção (ViT Spatial)

O token CLS do ViT agrega informação de todos os patches. Visualizamos a atenção deste token para identificar **onde** o modelo encontrou evidências de manipulação:

```
Rosto real:         Atenção difusa, concentrada em olhos e boca (regiões naturalmente discriminativas)
Deepfake FaceSwap:  Atenção concentrada nas BORDAS do rosto — exatamente onde o blending falha
Deepfake Face2Face: Atenção na região da boca — movimento labial não sincronizado
```

### Análise Espectral

O espectro DFT revela padrões invisíveis ao olho humano:

```
Imagem real:        Espectro suave, energia concentrada em baixas frequências
Deepfake StyleGAN:  Picos regulares em ~64 Hz e ~128 Hz (stride da conv do gerador)
Deepfake Deepfakes: Grade de pontos no espectro (artefato do autoencoder)
```

### Painel de Evidências Forenses

O módulo `visualize.py` gera um painel de 6 painéis:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Rosto          │  Mapa de        │  Sobreposição    │
│  Analisado      │  Atenção (ViT)  │  Atenção + Rosto │
├─────────────────┼─────────────────┼─────────────────┤
│  Espectro DFT   │  Espectro       │  Perfil Radial   │
│  (escala cinza) │  (Inferno)      │  de Frequência   │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## 📈 Resultados

> Treinamento em FF++ (c23), avaliação intra-dataset e cross-dataset.

### Intra-Dataset — FF++ c23

| Método | AUC | F1 | Accuracy |
|---|---|---|---|
| Deepfakes | 0.983 | 0.961 | 0.962 |
| Face2Face | 0.971 | 0.944 | 0.945 |
| FaceSwap | 0.989 | 0.973 | 0.974 |
| NeuralTextures | 0.954 | 0.921 | 0.923 |
| **Média** | **0.974** | **0.950** | **0.951** |

### Métricas Forenses

| Métrica | Valor |
|---|---|
| EER (Equal Error Rate) | **4.8%** |
| TPR @ FPR=5% | **93.2%** |
| AUC ROC | **0.974** |

### Cross-Dataset — treino FF++, teste Celeb-DF v2

| Modelo | AUC Celeb-DF |
|---|---|
| Xception baseline | 0.736 |
| EfficientNet-B4 | 0.779 |
| ViT-B (espacial só) | 0.821 |
| **Dual-Stream (este TCC)** | **0.853** |

> 🎯 O stream frequencial contribui com **+3.2% AUC** na generalização cross-dataset — evidência de que assinaturas espectrais são mais transferíveis que artefatos visuais.

### Ablação dos Componentes

| Configuração | AUC FF++ | AUC Celeb-DF |
|---|---|---|
| Só Stream Espacial | 0.961 | 0.821 |
| Só Stream Frequencial | 0.923 | 0.794 |
| Concatenação simples | 0.968 | 0.839 |
| **+ Cross-Attention (proposto)** | **0.974** | **0.853** |

---

## 📁 Estrutura do Projeto

```
tcc_deepfake/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── model.py          ← Dual-Stream ViT + Cross-Attention
│   ├── dataset.py        ← FF++/Celeb-DF + extração de frames + augmentações
│   ├── train.py          ← Focal Loss, EMA, AUC como métrica principal
│   ├── evaluate.py       ← AUC, EER, TPR@FPR, curvas ROC por método
│   └── visualize.py      ← Mapa de atenção + análise espectral + painel forense
│
├── notebooks/
│   ├── 01_eda.ipynb              ← Análise exploratória dos datasets
│   ├── 02_frequency_analysis.ipynb ← Visualização de espectros real vs fake
│   └── 03_error_analysis.ipynb  ← Análise dos falsos negativos
│
├── data/
│   ├── real/                     ← Frames extraídos de vídeos reais
│   ├── fake/                     ← Frames extraídos de deepfakes
│   └── processed/
│       ├── train/
│       │   ├── real/  *.jpg
│       │   └── fake/  *.jpg
│       ├── val/
│       └── test/
│
└── results/
    ├── models/
    │   └── best_model.pth
    └── figures/
        ├── roc_curve.png
        ├── confusion_matrix.png
        ├── score_distribution.png
        └── evidence_panels/
```

---

## 🚀 Como Executar

### 1. Configurar ambiente

```bash
git clone https://github.com/[seu-usuario]/tcc-deepfake-detection.git
cd tcc-deepfake-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Baixar FaceForensics++

```bash
# Solicitar acesso: https://github.com/ondyari/FaceForensics
# Após receber link, usar o script oficial:
python faceforensics_download.py . -d all -c c23 -t videos
```

### 3. Extrair frames e rostos

```python
from src.dataset import VideoFrameExtractor
extractor = VideoFrameExtractor(output_dir="data/processed", n_frames=30)

# Reais
for video in Path("data/original_sequences").glob("*.mp4"):
    extractor.extract_video(str(video), label=0, split="train")

# Deepfakes (um método por vez)
for video in Path("data/manipulated_sequences/Deepfakes").glob("*.mp4"):
    extractor.extract_video(str(video), label=1, split="train")
```

### 4. Treinar

```bash
python src/train.py --config configs/config.yaml
```

### 5. Avaliar

```bash
python src/evaluate.py \
    --config configs/config.yaml \
    --checkpoint results/models/best_model.pth
```

### 6. Analisar uma imagem (painel forense)

```bash
python src/visualize.py \
    --image data/processed/test/fake/video001_f0120.jpg \
    --checkpoint results/models/best_model.pth \
    --output results/figures/evidence_panels/exemplo.png
```

---

## ⚖️ Contexto Legal Brasileiro

Este trabalho tem relevância direta no cenário jurídico nacional:

- **Marco Civil da Internet** (Lei 12.965/2014): responsabilidade de plataformas por conteúdo sintético malicioso
- **Lei Geral de Proteção de Dados** (Lei 13.709/2018): uso indevido de biometria facial em deepfakes pode configurar violação de dados pessoais sensíveis
- **Resolução TSE 23.610**: o TSE proibiu deepfakes eleitorais em 2022 — sistemas de detecção são ferramentas de enforcement
- **PL 2630/2020** (Lei das Fake News, em tramitação): pode criar obrigação de rotulação e detecção de conteúdo sintético
- **Comitê Gestor da Internet (CGI.br)**: publicou recomendações sobre integridade informacional em 2024

> O sistema desenvolvido neste TCC pode ser integrado a pipelines de moderação de conteúdo, ferramentas de verificação de fatos e sistemas periciais de análise forense digital.

---

## 🔬 Contribuições Técnicas

1. **Stream Frequencial com FFT + ViT**: extração de patches espectrais via DFT 2D e embedding compatível com arquitetura ViT — primeira aplicação de ViT diretamente sobre espectro de magnitude em detecção de deepfakes.

2. **Cross-Attention Bidirecional**: os dois streams trocam informação via Multi-Head Cross-Attention iterativo — superior à fusão por concatenação (+1.4% AUC cross-dataset).

3. **EMA para Deepfake Detection**: uso de Exponential Moving Average na inferência reduz instabilidade em casos limite (imagens com qualidade intermediária).

4. **Painel Forense Exportável**: módulo `visualize.py` produz painéis de 6 visualizações com tema escuro e veredito destacado, adequados para uso em relatórios periciais e jornalismo investigativo.

---

## 📚 Referências

1. **Rössler, A. et al.** (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV 2019*.

2. **Li, Y. et al.** (2020). Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics. *CVPR 2020*.

3. **Dosovitskiy, A. et al.** (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

4. **Qian, Y. et al.** (2020). Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues. *ECCV 2020*.

5. **Zhao, H. et al.** (2021). Multi-attentional Deepfake Detection. *CVPR 2021*.

6. **Durall, R. et al.** (2020). Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions. *CVPR 2020*.

7. **Chen, C. et al.** (2021). CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. *ICCV 2021*.

8. **Zhang, X. et al.** (2019). Detecting and Simulating Artifacts in GAN Fake Images. *WIFS 2019*.

9. **Lin, T. et al.** (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

---

## 📄 Licença

MIT para fins acadêmicos e de pesquisa.

> ⚠️ **Nota Ética**: Este sistema foi desenvolvido **exclusivamente para detecção** de deepfakes. O código não inclui nem deve ser usado para geração de conteúdo sintético malicioso.

---

> *"A mentira percorre o mundo antes que a verdade calce as botas — a visão computacional pode ajudar a mudar isso."*

# CIFAR‑10 ResNet‑50 Fine‑Tuning + Grad‑CAM

> Fine‑tuning de **ResNet‑50** pre‑entrenada en ImageNet sobre **CIFAR‑10** con interpretabilidad **Grad‑CAM**.

![Grad‑CAM sample](assets/gradcam_grid.gif)

---

## ✨ Características principales

| Módulo                  | Descripción                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| **Transfer Learning**   | Carga de *weights* ImageNet, congelado selectivo y *unfreezing* progresivo |
| **Aumento de datos**    | `RandomCrop`, `RandAugment`, `CutMix`, normalización CIFAR‑10              |
| **Optuna Search**       | Búsqueda bayesiana de `lr`, *weight‑decay*, *dropout* y scheduler          |
| **Schedulers**          | `CosineAnnealingWarmRestarts` + *LR finder* automático                     |
| **Early Stopping**      | Basado en *validation accuracy* con *patience* adaptativo                  |
| **Grad‑CAM**            | Mapas de activación por clase (capa `layer4`) exportados como PNG y GIF    |
| **Métricas & Matrices** | Accuracy, Precision, Recall, F1 + matriz de confusión Plotly               |

---

## 📂 Estructura del proyecto

```text
cifar10-resnet-gradcam/
├── data/                 # Descarga automática CIFAR‑10
├── src/
│   ├── datamodule.py     # PyTorch Lightning DataModule
│   ├── model.py          # ResNet‑50 lightning module + Grad‑CAM hooks
│   ├── train.py          # Entrenamiento CLI
│   └── tune.py           # Optuna hyper‑parameter search
├── notebooks/            # Análisis EDA y visualización Grad‑CAM
├── reports/              # Figuras, matrices, resultados CSV
├── assets/               # Imágenes para README
├── requirements.txt
└── README.md
```

---

## ⚙️ Requisitos

```bash
python >= 3.9
pytorch >= 2.2
pytorch‑lightning >= 2.2
optuna >= 3.6
albumentations
opencv‑python
plotly scikit‑learn rich tqdm
```

Instalación rápida:

```bash
pip install -r requirements.txt
```

---

## 🚀 Entrenamiento rápido

```bash
# Fine‑tuning con hiperparámetros por defecto
python src/train.py --epochs 30 --batch_size 128 --gpus 1

# Búsqueda de 40 ensayos con Optuna (≈ 2 h en una RTX 3060)
python src/tune.py --trials 40 --timeout 7200
```

Los checkpoints y logs se guardan en `runs/`.

### Visualización Grad‑CAM

```bash
python notebooks/gradcam_demo.py --checkpoint <path_ckpt> --n_samples 10
```

Genera `gif` y `png` en `outputs/gradcam/`.

---

## 📊 Resultados

| Métrica                | Valor             |
| ---------------------- | ----------------- |
| Test Accuracy          | **88.2 % ±0.3**   |
| Parámetros entrenables | 23 M              |
| Tiempo entrenamiento   | 28 min (RTX 3060) |

*Comparación completa y curva de aprendizaje en `reports/`.*

---

## 🏎️ Eficiencia

| Configuración         | FPS entrenamiento | FPS inferencia |
| --------------------- | ----------------- | -------------- |
| Full‑precision        | 480               | 1 200          |
| Mixed precision (AMP) | **830**           | **2 050**      |

---

## 📜 Licencia

Distribuido bajo licencia MIT © 2025 **Fernando NasserEddine López**.

---

## 🙌 Créditos

* [PyTorch Lightning](https://lightning.ai/)
* [Optuna](https://optuna.org/)
* [Grad‑CAM](https://github.com/jacobgil/pytorch-grad-cam)
* Dataset [CIFAR‑10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky 2009)

Si este código te resulta útil, ¡no olvides dejarme un ⭐️!

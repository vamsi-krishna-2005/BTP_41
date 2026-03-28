# BTP Review Prep — UrbanLens Land-Cover Segmentation (U-Net / Swin-UNet)

This document explains **what each file does**, **how training works**, the **key hyperparameters/parameters**, and the **math** behind loss + metrics. It is based on your code in:

- `train.py`
- `dataset.py`
- `models.py`
- `utils.py`
- `split_data.py`
- `metrics_visual.py`
- `visual_compare.py`
- `btp_preprocessing.py` (Colab-exported preprocessing / sanity checks)

---

## 1) One-line project description (say this first in review)

We solve **semantic segmentation** on satellite imagery: for every pixel, predict one of **7 land-cover classes** (Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown) using either a small **U-Net** baseline or a **Swin Transformer encoder + CNN decoder** variant.

---

## 2) End-to-end workflow (flow of work)

```mermaid
flowchart TD
  A[Raw dataset folder\n*_sat.jpg + *_mask.png] --> B[split_data.py\n(train/valid/test split)]
  B --> C[dataset.py UrbanLensDataset\nRGB mask -> class-id mask]
  C --> D[Albumentations\nNormalize + optional flip]
  D --> E[DataLoader\nmini-batches]
  E --> F[models.py\nUNet or SwinUNet]
  F --> G[Logits tensor\nB x 7 x H x W]
  G --> H[CrossEntropyLoss\npixel-wise]
  H --> I[Mixed precision autocast + GradScaler]
  I --> J[AdamW optimizer\nupdate weights]
  J --> K[utils.py metrics\nmIoU + GAF]
  K --> L[Checkpoint + CSV\n(checkpoints/, results/)]
  L --> M[metrics_visual.py\nplot curves]
  L --> N[visual_compare.py\nqualitative comparisons]
```

---

## 3) Data + labels

### 3.1 Input/Output format

- **Input image**: RGB satellite image, resized to \(224 \times 224\)
  - Tensor shape: \([3, 224, 224]\)
- **Target mask**: integer class-id per pixel
  - Tensor shape: \([224, 224]\), dtype `long` (int64)
- **Number of classes**: \(C = 7\)

### 3.2 Mask encoding (RGB → class id)
In `dataset.py`, each pixel in the RGB mask is mapped to a class id:

- (0, 255, 255) → 0 Urban
- (255, 255, 0) → 1 Agriculture
- (255, 0, 255) → 2 Rangeland
- (0, 255, 0) → 3 Forest
- (0, 0, 255) → 4 Water
- (255, 255, 255) → 5 Barren
- (0, 0, 0) → 6 Unknown

Mathematically, for a pixel at location \((i,j)\) with RGB value \(r_{ij} \in \mathbb{R}^3\):
\[
y_{ij} =
\begin{cases}
0 & \text{if } r_{ij}=(0,255,255)\\
1 & \text{if } r_{ij}=(255,255,0)\\
\dots\\
6 & \text{if } r_{ij}=(0,0,0)
\end{cases}
\]

---

## 4) Augmentations / preprocessing (what happens to images)

In `train.py`:

- **Training**:
  - Random horizontal flip with probability 0.5
  - Normalize with ImageNet stats:
    - mean = (0.485, 0.456, 0.406)
    - std = (0.229, 0.224, 0.225)
  - Convert to tensor (`ToTensorV2`)

- **Validation**:
  - Normalize (same)
  - Convert to tensor

Important review note: **Normalize expects image in 0–255 or 0–1 depending on library config**; Albumentations typically handles it consistently, but if asked, you can say “we normalize inputs to stabilize optimization and match common pretrained conventions (even though `pretrained=False` here).”

---

## 5) Models (what architectures you have)

### 5.1 Baseline `UNet` (very small U-Net)
File: `models.py`

This is a compact encoder-decoder with one skip connection:

- Encoder:
  - `enc1`: DoubleConv(3 → 64)
  - MaxPool
  - `enc2`: DoubleConv(64 → 128)
- Decoder:
  - Transposed conv upsample: (128 → 64)
  - Concatenate with early features (skip connection)
  - Final 1×1 conv to `n_classes=7`

**Output**: logits \([B, 7, 224, 224]\)

### 5.2 `SwinUNet` (Swin Transformer encoder + decoder)
File: `models.py`

- Encoder backbone:
  - `timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, features_only=True)`
  - Produces multi-scale feature maps (from code comment):
    - \(f_0: [B, 56, 56, 96]\)
    - \(f_1: [B, 28, 28, 192]\)
    - \(f_2: [B, 14, 14, 384]\)
    - \(f_3: [B, 7, 7, 768]\)
  - They are permuted from **channels-last** to **channels-first**: \([B,H,W,C] \to [B,C,H,W]\)

- Decoder:
  - `up1`: 768→384, 7→14
  - `up2`: 384→192, 14→28 (uses additive skip: `d1 + f[2]`)
  - `up3`: 192→96, 28→56 (uses additive skip: `d2 + f[1]`)
  - `final`: 1×1 conv (96→7), then upsample ×4 (56→224)

**Output**: logits \([B, 7, 224, 224]\)

Review note: compared to U-Net, this uses a stronger **global-context encoder** (transformer) which is often beneficial for land-cover segmentation where large-scale context matters.

---

## 6) Training loop (how it is trained)

File: `train.py`

### 6.1 Key hyperparameters (mention these in review)

- **Epochs**: 30
- **Batch size**: 8
- **Optimizer**: AdamW
  - weight_decay = \(1\times 10^{-4}\)
  - learning rate:
    - Swin: \(3\times 10^{-5}\)
    - U-Net: \(1\times 10^{-4}\)
- **Loss**: `CrossEntropyLoss` (multi-class per pixel)
- **Mixed precision**:
  - `autocast("cuda")` + `GradScaler` if GPU available
- **Checkpoints**: saved each epoch to `checkpoints/Albumented_{RUN_TAG}_latest.pth`
- **Metrics logged to CSV**: `results/Albumented_{RUN_TAG}_pretrained=false_metrics.csv`

### 6.2 Shapes through the loop (common review question)

Let:
- batch size \(B\)
- classes \(C=7\)
- height/width \(H=W=224\)

Then:
- images \(X \in \mathbb{R}^{B \times 3 \times H \times W}\)
- model output logits \(Z = f_\theta(X) \in \mathbb{R}^{B \times C \times H \times W}\)
- ground truth \(Y \in \{0,\dots,6\}^{B \times H \times W}\)

### 6.3 Loss math (Cross Entropy for segmentation)

For each pixel \(p=(i,j)\), the model outputs logits \(z_{b,c,p}\).
Softmax probability:
\[
P_{b,c,p} = \frac{\exp(z_{b,c,p})}{\sum_{k=0}^{C-1}\exp(z_{b,k,p})}
\]
Pixel-wise cross-entropy:
\[
\ell_{b,p} = -\log\left(P_{b,\,y_{b,p},\,p}\right)
\]
Overall loss (typical implementation): mean across pixels and batch:
\[
\mathcal{L}(\theta) = \frac{1}{BHW}\sum_{b=1}^{B}\sum_{p}\ell_{b,p}
\]

### 6.4 Optimization math (AdamW in words)

AdamW is Adam + decoupled weight decay. High-level:

- compute gradient \(g_t = \nabla_\theta \mathcal{L}(\theta_t)\)
- update moving averages \(m_t, v_t\)
- apply parameter update with learning rate \(\alpha\)
- apply weight decay separately: \(\theta \leftarrow \theta - \alpha \lambda \theta\)

What to say: “We use AdamW because it’s stable for transformers and segmentation; weight decay improves generalization.”

### 6.5 Mixed precision (why it’s there)

On CUDA:
- `autocast` runs many ops in FP16/TF32 where safe → faster and less memory
- `GradScaler` scales the loss to avoid FP16 underflow during backprop

---

## 7) Metrics (how you evaluate)

File: `utils.py`

### 7.1 mIoU (mean Intersection over Union)

For a given class \(c\):

- intersection: pixels predicted as \(c\) AND true label is \(c\)
- union: pixels predicted as \(c\) OR true label is \(c\)

\[
\text{IoU}_c = \frac{| \{p: \hat{y}_p=c \wedge y_p=c\} |}{| \{p: \hat{y}_p=c \vee y_p=c\} |}
\]

Then mean over classes that actually appear (union > 0 in code):
\[
\text{mIoU} = \frac{1}{|\mathcal{C}'|}\sum_{c \in \mathcal{C}'} \text{IoU}_c
\]
where \(\mathcal{C}' = \{c : \text{union}_c > 0\}\).

Implementation detail: your `get_miou` does `argmax` over channel dimension to get predicted class per pixel.

### 7.2 GAF (custom weighted area score)

File: `utils.py`

You defined weights:
```text
0 Urban: 0.0
1 Agriculture: 0.5
2 Rangeland: 0.7
3 Forest: 1.0
4 Water: 1.0
5 Barren: 0.0
6 Unknown: 0.0
```

Given a predicted mask \(M\) with pixel labels, let \(N\) be total pixels and \(n_c\) be number of pixels predicted as class \(c\).

\[
\text{GAF}(M) = \frac{1}{N}\sum_{c=0}^{6} w_c \, n_c
\]

Interpretation: “GAF is a **weighted fraction of area**, rewarding classes (Forest, Water) more than others.”

Important: in validation, you compute `calculate_gaf(out)` on model output. The code expects either:
- a class-id mask, or
- a `(C,H,W)` score map, in which case it `argmax`es to class-id.

---

## 8) Checkpointing + logging (what gets saved)

### 8.1 Checkpoints
File: `utils.py`

- Save: `checkpoints/Albumented_{model_name}_latest.pth`
- Contains:
  - `epoch`
  - `state_dict` (model weights)
  - `optimizer` state

Resume logic:
- if checkpoint exists AND `--resume` is passed, training starts from saved `epoch`
- otherwise starts from 0

### 8.2 CSV metrics
File: `train.py`

Each epoch appends:
- `train_loss`, `val_loss`
- `train_mIoU`, `val_mIoU`
- `avg_gaf`

Saved to:
- `results/Albumented_{RUN_TAG}_pretrained=false_metrics.csv`

---

## 9) Visualization / reporting utilities

### 9.1 Plot curves
File: `metrics_visual.py`

Reads the results CSV(s) and plots:
- loss vs epoch
- mIoU vs epoch
- GAF vs epoch

Saves png to `results/Metrics_visual_after_albumentation_{model_name}_pretrained=false.png`

### 9.2 Qualitative comparison (U-Net vs Swin-UNet)
File: `visual_compare.py`

- Loads a few samples from `TEST_DIR`
- Loads checkpoints:
  - `checkpoints/Albumented_unet_gpu_latest.pth`
  - `checkpoints/Albumented_swin_gpu_latest.pth`
- Runs inference and saves a 4-panel figure:
  1) input image
  2) ground truth mask (+ GAF)
  3) U-Net prediction (+ mIoU, GAF)
  4) Swin-UNet prediction (+ mIoU, GAF)

Saves to:
- `results/final_comparison_Albumentation_Pretrained=false_{idx}.png`

---

## 10) Data splitting helper

File: `split_data.py`

Moves a subset of labeled images (having masks) from train into:
- valid (10%)
- test (10%)
Remaining stays train (80%).

Review note: this script **moves files**, not copies—so it permanently rearranges the dataset folder structure.

---

## 11) Preprocessing notebook export (optional / historical)

File: `btp_preprocessing.py`

This is a Colab-generated script that:
- unzips an archive
- defines a dataset class similar to yours
- does sanity checks (unique mask values, visualization)

It’s useful as “evidence we verified masks are correctly converted”.

---

## 12) “How to run” (commands)

From the project folder:

- Train Swin-UNet:
  - `python train.py --model swin`
- Train U-Net:
  - `python train.py --model unet`
- Resume from latest checkpoint:
  - `python train.py --model swin --resume`

Note: `TRAIN_DIR` / `VAL_DIR` / `TEST_DIR` paths in code are currently **hard-coded Linux paths** (e.g., `/home/...`). For Windows, update them to your local dataset location (or convert them to CLI arguments for a cleaner review story).

---

## 13) Common review questions (and crisp answers)

- **Why CrossEntropyLoss?**
  - “Because segmentation is multi-class per pixel; CrossEntropyLoss directly optimizes the negative log-likelihood of the correct class for each pixel.”

- **Why Swin?**
  - “Transformers capture larger context; satellite scenes often need global context (e.g., large water bodies, urban sprawl).”

- **What is the model output?**
  - “Logits per pixel per class: \([B,7,H,W]\). We apply `argmax` to get the final class map.”

- **What does mIoU measure?**
  - “Overlap quality between predicted vs ground-truth regions, averaged across present classes.”

- **What is GAF?**
  - “A custom weighted area score: fraction of pixels in each class multiplied by that class weight.”

---

## 14) 5–7 minute review talk track (ready-to-speak)

Use this as your verbal script:

1) **Problem & output**
   - “We do land-cover semantic segmentation on satellite images. For every pixel we predict one of 7 classes.”

2) **Dataset & labels**
   - “Masks are provided as RGB colors; we convert them into integer class IDs using a fixed color→class mapping.”
   - “Images and masks are resized to 224×224 so training is consistent and matches Swin’s default input size.”

3) **Pipeline**
   - “DataLoader gives mini-batches. We apply Albumentations: horizontal flip for augmentation and ImageNet-style normalization.”

4) **Models**
   - “Baseline is a small U-Net with skip connection.”
   - “Main model is SwinUNet: Swin Transformer encoder from timm (`swin_tiny`) and a light decoder with transposed convolutions and skip fusion.”

5) **Training**
   - “We use pixel-wise CrossEntropyLoss, AdamW optimizer, and mixed precision on GPU for speed and memory.”
   - “We checkpoint each epoch and log metrics to CSV for plotting.”

6) **Metrics**
   - “Primary metric is mIoU, standard for segmentation.”
   - “We also track a custom GAF score: a weighted area fraction that emphasizes classes like Forest and Water.”

7) **Results & evidence**
   - “We plot curves from CSV and generate qualitative comparisons showing input vs ground truth vs predictions for both models.”

8) **Closing**
   - “SwinUNet is expected to improve due to better global context. Next steps are stronger augmentations, class imbalance handling, and removing hard-coded paths for portability.”

---

## 15) Limitations / risks (mention if asked)

- **Hard-coded dataset paths**: training/val/test dirs currently use Linux absolute paths in `train.py`, `split_data.py`, and `visual_compare.py`.
- **Very small U-Net baseline**: your `UNet` is intentionally minimal; it’s a baseline, not a full-depth U-Net.
- **Class imbalance**: CrossEntropyLoss is unweighted; if some classes are rare, mIoU can suffer (common in remote sensing).
- **GAF is custom**: it is not a standard metric; explain it as a domain-driven weighted area proxy (and keep mIoU as the primary scientific metric).
- **`visual_compare.py` device note**: it sets `device = cpu` (so it always runs on CPU) even though there’s a comment about moving to GPU after loading.



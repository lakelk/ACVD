# Anatomy-Constrained Voxel Diffusion (ACVD) for Controllable Synthesis of Complex Lung Nodules

##  Overview

Accurate synthesis of pulmonary nodules with complex topological interactions (e.g., pleural tagging or vascular attachment) is essential for advancing Data-centric AI in medical imaging. Existing generative models face two primary bottlenecks when generating micro-scale localized lesions:
1. **Spatial Misalignment**: Prevalent Latent Diffusion Models (LDMs) compress 3D volumes via Variational Autoencoders (VAEs or VQ-VAEs). This dimensional reduction destroys sub-millimeter high-frequency spatial alignment, causing severe boundary blurring and texture smoothing.
2. **Contextual Ambiguity**: Standard texture-guided methods (e.g., global histogram injection) lack deterministic spatial boundaries, failing to provide the physical constraints required to render biologically coherent tissue interfaces.

**Anatomy-Constrained Voxel Diffusion (ACVD)** establishes a high-fidelity computational simulation engine operating directly in the uncompressed $48^3$ voxel space. By integrating dense multi-channel anatomical priors via a parallel 3D ControlNet branch and employing a deterministic Masked Blended Diffusion inference strategy, ACVD achieves state-of-the-art micro-texture realism and strict topological structural adherence.

---

##  Repository Structure

The codebase is organized cleanly into standardized training scripts, evaluation modules, and architectural definitions:

```text
├── config.py                 # Global configuration, directory paths, and model checkpoint locations
├── models/
│   └── resnet.py             # 3D ResNet-50 feature backbone pre-trained on Med3D for volumetric FID evaluation
├── weights/
│   └── resnet_50_23dataset.pth # Pre-trained Med3D ResNet-50 weights downloaded from 3DMedicalNet for FID evaluation
├── train_acvd.py             # [Ours] Train ACVD (Voxel space $48^3$ + Full 7-Channel Anatomical Conditioning)
├── train_voxel.py            # [Ablation] Train ACVD w.o. Anatomical Prior (Voxel space + Basic 3-Channel Conditioning)
├── train_voxel_hist.py       # [Baseline] Train Voxel-Hist (Voxel space + AdaGN Histogram Texture Guidance)
├── train_vae.py              # [Stage-1] Train Continuous VAE (for Latent-VAE baselines and ablations)
├── train_vqvae.py            # [Stage-1] Train Discrete VQ-VAE (for Latent-VQ baselines and ablations)
├── train_ldm.py              # [Ablation] Train ACVD w.o. Voxel Space (VAE Latent space + Full 7-Channel Conditioning)
├── train_ldm_raw.py          # [Baseline] Train Latent-VAE (VAE Latent space + Basic 3-Channel Conditioning)
├── train_vq_ldm.py           # [Ablation] Train ACVD w.o. Voxel Space (VQ Latent space + Full 7-Channel Conditioning)
├── train_vq_ldm_raw.py       # [Baseline] Train Latent-VQ (VQ Latent space + Basic 3-Channel Conditioning)
├── test_vae.py               # [Evaluation] Evaluate Stage-1 Autoencoder reconstruction fidelity (L1, PSNR, SSIM)
└── test.py                   # [Evaluation] Comprehensive generative fidelity benchmark (3D FID, Masked PSNR/MAE, SSIM)
```

---

##  Environment Setup

Ensure you have Python $\ge 3.9$ and a CUDA-enabled GPU environment installed. Install the required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai monai-generative scipy numpy tqdm matplotlib
```

### Pre-trained Backbone Weights for Evaluation
To evaluate 3D Fréchet Inception Distance (FID) during testing (`test.py`), download the pre-trained **Med3D ResNet-50** weights (`resnet_50_23dataset.pth`) from the official [3DMedicalNet / Med3D repository](https://github.com/Tencent/MedicalNet) and place them inside the `weights/` directory:

```text
weights/resnet_50_23dataset.pth
```

---

##  Dataset Preparation & Multi-Channel Conditioning

> **Note on Data Release:** The processed multi-conditional 3D datasets containing over **7,000 paired nodule-anatomy crops** derived from LUNA16 and LUNA25 (clipped to standard lung window `[-1000, 400] HU` and resampled to isotropic $1\text{ mm}^3$ spacing) will be open-sourced and made publicly available in the subsequent data repository release.

All models operate on $48 \times 48 \times 48$ volumetric crops. To enforce explicit environmental context during diffusion, ACVD constructs a **7-channel conditioning tensor** $\mathbf{c} \in \mathbb{R}^{7 \times 48 \times 48 \times 48}$:

Where the individual channels represent:
1. $\mathbf{M}_{nod}$: Binary Target Nodule Core Mask
2. $\mathbf{M}_{ves}$: Fine-grained Pulmonary Vessel Mask
3. $\mathbf{M}_{air}$: Airway Branch Mask
4. $\mathbf{M}_{lung}$: Macro Lung Parenchyma Mask
5. $\mathbf{M}_{bone}$: Rib / Bone Boundary Mask
6. $\mathbf{I}_{masked}$: Background CT crop with the synthesis region zeroed out ($\text{intensity} = -0.2$)
7. $\mathbf{M}_{region}$: Morphologically dilated 3-voxel transition region mask ($\text{iterations} = 3$)

---

##  Training Workflows

All diffusion training scripts support **Distributed Data Parallel (DDP)** training with mixed-precision (`torch.amp`) acceleration.

### Step 1: Stage-1 Autoencoder Pre-training (For Latent Baselines & Ablations)
Before training latent-space models (`LDM` / `VQ_LDM`), pre-train the 3D autoencoders to compress $48^3$ crops into $8 \times 12 \times 12 \times 12$ latent representations:

```bash
# Train Continuous VAE (KL-regularized)
torchrun --nproc_per_node=4 train_vae.py

# Train Discrete VQ-VAE (Codebook-regularized)
torchrun --nproc_per_node=4 train_vqvae.py
```

### Step 2: Diffusion Model Training
Launch multi-GPU DDP training for the primary ACVD engine or comparative models.

```bash
# 1. Train ACVD (Ours - Voxel Space + Full 7-Channel Anatomical ControlNet)
torchrun --nproc_per_node=4 train_acvd.py

# 2. Train Voxel-Space Ablation (ACVD w.o. Anatomical Prior - Basic 3-Channel)
torchrun --nproc_per_node=4 train_voxel.py

# 3. Train Voxel-Hist Baseline (AdaGN Histogram Guidance)
torchrun --nproc_per_node=4 train_voxel_hist.py

# 4. Train Latent-Space Ablations (Full 7-Channel Conditioning in Latent Space)
torchrun --nproc_per_node=4 train_ldm.py
torchrun --nproc_per_node=4 train_vq_ldm.py

# 5. Train Standard Latent Baselines (Basic 3-Channel Conditioning)
torchrun --nproc_per_node=4 train_ldm_raw.py
torchrun --nproc_per_node=4 train_vq_ldm_raw.py
```

---

##  Evaluation & Benchmarking

### 1. Stage-1 Autoencoder Reconstruction Fidelity
To verify the reconstruction capability (L1, PSNR, SSIM) of the trained VAE and VQ-VAE models on the validation set:

```bash
python test_vae.py
```

### 2. Comprehensive Generative Fidelity Audit
`test.py` executes an automated benchmarking pipeline across all trained generative paradigms. It implements the **Deterministic Masked Blended Diffusion** strategy via a 50-step DDIM scheduler to ensure exact background invariance outside $\mathbf{M}_{region}$.

```bash
python test.py
```

The script automatically reports:
- **3D Fréchet Inception Distance (FID)**: Evaluated using deep feature representations extracted from a 3D ResNet-50 network pre-trained on Med3D (`models/resnet.py`).
- **Masked PSNR & Masked MAE**: Computed strictly within the localized region of interest ($\text{ROI} = \mathbf{M}_{region}$) to prevent unmasked background invariance from artificially diluting synthesis error metrics.
- **Volumetric Structural Similarity Index (SSIM)**: Evaluated across the entire 3D volume.

---

##  License

This repository is shared for double-blind academic peer review. All code and methodologies are protected under standard academic research guidelines.

import os

# Base directory on the cluster (adaptive to the user's home directory)
BASE_DIR = os.path.expanduser("~")


# Local workspace root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Dataset Paths ---
DATA_ROOT = os.path.join(BASE_DIR, "Data/ACVD")
LUNA16_ROOT = os.path.join(DATA_ROOT, "Dataset_LUNA16")
LUNA25_ROOT = os.path.join(DATA_ROOT, "Dataset_LUNA25")

# LUNA16 sub-paths
LUNA16_CONTROLNET_DATA = os.path.join(LUNA16_ROOT, "ControlNet_Data")
LUNA16_LDM_DATA_48 = os.path.join(LUNA16_ROOT, "LDM_Baseline_Data_48")

# LUNA25 sub-paths
LUNA25_CONTROLNET_DATA = os.path.join(LUNA25_ROOT, "ControlNet_Data")
LUNA25_LDM_DATA_48 = os.path.join(LUNA25_ROOT, "LDM_Baseline_Data_48")

# --- Model & Weights Paths ---
BASE_MODEL_DIR = os.path.join(BASE_DIR, "BASE/ACVD")

# VAE (Pointing to local workspace directory's trained_vae)
TRAINED_VAE_DIR = os.path.join(PROJECT_ROOT, "trained_vae")
VAE_PATH = os.path.join(TRAINED_VAE_DIR, "best_vae.pth")
VAE_CACHE_DIR = os.path.join(DATA_ROOT, "VAECache_80x80x80")

# VQ-VAE (Pointing to local workspace directory's trained_vqvae)
TRAINED_VQVAE_DIR = os.path.join(PROJECT_ROOT, "trained_vqvae")
VQVAE_PATH = os.path.join(TRAINED_VQVAE_DIR, "best_vqvae.pth")

# ResNet-50 Feature Backbone
RESNET_WEIGHTS = os.path.join(BASE_MODEL_DIR, "weights/resnet_50_23dataset.pth")

# ACVD Checkpoints (Voxel ControlNet)
TRAINED_ACVD_DIR = os.path.join(BASE_MODEL_DIR, "trained_acvd")
ACVD_CKPT_LATEST = os.path.join(TRAINED_ACVD_DIR, "checkpoint_latest.pth")
ACVD_BEST_MODEL = os.path.join(TRAINED_ACVD_DIR, "best_model.pth")

# VOXEL Checkpoints (Voxel Ablation ControlNet)
TRAINED_VOXEL_DIR = os.path.join(BASE_MODEL_DIR, "trained_voxel")
VOXEL_CKPT_LATEST = os.path.join(TRAINED_VOXEL_DIR, "checkpoint_latest.pth")
VOXEL_BEST_MODEL = os.path.join(TRAINED_VOXEL_DIR, "best_model.pth")

# LeFusion Checkpoints (Grafted version)
TRAINED_LEFUSION_DIR = os.path.join(BASE_MODEL_DIR, "trained_lefusion")
LEFUSION_CKPT_LATEST = os.path.join(TRAINED_LEFUSION_DIR, "checkpoint_latest.pth")
LEFUSION_BEST_MODEL = os.path.join(TRAINED_LEFUSION_DIR, "best_model.pth")

# Voxel_Hist Checkpoints (VOXEL with Histogram proxy)
TRAINED_VOXEL_HIST_DIR = os.path.join(BASE_MODEL_DIR, "trained_voxel_hist")
VOXEL_HIST_CKPT_LATEST = os.path.join(TRAINED_VOXEL_HIST_DIR, "checkpoint_latest.pth")
VOXEL_HIST_BEST_MODEL = os.path.join(TRAINED_VOXEL_HIST_DIR, "best_model.pth")

# ACVD_Hist Checkpoints (ACVD with Histogram proxy)
TRAINED_ACVD_HIST_DIR = os.path.join(BASE_MODEL_DIR, "trained_acvd_hist")
ACVD_HIST_CKPT_LATEST = os.path.join(TRAINED_ACVD_HIST_DIR, "checkpoint_latest.pth")
ACVD_HIST_BEST_MODEL = os.path.join(TRAINED_ACVD_HIST_DIR, "best_model.pth")

# LDM Checkpoints
TRAINED_LDM_DIR = os.path.join(BASE_MODEL_DIR, "trained_ldm")
LDM_CKPT_LATEST = os.path.join(TRAINED_LDM_DIR, "checkpoint_latest.pth")
LDM_BEST_MODEL = os.path.join(TRAINED_LDM_DIR, "best_model.pth")

# VQ_LDM Checkpoints
TRAINED_VQ_LDM_DIR = os.path.join(BASE_MODEL_DIR, "trained_vq_ldm")
VQ_LDM_CKPT_LATEST = os.path.join(TRAINED_VQ_LDM_DIR, "checkpoint_latest.pth")
VQ_LDM_BEST_MODEL = os.path.join(TRAINED_VQ_LDM_DIR, "best_model.pth")

# LDM Raw Checkpoints
TRAINED_LDM_RAW_DIR = os.path.join(BASE_MODEL_DIR, "trained_ldm_raw")
LDM_RAW_CKPT_LATEST = os.path.join(TRAINED_LDM_RAW_DIR, "checkpoint_latest.pth")
LDM_RAW_BEST_MODEL = os.path.join(TRAINED_LDM_RAW_DIR, "best_model.pth")

# VQ_LDM Raw Checkpoints
TRAINED_VQ_LDM_RAW_DIR = os.path.join(BASE_MODEL_DIR, "trained_vq_ldm_raw")
VQ_LDM_RAW_CKPT_LATEST = os.path.join(TRAINED_VQ_LDM_RAW_DIR, "checkpoint_latest.pth")
VQ_LDM_RAW_BEST_MODEL = os.path.join(TRAINED_VQ_LDM_RAW_DIR, "best_model.pth")



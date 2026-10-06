import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.metrics import SSIMMetric
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from generative.networks.nets import AutoencoderKL, VQVAE

import warnings
warnings.filterwarnings("ignore")

# ================= 1. Paths and Config =================
from config import LUNA16_CONTROLNET_DATA, VAE_PATH, VQVAE_PATH, BASE_MODEL_DIR

VAL_DATA_DIR = LUNA16_CONTROLNET_DATA
VIS_DIR = os.path.join(BASE_MODEL_DIR, "vae_48_verify")
os.makedirs(VIS_DIR, exist_ok=True)

CROP_SIZE = 48

# ================= 2. Dataset =================
class NpyDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        try:
            data = np.load(fpath, allow_pickle=True).item()
            # 1. Load 64^3 original crop
            img_64 = data['gt_image']
            # 2. Center crop to 48^3
            d = (64 - CROP_SIZE) // 2
            img_48 = img_64[:, d:d+CROP_SIZE, d:d+CROP_SIZE, d:d+CROP_SIZE]
            return torch.from_numpy(img_48).float()
        except Exception as e:
            # Exception handling
            return torch.zeros((1, CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=torch.float32)

def main():
    # GPU check
    num_gpus = torch.cuda.device_count()
    print(f" Detected {num_gpus} GPUs.")
    device_ids = list(range(num_gpus))
    main_device = torch.device("cuda:0" if num_gpus > 0 else "cpu")

    # 1. Load VAE
    print(" [Step 1] Loading VAE (8 channels)...")
    vae = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1,
        num_channels=(64, 128, 256), latent_channels=8,
        num_res_blocks=2, norm_num_groups=32, attention_levels=(False, False, True),
    ).to(main_device)

    vae_loaded = False
    if os.path.exists(VAE_PATH):
        ckpt_vae = torch.load(VAE_PATH, map_location=main_device, weights_only=False)
        state_dict_vae = ckpt_vae['model'] if 'model' in ckpt_vae else ckpt_vae
        vae.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae.items()}, strict=False)
        vae.eval()
        if num_gpus > 1:
            vae = nn.DataParallel(vae, device_ids=device_ids)
        vae_loaded = True
        print(" VAE Loaded Successfully.")
    else:
        print(f" VAE checkpoint not found at: {VAE_PATH}")

    # 2. Load VQ-VAE
    print(" [Step 2] Loading VQ-VAE (8 embedding channels)...")
    vqvae = VQVAE(
        spatial_dims=3, in_channels=1, out_channels=1,
        num_channels=(64, 128, 256), num_res_channels=(64, 128, 256),
        num_res_layers=2, num_embeddings=8192, embedding_dim=8,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (1, 3, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (1, 3, 1, 1, 0)),
    ).to(main_device)

    vqvae_loaded = False
    if os.path.exists(VQVAE_PATH):
        ckpt_vqvae = torch.load(VQVAE_PATH, map_location=main_device, weights_only=False)
        state_dict_vqvae = ckpt_vqvae['model'] if 'model' in ckpt_vqvae else ckpt_vqvae
        vqvae.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vqvae.items()}, strict=False)
        vqvae.eval()
        if num_gpus > 1:
            vqvae = nn.DataParallel(vqvae, device_ids=device_ids)
        vqvae_loaded = True
        print(" VQ-VAE Loaded Successfully.")
    else:
        print(f" VQ-VAE checkpoint not found at: {VQVAE_PATH}")

    if not vae_loaded and not vqvae_loaded:
        print(" Error: Neither VAE nor VQ-VAE checkpoints were loaded. Exiting.")
        return

    # Get validation files
    val_files = sorted(glob.glob(os.path.join(VAL_DATA_DIR, "*.npy")))
    if not val_files:
        print(f" Validation data not found: {VAL_DATA_DIR}")
        return
    print(f" Found {len(val_files)} validation samples. Testing on the ENTIRE dataset...")

    # ================= 3. DataLoader =================
    # Batch configuration
    batch_size_per_gpu = 32
    batch_size = max(1, batch_size_per_gpu * num_gpus)

    # CPU worker configuration
    num_workers = min(40, os.cpu_count() - 2) if os.cpu_count() is not None else 8

    print(f" Batch Size: {batch_size} (per GPU: {batch_size_per_gpu}) | CPU Workers: {num_workers}")

    dataset = NpyDataset(val_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # ================= 4. Evaluation Metrics =================
    vae_ssim_metric = SSIMMetric(spatial_dims=3, data_range=2.0)
    vqvae_ssim_metric = SSIMMetric(spatial_dims=3, data_range=2.0)

    vae_metrics = {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "count": 0}
    vqvae_metrics = {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "count": 0}

    # ================= 5. Validation =================
    saved_visualizations = 0

    for batch_idx, img_t in enumerate(tqdm(loader, desc="Testing VAEs")):
        img_t = img_t.to(main_device)

        with torch.no_grad(), autocast("cuda"):
            # VAE inference and evaluation
            if vae_loaded:
                recon_vae, _, _ = vae(img_t)
                vae_ssim_metric(y_pred=recon_vae, y=img_t)


                l1_val = torch.mean(torch.abs(recon_vae - img_t), dim=(1,2,3,4))
                mse_val = torch.mean((recon_vae - img_t) ** 2, dim=(1,2,3,4))
                psnr_val = 20 * torch.log10(2.0 / (torch.sqrt(mse_val) + 1e-8))

                vae_metrics["l1"] += torch.sum(l1_val).item()
                vae_metrics["mse"] += torch.sum(mse_val).item()
                vae_metrics["psnr"] += torch.sum(psnr_val).item()
                vae_metrics["count"] += img_t.size(0)

            # VQ-VAE inference and evaluation
            if vqvae_loaded:
                recon_vqvae, _ = vqvae(img_t)
                vqvae_ssim_metric(y_pred=recon_vqvae, y=img_t)


                l1_val = torch.mean(torch.abs(recon_vqvae - img_t), dim=(1,2,3,4))
                mse_val = torch.mean((recon_vqvae - img_t) ** 2, dim=(1,2,3,4))
                psnr_val = 20 * torch.log10(2.0 / (torch.sqrt(mse_val) + 1e-8))

                vqvae_metrics["l1"] += torch.sum(l1_val).item()
                vqvae_metrics["mse"] += torch.sum(mse_val).item()
                vqvae_metrics["psnr"] += torch.sum(psnr_val).item()
                vqvae_metrics["count"] += img_t.size(0)

            # Print shape info on first step
            if batch_idx == 0:
                print("\n" + "="*50)
                print(f" Input Image Shape: {img_t.shape} -> (B, C, D, H, W)")
                if vae_loaded:
                    print(f" VAE Recon Shape:   {recon_vae.shape}")
                if vqvae_loaded:
                    print(f" VQ-VAE Recon Shape: {recon_vqvae.shape}")
                print("="*50 + "\n")

            # Save visualization for first 5 samples
            if batch_idx == 0 and saved_visualizations < 5:
                for idx in range(min(5, img_t.size(0))):
                    img_np = img_t[idx, 0].cpu().numpy()
                    cz = CROP_SIZE // 2 # Center slice

                    num_cols = 1 + (2 if vae_loaded else 0) + (2 if vqvae_loaded else 0)
                    fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

                    col_idx = 0
                    # [Col 0] Original
                    axs[col_idx].imshow(img_np[cz], cmap='gray', vmin=-1, vmax=1)
                    axs[col_idx].set_title("Original 48^3 Crop")
                    axs[col_idx].axis('off')
                    col_idx += 1

                    # [Col 1-2] VAE Recon & Error
                    if vae_loaded:
                        recon_vae_np = recon_vae[idx, 0].cpu().numpy()
                        diff_vae_np = np.abs(img_np - recon_vae_np)

                        axs[col_idx].imshow(recon_vae_np[cz], cmap='gray', vmin=-1, vmax=1)
                        axs[col_idx].set_title("VAE Recon")
                        axs[col_idx].axis('off')
                        col_idx += 1

                        axs[col_idx].imshow(diff_vae_np[cz], cmap='jet', vmin=0, vmax=1)
                        axs[col_idx].set_title("VAE Error Heatmap")
                        axs[col_idx].axis('off')
                        col_idx += 1

                    # [Col 3-4] VQ-VAE Recon & Error
                    if vqvae_loaded:
                        recon_vq_np = recon_vqvae[idx, 0].cpu().numpy()
                        diff_vq_np = np.abs(img_np - recon_vq_np)

                        axs[col_idx].imshow(recon_vq_np[cz], cmap='gray', vmin=-1, vmax=1)
                        axs[col_idx].set_title("VQ-VAE Recon")
                        axs[col_idx].axis('off')
                        col_idx += 1

                        axs[col_idx].imshow(diff_vq_np[cz], cmap='jet', vmin=0, vmax=1)
                        axs[col_idx].set_title("VQ-VAE Error Heatmap")
                        axs[col_idx].axis('off')
                        col_idx += 1

                    plt.tight_layout()
                    plt.savefig(os.path.join(VIS_DIR, f"compare_vae_vqvae_{saved_visualizations}.png"))
                    plt.close()
                    saved_visualizations += 1

    print("\n [Verification Complete]")
    if vae_loaded:
        v_count = vae_metrics["count"]
        mean_ssim = vae_ssim_metric.aggregate().item()
        mean_l1 = vae_metrics["l1"] / v_count
        mean_mse = vae_metrics["mse"] / v_count
        mean_psnr = vae_metrics["psnr"] / v_count
        print(f"⭐ VAE Results on {v_count} samples:")
        print(f"  - SSIM: {mean_ssim:.4f}")
        print(f"  - MAE (L1): {mean_l1:.4f}")
        print(f"  - MSE: {mean_mse:.6f}")
        print(f"  - PSNR: {mean_psnr:.2f} dB")

    if vqvae_loaded:
        vq_count = vqvae_metrics["count"]
        mean_ssim = vqvae_ssim_metric.aggregate().item()
        mean_l1 = vqvae_metrics["l1"] / vq_count
        mean_mse = vqvae_metrics["mse"] / vq_count
        mean_psnr = vqvae_metrics["psnr"] / vq_count
        print(f"⭐ VQ-VAE Results on {vq_count} samples:")
        print(f"  - SSIM: {mean_ssim:.4f}")
        print(f"  - MAE (L1): {mean_l1:.4f}")
        print(f"  - MSE: {mean_mse:.6f}")
        print(f"  - PSNR: {mean_psnr:.2f} dB")

if __name__ == "__main__":
    main()

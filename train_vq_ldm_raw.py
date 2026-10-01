import os
import glob
import torch
from contextlib import nullcontext
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import ndimage

# MONAI & Generative
from monai import transforms
from monai.utils import set_determinism, first
from generative.networks.nets import ControlNet, VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import warnings
warnings.filterwarnings("ignore")

# ================= 1. Stats and Config =================
LATENT_STATS = {
    "mean": 0.0,
    "scale": 1.0
}
LUNG_MEAN_PIXEL = -0.2

def norm_latent(z):
    return (z - LATENT_STATS["mean"]) * LATENT_STATS["scale"]

def denorm_latent(z):
    return (z / LATENT_STATS["scale"]) + LATENT_STATS["mean"]

# ================= 2. Paths =================
from config import LUNA25_CONTROLNET_DATA, VQVAE_PATH, TRAINED_VQ_LDM_RAW_DIR, VQ_LDM_RAW_CKPT_LATEST, VQ_LDM_RAW_BEST_MODEL

TRAIN_DIR = LUNA25_CONTROLNET_DATA
VQVAE_PATH = VQVAE_PATH

SAVE_DIR = TRAINED_VQ_LDM_RAW_DIR
IMG_SAVE_DIR = os.path.join(SAVE_DIR, "vis")
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

CKPT_PATH = VQ_LDM_RAW_CKPT_LATEST
BEST_CKPT_PATH = VQ_LDM_RAW_BEST_MODEL

# Distributed training setup
BATCH_SIZE_PER_GPU = 16
LR = 1e-4
MAX_EPOCHS = 1000

VAL_INTERVAL = 5



def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ================= 3. Model Wrapper =================
class LDMControlSystem(nn.Module):
    def __init__(self, base_unet, controlnet):
        super().__init__()
        self.base_unet = base_unet
        self.controlnet = controlnet
    def forward(self, x, timesteps, cond_48):
        down_res, mid_res = self.controlnet(x=x, timesteps=timesteps, controlnet_cond=cond_48)
        return self.base_unet(x=x, timesteps=timesteps,
                              down_block_additional_residuals=down_res,
                              mid_block_additional_residual=mid_res)


class LDMBaselineDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.mode = mode
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        import random
        rng = random.Random(42)
        rng.shuffle(all_files)

        split_idx = int(len(all_files) * 0.8)
        if mode == "train":
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

        self.aug = transforms.Compose([
            transforms.RandFlip(prob=0.5, spatial_axis=[0, 1, 2]),
            transforms.RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        img = torch.from_numpy(data['gt_image']).float() # (1, 64, 64, 64)
        m = torch.from_numpy(data['conditions']).float() # (5, 64, 64, 64)

        # Crop to 48^3 with random jitter
        if self.mode == "train":
            dz = np.random.randint(6, 11)  # [6, 10]
            dy = np.random.randint(6, 11)
            dx = np.random.randint(6, 11)
        else:
            dz = dy = dx = 8
        img = img[:, dz:dz+48, dy:dy+48, dx:dx+48]
        m = m[:, dz:dz+48, dy:dy+48, dx:dx+48]

        if self.mode == "train":
            seed = np.random.randint(2147483647)
            set_determinism(seed=seed); img = self.aug(img)
            set_determinism(seed=seed); m = self.aug(m)

        nod_mask = m[0:1] # M_nod

        # Compute 3px dilated region mask
        #
        m_np = nod_mask[0].numpy()
        m_region = torch.from_numpy(ndimage.binary_dilation(m_np, iterations=3).astype(np.float32)).unsqueeze(0)

        # Compute masked background
        i_masked = img * (1.0 - m_region) + LUNG_MEAN_PIXEL * m_region

        # 3 channels configuration
        cond_48 = torch.cat([nod_mask, i_masked, m_region], dim=0)

        return {"img": img, "cond": cond_48, "nodule_gt": nod_mask, "m_region": m_region}


@torch.no_grad()
def smart_vis(full_model, vae, val_loader, epoch, device, rank):
    if rank != 0: return
    full_model.eval()
    batch = first(val_loader)

    r_idx = torch.argmax(torch.sum(batch['nodule_gt'], dim=(2,3,4))).item()
    img_gt_raw = batch['img'][r_idx:r_idx+1].to(device)
    cond = batch['cond'][r_idx:r_idx+1].to(device)
    nod_gt = batch['nodule_gt'][r_idx:r_idx+1].to(device)
    m_region = batch['m_region'][r_idx:r_idx+1].to(device)


    with torch.no_grad():
        z_gt_raw = vae.encode(img_gt_raw)
        z_gt_quant, _ = vae.quantize(z_gt_raw)
        z_gt = norm_latent(z_gt_quant)

    # Latent mask
    m_region_latent = F.interpolate(m_region, size=z_gt.shape[2:], mode='nearest')

    val_sch = DDIMScheduler(num_train_timesteps=1000)
    val_sch.set_timesteps(50)
    z_t = torch.randn_like(z_gt)

    for i, t in enumerate(val_sch.timesteps):
        t_tensor = torch.full((1,), t, device=device).long()
        noise_pred = full_model(z_t, t_tensor, cond)
        z_t, _ = val_sch.step(noise_pred, t, z_t)

        # Latent space blending
        if i < len(val_sch.timesteps) - 1:
            prev_t = val_sch.timesteps[i+1]
            noise = torch.randn_like(z_gt)
            z_ref = val_sch.add_noise(z_gt, noise, torch.full((1,), prev_t, device=device).long())
            z_t = m_region_latent * z_t + (1.0 - m_region_latent) * z_ref

    img_gt = vae.decode(denorm_latent(z_gt))
    img_gen = vae.decode(denorm_latent(z_t))

    cz = torch.argmax(torch.sum(nod_gt[0, 0], dim=(1,2))).item()
    diff = torch.abs(img_gen - img_gt)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img_gt[0,0,cz].cpu(), cmap='gray', vmin=-1, vmax=1); axs[0].set_title("GT (VQ-VAE Recon)")
    axs[1].imshow(nod_gt[0,0,cz].cpu(), cmap='gray'); axs[1].set_title("Target Mask")
    axs[2].imshow(img_gen[0,0,cz].cpu(), cmap='gray', vmin=-1, vmax=1); axs[2].set_title("VQ-LDM Gen (Eq.5 Blended)")
    axs[3].imshow(diff[0,0,cz].cpu(), cmap='jet', vmin=0, vmax=0.5); axs[3].set_title("Texture Residual")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_SAVE_DIR, f"ep_{epoch}.png"))
    plt.close()

# ================= 6. Main =================
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if local_rank == 0:
        print(f"VQ-LDM Raw (3 channels) Starting")

    # Prepare data
    train_ds = LDMBaselineDataset(TRAIN_DIR, mode="train")
    val_ds = LDMBaselineDataset(TRAIN_DIR, mode="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=DistributedSampler(train_ds), num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=4)

    # 1. LDM init
    base_unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=8, out_channels=8,
        num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)
    ).to(device)

    controlnet = ControlNet(
        spatial_dims=3, in_channels=8,
        conditioning_embedding_in_channels=3,
        conditioning_embedding_num_channels=(16, 32, 64),
        num_channels=(32, 64, 128),
        num_res_blocks=2, attention_levels=(False, False, True)
    ).to(device)

    controlnet.load_state_dict(base_unet.state_dict(), strict=False)
    full_model = DDP(LDMControlSystem(base_unet, controlnet).to(device), device_ids=[local_rank], find_unused_parameters=True)

    # 2. Load VQ-VAE
    vae = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        num_res_channels=(64, 128, 256),
        num_res_layers=2,
        num_embeddings=8192,
        embedding_dim=8,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (1, 3, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (1, 3, 1, 1, 0)),
    ).to(device)
    vae.load_state_dict(torch.load(VQVAE_PATH, map_location=device, weights_only=False), strict=False)
    vae.eval()

    # Load checkpoint
    best_val_loss = float('inf')
    start_epoch = 1
    checkpoint = None
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # 3. Calibration of latent statistics
    if checkpoint is not None and 'latent_stats' in checkpoint:
        LATENT_STATS["mean"] = checkpoint['latent_stats']['mean']
        LATENT_STATS["scale"] = checkpoint['latent_stats']['scale']
        if local_rank == 0:
            print(f"Loaded VQ-LDM Latent Stats from checkpoint: Mean={LATENT_STATS['mean']:.6f}, Scale Factor={LATENT_STATS['scale']:.6f}")
    else:
        if local_rank == 0:
            print("Calibrating VQ-VAE latent space statistics over the ENTIRE training dataset...")

        local_sum = 0.0
        local_sum_sq = 0.0
        local_count = 0

        pbar = tqdm(train_loader, desc="Calibrating statistics", disable=(local_rank != 0))
        with torch.no_grad():
            for batch in pbar:
                img = batch['img'].to(device)
                z0_raw = vae.encode(img)
                z0_quant, _ = vae.quantize(z0_raw)
                local_sum += z0_quant.sum().double()
                local_sum_sq += (z0_quant ** 2).sum().double()
                local_count += z0_quant.numel()

        local_stats = torch.tensor([local_sum, local_sum_sq, float(local_count)], device=device, dtype=torch.double)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        global_sum = local_stats[0].item()
        global_sum_sq = local_stats[1].item()
        global_count = local_stats[2].item()

        global_mean = global_sum / global_count
        global_var = (global_sum_sq / global_count) - (global_mean ** 2)
        global_std = np.sqrt(max(0.0, global_var))

        LATENT_STATS["mean"] = global_mean
        LATENT_STATS["scale"] = 1.0 / (global_std + 1e-8)

        if local_rank == 0:
            print(f"Calibrated VQ-LDM Latent Stats (Entire Train Set): Mean={LATENT_STATS['mean']:.6f}, Std={global_std:.6f}, Scale Factor={LATENT_STATS['scale']:.6f}")

    # 4. Optimizers
    optimizer = AdamW(full_model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scheduler_ddpm = DDPMScheduler(num_train_timesteps=1000)
    scaler = GradScaler()
    if checkpoint is not None:
        full_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if local_rank == 0: print(f"Resumed from Ep {start_epoch}, Current Best: {best_val_loss:.4f}")

    global_batch_size = 64
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    accum_steps = max(1, global_batch_size // (BATCH_SIZE_PER_GPU * world_size))
    if local_rank == 0:
        print(f"Gradient Accumulation: Global BS = {global_batch_size}, Local BS = {BATCH_SIZE_PER_GPU}, World Size = {world_size}, Accum Steps = {accum_steps}")

    # 5. Training loop
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loader.sampler.set_epoch(epoch)
        full_model.train()

        pbar = tqdm(train_loader, desc=f"Ep {epoch} [Train]", disable=(local_rank!=0))
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            img = batch['img'].to(device)
            cond = batch['cond'].to(device)
            m_region = batch['m_region'].to(device)

            # VQ-VAE encode
            with torch.no_grad():
                z0_raw = vae.encode(img)
                z0_quant, _ = vae.quantize(z0_raw)
                z0 = norm_latent(z0_quant)

            t = torch.randint(0, 1000, (z0.shape[0],), device=device).long()
            noise = torch.randn_like(z0)
            zt = scheduler_ddpm.add_noise(z0, noise, t)

            is_accum_step = (step + 1) % accum_steps != 0
            is_last_batch = (step + 1) == len(train_loader)
            should_sync = not is_accum_step or is_last_batch

            context = full_model.no_sync() if (not should_sync and isinstance(full_model, DDP)) else nullcontext()

            with context:
                with autocast("cuda"):
                    pred = full_model(zt, t, cond)
                    m_region_12 = F.interpolate(m_region, size=z0.shape[2:], mode='nearest')
                    mse = F.mse_loss(pred, noise, reduction='none')
                    loss_w = torch.where(m_region_12 > 0.5, 20.0, 1.0)
                    loss = (mse * loss_w).mean()
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

            if should_sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"L": f"{(loss.item() * accum_steps):.4f}"})

        # Distributed validation
        if epoch % VAL_INTERVAL == 0:
            full_model.eval()
            val_loss_accum = 0.0
            val_steps = 0
            val_generator = torch.Generator(device=device).manual_seed(42 + local_rank)
            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch['img'].to(device)
                    v_cond = v_batch['cond'].to(device)
                    v_m_region = v_batch['m_region'].to(device)

                    with torch.no_grad():
                        v_z0_raw = vae.encode(v_img)
                        v_z0_quant, _ = vae.quantize(v_z0_raw)
                        v_z0 = norm_latent(v_z0_quant)

                    v_t = torch.randint(0, 1000, (v_z0.shape[0],), device=device, generator=val_generator).long()
                    v_noise = torch.randn(v_z0.shape, device=device, generator=val_generator)
                    v_zt = scheduler_ddpm.add_noise(v_z0, v_noise, v_t)

                    with autocast("cuda"):
                        v_pred = full_model(v_zt, v_t, v_cond)
                        v_m_region_12 = F.interpolate(v_m_region, size=v_z0.shape[2:], mode='nearest')
                        v_mse = F.mse_loss(v_pred, v_noise, reduction='none')
                        v_loss_w = torch.where(v_m_region_12 > 0.5, 20.0, 1.0)
                        v_loss = (v_mse * v_loss_w).mean()

                    val_loss_accum += v_loss.item()
                    val_steps += 1

            # Reduce loss across GPUs
            val_tensor = torch.tensor(val_loss_accum / max(1, val_steps)).to(device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            val_loss_avg = val_tensor.item() / dist.get_world_size()

            if local_rank == 0:
                print(f"   [Val] Epoch {epoch} | Loss: {val_loss_avg:.4f} (Best: {best_val_loss:.4f})")
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': full_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'latent_stats': {
                            'mean': LATENT_STATS['mean'],
                            'scale': LATENT_STATS['scale']
                        }
                    }, BEST_CKPT_PATH)
                    print(f"New Best Model Saved (Loss: {best_val_loss:.4f})")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': full_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'latent_stats': {
                        'mean': LATENT_STATS['mean'],
                        'scale': LATENT_STATS['scale']
                    }
                }, CKPT_PATH)
                smart_vis(full_model.module, vae, val_loader, epoch, device, local_rank)

        scheduler.step()
        dist.barrier()

    cleanup_ddp()

if __name__ == "__main__":
    main()

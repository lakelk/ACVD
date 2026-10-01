import os
import glob
import torch
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
from contextlib import nullcontext
from scipy import ndimage


# MONAI & Generative
from monai import transforms
from monai.utils import set_determinism
from generative.networks.nets import ControlNet, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import warnings
warnings.filterwarnings("ignore")

# ================= 1. Paths and Config =================
from config import LUNA25_CONTROLNET_DATA, LUNA16_CONTROLNET_DATA, TRAINED_ACVD_DIR, ACVD_CKPT_LATEST, ACVD_BEST_MODEL

DATA_DIR = LUNA25_CONTROLNET_DATA
VAL_DATA_DIR = LUNA16_CONTROLNET_DATA

SAVE_DIR = TRAINED_ACVD_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
IMG_SAVE_DIR = os.path.join(SAVE_DIR, "vis_progress")
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

CKPT_PATH = ACVD_CKPT_LATEST
BEST_CKPT_PATH = ACVD_BEST_MODEL

BATCH_SIZE_PER_GPU = 8
LR = 1e-4
MAX_EPOCHS = 1000
VAL_INTERVAL = 5
LUNG_MEAN_PIXEL = -0.2  # Lung parenchyma intensity background

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# ================= 2. Model Definition =================
class PixelControlSystem(nn.Module):
    def __init__(self, unet, controlnet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
    def forward(self, x, timesteps, cond):
        down_res, mid_res = self.controlnet(x=x, timesteps=timesteps, controlnet_cond=cond)
        return self.unet(x=x, timesteps=timesteps,
                         down_block_additional_residuals=down_res,
                         mid_block_additional_residual=mid_res)

# ================= 3. Dataset =================
class PixelDataset(Dataset):
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
        img = torch.from_numpy(data['gt_image']).float()
        masks = torch.from_numpy(data['conditions']).float()

        # Crop to 48^3 with random jitter
        if self.mode == "train":
            dz = np.random.randint(6, 11)  # [6, 10]
            dy = np.random.randint(6, 11)
            dx = np.random.randint(6, 11)
        else:
            dz = dy = dx = 8
        img = img[:, dz:dz+48, dy:dy+48, dx:dx+48]
        masks = masks[:, dz:dz+48, dy:dy+48, dx:dx+48]

        if self.mode == "train":
            seed = np.random.randint(2147483647)
            set_determinism(seed=seed); img = self.aug(img)
            set_determinism(seed=seed); masks = self.aug(masks)

        nod_gt = masks[0:1] # Solid nodule mask

        m_np = nod_gt[0].numpy()
        mask_dilated = torch.from_numpy(ndimage.binary_dilation(m_np, iterations=3).astype(np.float32)).unsqueeze(0)

        masked_img = img * (1.0 - mask_dilated) + LUNG_MEAN_PIXEL * mask_dilated

        cond = torch.cat([nod_gt, masks[1:5], masked_img, mask_dilated], dim=0)

        return {"img": img, "cond": cond, "nodule_gt": nod_gt, "mask_dilated": mask_dilated}

# ================= 4. Visualization (Inpainting) =================
@torch.no_grad()
def smart_vis(model, val_loader, epoch, device, rank):
    if rank != 0: return
    model.eval()

    # 1. Random sample selection
    val_iter = iter(val_loader)
    for _ in range(np.random.randint(1, 5)):
        try: batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

    r_idx = np.random.randint(0, batch['img'].shape[0])
    img_gt = batch['img'][r_idx:r_idx+1].to(device)
    cond = batch['cond'][r_idx:r_idx+1].to(device)
    nod_gt = batch['nodule_gt'][r_idx:r_idx+1].to(device) # Original mask
    hole_mask = batch['mask_dilated'][r_idx:r_idx+1].to(device) # Dilated mask

    val_sch = DDIMScheduler(num_train_timesteps=1000)
    val_sch.set_timesteps(50)
    x_t = torch.randn_like(img_gt)


    for t in val_sch.timesteps:
        t_t = torch.full((1,), t, device=device).long()
        noise_pred = model(x_t, t_t, cond)
        x_t_gen, _ = val_sch.step(noise_pred, t, x_t)

        # Track background reference
        noise_ref = torch.randn_like(img_gt)
        x_t_ref = val_sch.add_noise(img_gt, noise_ref, t_t)

        # Fusion logic
        x_t = x_t_gen * hole_mask + x_t_ref * (1.0 - hole_mask)

        # Track raw model output
        final_raw_output = x_t_gen

    img_gen = x_t

    # 3. Find nodule center slice
    cz = torch.argmax(torch.sum(nod_gt[0, 0], dim=(1,2))).item()
    if cz!=0:
        cz=cz-1
    diff = torch.abs(img_gen - img_gt)

    # ================== Plotting =================
    fig, axs = plt.subplots(1, 6, figsize=(30, 5))
    plt.suptitle(f"Epoch {epoch} | Pixel-to-Pixel Synthesis Audit", fontsize=16)

    # Original GT
    axs[0].imshow(img_gt[0,0,cz].cpu(), cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title("1. Original CT (GT)")

    # Nodule GT
    axs[1].imshow(nod_gt[0,0,cz].cpu(), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("2. Target Nodule Mask")

    # Dilated Mask
    axs[2].imshow(hole_mask[0,0,cz].cpu(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("3. Dilated Inpaint Area")

    # Raw Output
    #
    axs[3].imshow(final_raw_output[0,0,cz].cpu(), cmap='gray', vmin=-1, vmax=1)
    axs[3].set_title("4. Raw Model Output\n(No Inpainting)")

    # Fused Result
    axs[4].imshow(img_gen[0,0,cz].cpu(), cmap='gray', vmin=-1, vmax=1)
    axs[4].set_title("5. Final Fused Result")

    # [5] Difference Map
    axs[5].imshow(diff[0,0,cz].cpu(), cmap='jet')
    axs[5].set_title("6. Difference Map")

    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_SAVE_DIR, f"pixel_audit_ep_{epoch}.png"))
    plt.close()

# ================= 5. Main =================
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if local_rank == 0:
        print("🚀 [Phase 3] Weighted Loss Pixel ControlNet Launching...")

    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        num_channels=(32, 64, 128), attention_levels=(False, False, True),
        num_res_blocks=2, num_head_channels=32,
    ).to(device)

    controlnet = ControlNet(
        spatial_dims=3, in_channels=1, conditioning_embedding_in_channels=7,
        conditioning_embedding_num_channels=(16,),
        num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)
    ).to(device)

    full_model = DDP(PixelControlSystem(unet, controlnet).to(device), device_ids=[local_rank], find_unused_parameters=True)

    train_ds = PixelDataset(DATA_DIR, mode="train")
    val_ds = PixelDataset(DATA_DIR, mode="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=DistributedSampler(train_ds), num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PER_GPU, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=4)

    optimizer = AdamW(full_model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scheduler_ddpm = DDPMScheduler(num_train_timesteps=1000)
    scaler = GradScaler()

    start_epoch = 1
    best_val_loss = float('inf')

    global_batch_size = 64
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    accum_steps = max(1, global_batch_size // (BATCH_SIZE_PER_GPU * world_size))
    if local_rank == 0:
        print(f"Gradient Accumulation: Global BS = {global_batch_size}, Local BS = {BATCH_SIZE_PER_GPU}, World Size = {world_size}, Accum Steps = {accum_steps}")


    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        full_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if local_rank == 0: print(f"✅ Resumed from Epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loader.sampler.set_epoch(epoch)
        full_model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", disable=(local_rank!=0))

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            img, cond, mask_focus = batch['img'].to(device), batch['cond'].to(device), batch['mask_dilated'].to(device)
            t = torch.randint(0, 1000, (img.shape[0],), device=device).long()
            noise = torch.randn_like(img)
            xt = scheduler_ddpm.add_noise(img, noise, t)

            is_accum_step = (step + 1) % accum_steps != 0
            is_last_batch = (step + 1) == len(train_loader)
            should_sync = not is_accum_step or is_last_batch

            context = full_model.no_sync() if (not should_sync and isinstance(full_model, DDP)) else nullcontext()

            with context:
                with autocast("cuda"):
                    pred = full_model(xt, t, cond)
                    mse = F.mse_loss(pred, noise, reduction='none')

                    # Weighted Loss
                    # Inside hole weight: 20.0, outside weight: 1.0
                    #
                    weights = torch.ones_like(mask_focus) + mask_focus * 19.0
                    loss = (mse * weights).mean()
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

            if should_sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"L": f"{(loss.item() * accum_steps):.4f}"})

        if epoch % VAL_INTERVAL == 0:
            full_model.eval()
            val_loss_accum = 0.0
            val_steps = 0
            val_generator = torch.Generator(device=device).manual_seed(42 + local_rank)
            with torch.no_grad():
                for v_batch in val_loader:
                    v_img, v_cond, v_mask = v_batch['img'].to(device), v_batch['cond'].to(device), v_batch['mask_dilated'].to(device)
                    v_t = torch.randint(0, 1000, (v_img.shape[0],), device=device, generator=val_generator).long()
                    v_noise = torch.randn(v_img.shape, device=device, generator=val_generator)
                    v_xt = scheduler_ddpm.add_noise(v_img, v_noise, v_t)
                    v_pred = full_model(v_xt, v_t, v_cond)

                    v_mse = F.mse_loss(v_pred, v_noise, reduction='none')
                    v_weights = torch.ones_like(v_mask) + v_mask * 19.0
                    v_loss = (v_mse * v_weights).mean()

                    val_loss_accum += v_loss.item()
                    val_steps += 1

            val_loss_avg = torch.tensor(val_loss_accum / max(1, val_steps)).to(device)
            dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
            val_loss_avg = val_loss_avg.item() / dist.get_world_size()

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
                    }, BEST_CKPT_PATH)
                    print(f"🔥 New Best Model Saved (Loss: {best_val_loss:.4f})")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': full_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, CKPT_PATH)
                smart_vis(full_model.module, val_loader, epoch, device, local_rank)

        scheduler.step()
        dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

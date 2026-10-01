import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings


from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.nn import L1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


from monai import transforms
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset as MonaiDataset
from monai.metrics import SSIMMetric


from generative.losses import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

warnings.filterwarnings("ignore")

# ================= 1. Global Config =================
#

from config import BASE_DIR, LUNA25_ROOT, LUNA16_ROOT, TRAINED_VAE_DIR, VAE_CACHE_DIR

# Disk cache setup
CACHE_DIR = VAE_CACHE_DIR
SAVE_DIR = TRAINED_VAE_DIR

# Hyperparameters
BATCH_SIZE_PER_GPU = 16
MAX_EPOCHS = 200
VAL_INTERVAL = 1

# Optimizer config
LR = 2e-4
W_L1   = 0.8        # L1 loss weight
W_PERC = 0.3        # Perceptual loss weight
W_KL   = 1e-6       # KL weight
W_ADV  = 0.3        # Adversarial loss weight
ADV_START_EPOCH = 5 #

# Sampling configuration
PATCH_SIZE_MEM = (80, 80, 80)
PATCH_SIZE_TRAIN = (48, 48, 48)
SAMPLES_POS = 4
SAMPLES_NEG = 4

# ================= 2. Distributed Setup =================
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def _select_fn(x): return x > -1.0
def _identity_collate(x): return x

# ================= 3. Data Mining =================

def get_mining_transforms(is_val=False):
    keys = ["image", "label"]
    return transforms.Compose([
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=400.0, b_min=-1.0, b_max=1.0, clip=True),
        transforms.CropForegroundd(keys=keys, source_key="image", select_fn=_select_fn),
        transforms.RandCropByPosNegLabeld(
            keys=keys, label_key="label",
            spatial_size=PATCH_SIZE_TRAIN if is_val else PATCH_SIZE_MEM,
            pos=1, neg=1,
            num_samples=SAMPLES_POS + SAMPLES_NEG,
            image_key="image", image_threshold=-1.0,
        ),
    ])

def mine_patches_to_disk(files, mode="train"):
    """
    Mine patches and save to disk.
    """
    is_val = (mode == "val")
    save_subdir = os.path.join(CACHE_DIR, mode)

    # Skip if already mined
    if os.path.exists(save_subdir) and len(glob.glob(os.path.join(save_subdir, "*.npy"))) > 100:
        print(f">>> [Disk] Mined cache detected in {save_subdir}, skipping.")
        print(f"    (Delete {CACHE_DIR} to re-mine)")
        npy_files = glob.glob(os.path.join(save_subdir, "*.npy"))
        return npy_files

    os.makedirs(save_subdir, exist_ok=True)
    miner = get_mining_transforms(is_val=is_val)
    temp_ds = MonaiDataset(data=files, transform=miner)

    # Acceleration
    loader = MonaiLoader(temp_ds, batch_size=1, num_workers=28, collate_fn=_identity_collate)

    npy_files = []
    desc = f"Mining {mode} (to Disk)"

    print(f">>> [Disk] Mining patches to disk: {desc} ...")
    patch_idx = 0

    for batch in tqdm(loader, desc=desc):
        if batch is None: continue
        patches = batch[0]
        for p in patches:
            img = p['image'].numpy()
            lbl = p['label'].numpy()

            # Concatenate and save
            data = np.concatenate([img, lbl], axis=0).astype(np.float32)

            save_name = os.path.join(save_subdir, f"p_{patch_idx:06d}.npy")
            np.save(save_name, data)
            npy_files.append(save_name)
            patch_idx += 1

    print(f"    ✅ Completed: {len(npy_files)} files.")
    return npy_files

# ================= 4. Dataset =================

class DiskDataset(Dataset):
    def __init__(self, file_list, is_val=False):
        self.file_list = file_list
        self.is_val = is_val

        # Data augmentation
        self.train_transform = transforms.Compose([
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=PATCH_SIZE_TRAIN, random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        ])

        # Validation crop
        self.val_transform = transforms.Compose([
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=PATCH_SIZE_TRAIN)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load
        fpath = self.file_list[idx]
        try:
            data = np.load(fpath) # (2, D, H, W)
            # To tensor
            data_t = torch.from_numpy(data).float()

            # Split
            img = data_t[0:1] # (1, D, H, W)
            lbl = data_t[1:2] # (1, D, H, W)
            item = {"image": img, "label": lbl}

            # Augmentation / Crop
            if not self.is_val:
                return self.train_transform(item)
            else:
                return self.val_transform(item)
        except Exception as e:
            # Exception handling
            print(f"Error loading {fpath}: {e}")
            # Random fallback
            return self.__getitem__(np.random.randint(len(self.file_list)))

# ================= 5. Main =================
def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad

def visualize(real, fake, epoch, save_dir):
    cz = real.shape[2] // 2
    r = (real[0, 0, cz].detach().cpu().numpy() + 1) / 2
    f = (fake[0, 0, cz].detach().cpu().numpy() + 1) / 2
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(r, cmap='gray', vmin=0, vmax=1); plt.title("Real")
    plt.subplot(1, 2, 2); plt.imshow(f, cmap='gray', vmin=0, vmax=1); plt.title("Recon")
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()

def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if local_rank == 0:
        print(f"\n🚀 [Start] VAE Pro Max | DDP Mode | Disk Cache Mode | Norm [-1, 1]")
        os.makedirs(os.path.join(SAVE_DIR, "vis"), exist_ok=True)

    # --- Prepare Files ---
    images = sorted(glob.glob(os.path.join(LUNA25_ROOT, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(LUNA25_ROOT, "labelsTr", "*.nii.gz")))
    min_len = min(len(images), len(labels))
    all_files = [{"image": i, "label": l} for i, l in zip(images[:min_len], labels[:min_len])]

    import random
    rng = random.Random(42)
    rng.shuffle(all_files)

    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # --- Mine Patches ---
    flag_path = os.path.join(CACHE_DIR, "mining_done.flag")
    if local_rank == 0:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(flag_path):
            os.remove(flag_path)
        _ = mine_patches_to_disk(train_files, mode="train")
        _ = mine_patches_to_disk(val_files, mode="val")
        with open(flag_path, "w") as f:
            f.write("done")
    else:
        import time
        while not os.path.exists(flag_path):
            time.sleep(5)

    dist.barrier()

    if local_rank == 0:
        if os.path.exists(flag_path):
            os.remove(flag_path)

    train_npy_list = sorted(glob.glob(os.path.join(CACHE_DIR, "train", "*.npy")))
    val_npy_list = sorted(glob.glob(os.path.join(CACHE_DIR, "val", "*.npy")))

    # --- DataLoaders ---
    train_ds = DiskDataset(train_npy_list, is_val=False)
    val_ds = DiskDataset(val_npy_list, is_val=True)

    train_loader = MonaiLoader(
        train_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=DistributedSampler(train_ds),
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = MonaiLoader(
        val_ds,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=DistributedSampler(val_ds, shuffle=False),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # --- Models ---
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, True),
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=3, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1
    ).to(device)

    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True).to(device)

    autoencoder = DDP(autoencoder, device_ids=[local_rank], find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[local_rank])

    optimizer_g = AdamW(autoencoder.parameters(), lr=LR, weight_decay=1e-2)
    optimizer_d = AdamW(discriminator.parameters(), lr=LR, weight_decay=1e-2)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=MAX_EPOCHS)
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    l1_loss = L1Loss()
    val_ssim = SSIMMetric(spatial_dims=3, data_range=2.0)

    best_ssim = 0.0
    start_epoch = 1
    latest_ckpt_path = os.path.join(SAVE_DIR, "checkpoint_latest.pth")
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        (autoencoder.module if hasattr(autoencoder, 'module') else autoencoder).load_state_dict(checkpoint['model_state_dict'])
        (discriminator.module if hasattr(discriminator, 'module') else discriminator).load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
        scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_ssim = checkpoint.get('best_ssim', 0.0)
        if local_rank == 0:
            print(f"✅ Resumed VAE from Epoch {start_epoch}, Current Best SSIM: {best_ssim:.4f}")

    if local_rank == 0:
        print(">>> Training Loop Started...")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loader.sampler.set_epoch(epoch)
        autoencoder.train(); discriminator.train()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{MAX_EPOCHS}", disable=(local_rank != 0))
        for batch in pbar:
            images = batch["image"].to(device)

            # --- G ---
            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast("cuda"):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                rec_loss = l1_loss(reconstruction, images)
                p_loss = perceptual_loss(reconstruction, images)
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-6) - 1) / images.shape[0]

                if epoch >= ADV_START_EPOCH:
                    # Call discriminator
                    logits_fake = (discriminator.module if hasattr(discriminator, "module") else discriminator)(reconstruction)[-1]
                    gen_loss = 0.5 * torch.mean((logits_fake - 1) ** 2)
                else:
                    gen_loss = torch.tensor(0.0, device=device)

                loss_g = (W_L1 * rec_loss) + (W_PERC * p_loss) + (W_KL * kl_loss) + (W_ADV * gen_loss)

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g); scaler_g.update()

            # --- D ---
            if epoch >= ADV_START_EPOCH:
                set_requires_grad(discriminator, True)
                optimizer_d.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    # Concat inputs to avoid DDP conflicts
                    combined_inputs = torch.cat([reconstruction.detach(), images], dim=0)
                    logits_combined = discriminator(combined_inputs)[-1]
                    logits_fake, logits_real = torch.chunk(logits_combined, 2, dim=0)

                    loss_d_fake = 0.5 * torch.mean(logits_fake ** 2)
                    loss_d_real = 0.5 * torch.mean((logits_real - 1) ** 2)
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d); scaler_d.update()

            if local_rank == 0:
                pbar.set_postfix({"L_G": f"{loss_g.item():.4f}", "SSIM": f"{best_ssim:.4f}"})

        scheduler_g.step()

        if epoch % VAL_INTERVAL == 0:
            autoencoder.eval()
            val_ssim.reset()
            with torch.no_grad():
                for j, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    with autocast("cuda"):
                        recons, _, _ = autoencoder(images)
                    val_ssim(y_pred=recons, y=images)
                    if j == 0 and local_rank == 0:
                        visualize(images, recons, epoch, os.path.join(SAVE_DIR, "vis"))

            # SSIM aggregation
            curr_ssim = val_ssim.aggregate().item()
            ssim_tensor = torch.tensor(curr_ssim).to(device)
            dist.all_reduce(ssim_tensor, op=dist.ReduceOp.SUM)
            avg_ssim = ssim_tensor.item() / dist.get_world_size()

            if local_rank == 0:
                print(f"  [Val] Ep {epoch} SSIM: {avg_ssim:.4f}")

                model_to_save = autoencoder.module if hasattr(autoencoder, 'module') else autoencoder
                disc_to_save = discriminator.module if hasattr(discriminator, 'module') else discriminator

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    torch.save(model_to_save.state_dict(), os.path.join(SAVE_DIR, "best_vae.pth"))
                    print(f"  ⭐ New Best Saved! SSIM: {best_ssim:.4f}")

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'discriminator_state_dict': disc_to_save.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'scheduler_g_state_dict': scheduler_g.state_dict(),
                    'scaler_g_state_dict': scaler_g.state_dict(),
                    'scaler_d_state_dict': scaler_d.state_dict(),
                    'best_ssim': best_ssim,
                }, latest_ckpt_path)

    cleanup_ddp()

if __name__ == "__main__":
    main()

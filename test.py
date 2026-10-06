import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from scipy import ndimage
import argparse
import json
import random
import sys
import types

def perturb_tensor(tensor, mean=0.0, std=1.0, bili=0.1, seed=None):
    tensor = torch.as_tensor(tensor)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        perturbation = torch.normal(mean, std, size=tensor.shape, generator=gen)
    else:
        perturbation = torch.normal(mean, std, size=tensor.shape)
    perturbation -= perturbation.mean(dim=-1, keepdim=True)
    max_perturbation = tensor.abs() * bili
    max_val = perturbation.abs().max(dim=-1, keepdim=True)[0]
    perturbation = perturbation / (max_val + 1e-8) * max_perturbation
    perturbed = tensor + perturbation
    perturbed = torch.clamp(perturbed, min=0.0)
    perturbed = perturbed / (perturbed.sum(dim=-1, keepdim=True) + 1e-8)
    return perturbed

# ================= 1. Seeds and Config =================
def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)


BG_FILL = -1.0

# ================= 2. Models and Metrics =================
from models.resnet import resnet50
from generative.networks.nets import ControlNet, AutoencoderKL, VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai.metrics import SSIMMetric

import warnings
warnings.filterwarnings("ignore")

# ================= 3. Config =================
from config import (
    LUNA16_CONTROLNET_DATA,
    RESNET_WEIGHTS,
    VAE_PATH,
    VQVAE_PATH,
    LDM_BEST_MODEL,
    VQ_LDM_BEST_MODEL,
    VOXEL_BEST_MODEL,
    ACVD_BEST_MODEL,
    VOXEL_HIST_BEST_MODEL,
    ACVD_HIST_BEST_MODEL,
    LDM_RAW_BEST_MODEL,
    VQ_LDM_RAW_BEST_MODEL
)

VAL_DATA_DIR = LUNA16_CONTROLNET_DATA
RESNET_WEIGHTS = RESNET_WEIGHTS
VAE_PATH = VAE_PATH
VQVAE_PATH = VQVAE_PATH

LUNG_MEAN_PIXEL = -0.2
CROP_SIZE = 48

MODELS_CONFIG = {
    "LDM": {
        "path": LDM_BEST_MODEL,
        "type": "latent", "cond_ch": 7, "clip": False
    },
    "VQ_LDM": {
        "path": VQ_LDM_BEST_MODEL,
        "type": "vq_latent", "cond_ch": 7, "clip": False
    },
    "LDM_Raw": {
        "path": LDM_RAW_BEST_MODEL,
        "type": "latent", "cond_ch": 3, "clip": False
    },
    "VQ_LDM_Raw": {
        "path": VQ_LDM_RAW_BEST_MODEL,
        "type": "vq_latent", "cond_ch": 3, "clip": False
    },
    "VOXEL": {
        "path": VOXEL_BEST_MODEL,
        "type": "pixel",  "cond_ch": 3, "clip": True
    },
    "ACVD": {
        "path": ACVD_BEST_MODEL,
        "type": "pixel",  "cond_ch": 7, "clip": True
    },
    "Voxel_Hist": {
        "path": VOXEL_HIST_BEST_MODEL,
        "type": "pixel_hist", "cond_ch": 3, "clip": True
    },
    "ACVD_Hist": {
        "path": ACVD_HIST_BEST_MODEL,
        "type": "pixel_hist", "cond_ch": 7, "clip": True
    }
}


# ================= 4. Utility Functions =================
def adagn_forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    h = x
    h = self.norm1(h)
    h = self.nonlinearity(h)

    if self.upsample is not None:
        if h.shape[0] >= 64:
            x = x.contiguous()
            h = h.contiguous()
        x = self.upsample(x)
        h = self.upsample(h)
    elif self.downsample is not None:
        x = self.downsample(x)
        h = self.downsample(h)

    h = self.conv1(h)

    temb = self.time_emb_proj(self.nonlinearity(emb))
    if self.spatial_dims == 2:
        temb = temb[:, :, None, None]
    else:
        temb = temb[:, :, None, None, None]

    scale, shift = temb.chunk(2, dim=1)

    h = self.norm2(h)
    h = h * (scale + 1) + shift
    h = self.nonlinearity(h)
    h = self.conv2(h)

    return self.skip_connection(x) + h

def convert_to_adagn(model, temb_channels=144):
    from generative.networks.nets.diffusion_model_unet import ResnetBlock
    for name, module in model.named_modules():
        if isinstance(module, ResnetBlock):
            device = next(module.parameters()).device
            module.time_emb_proj = nn.Linear(temb_channels, module.out_channels * 2).to(device)
            module.forward = types.MethodType(adagn_forward, module)

class HistTimeEmbedder(nn.Module):
    def __init__(self, original_time_embed):
        super().__init__()
        self.original_time_embed = original_time_embed
        self.current_hist = None

    def forward(self, t_emb):
        t_emb_out = self.original_time_embed(t_emb)
        if self.current_hist is not None:
            t_emb_out = torch.cat([t_emb_out, self.current_hist], dim=-1)
        return t_emb_out

class MedicalNetExtractor(nn.Module):
    def __init__(self, weights_path, device):
        super().__init__()
        self.model = resnet50(sample_input_D=CROP_SIZE, sample_input_H=CROP_SIZE, sample_input_W=CROP_SIZE, num_seg_classes=1)
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
        self.model.to(device).eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
            x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
            return torch.flatten(F.adaptive_avg_pool3d(x, (1, 1, 1)), 1)

class PixelControlSystem(nn.Module):
    def __init__(self, u, c): super().__init__(); self.unet=u; self.controlnet=c
    def forward(self, x, t, cond):
        d, m = self.controlnet(x=x, timesteps=t, controlnet_cond=cond)
        return self.unet(x=x, timesteps=t, down_block_additional_residuals=d, mid_block_additional_residual=m)

class PixelControlSystemWithHist(nn.Module):
    def __init__(self, u, c): super().__init__(); self.unet=u; self.controlnet=c
    def forward(self, x, t, cond, hist):
        self.unet.time_embed.current_hist = hist
        self.controlnet.time_embed.current_hist = hist
        d, m = self.controlnet(x=x, timesteps=t, controlnet_cond=cond)
        out = self.unet(x=x, timesteps=t, down_block_additional_residuals=d, mid_block_additional_residual=m)
        self.unet.time_embed.current_hist = None
        self.controlnet.time_embed.current_hist = None
        return out

class LDMControlSystem(nn.Module):
    def __init__(self, u, c): super().__init__(); self.unet=u; self.controlnet=c
    def forward(self, x, t, cond):
        d, m = self.controlnet(x=x, timesteps=t, controlnet_cond=cond)
        return self.unet(x=x, timesteps=t, down_block_additional_residuals=d, mid_block_additional_residual=m)

class ValDataset(Dataset):
    def __init__(self, data_dir, model_type="ACVD"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.model_type = model_type

        if self.model_type in ["VOXEL_HIST", "ACVD_HIST"]:
            json_path = "acvd_hist_clusters.json"
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.cluster_centers = np.array(json.load(f)[0]["centers"])
            else:
                print("Warning: acvd_hist_clusters.json not found. Run train_voxel_hist.py first.")
                self.cluster_centers = None

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        img = torch.from_numpy(data['gt_image']).float()
        masks = torch.from_numpy(data['conditions']).float()
        d = (64 - CROP_SIZE) // 2
        img = img[:, d:d+CROP_SIZE, d:d+CROP_SIZE, d:d+CROP_SIZE]
        masks = masks[:, d:d+48, d:d+48, d:d+48]
        nod_gt = masks[0:1]
        m_dil = torch.from_numpy(ndimage.binary_dilation(nod_gt[0].numpy(), iterations=3).astype(np.float32)).unsqueeze(0)

        if self.model_type in ["VOXEL_HIST", "ACVD_HIST"]:
            random.seed(idx)  # Fixed seed per sample
            center = random.choice(self.cluster_centers)
            hist = torch.from_numpy(center).float()
            hist = perturb_tensor(hist, bili=0.1, seed=idx)
        else:
            hist = torch.zeros(1)

        return {"img": img, "masks": masks, "m_dil": m_dil, "hist": hist, "idx": idx}

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ================= 5. Inference and Evaluation =================
def main():
    parser = argparse.ArgumentParser(
        description="DDP Evaluation Script for ACVD Models.\n"
                    "Must be launched via torchrun (supports both single-GPU and multi-GPU).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["LDM", "VQ_LDM", "LDM_Raw", "VQ_LDM_Raw", "VOXEL", "ACVD", "Voxel_Hist", "ACVD_Hist"],
                        help="Model architecture to evaluate.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU rank during evaluation.")
    args = parser.parse_args()

    # DDP launch check
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "This script must be launched via torchrun.\n"
            "Example running on 1 GPU:\n"
            "  torchrun --nproc_per_node=1 test.py --model LDM\n"
            "Example running on 8 GPUs:\n"
            "  torchrun --nproc_per_node=8 test.py --model LDM"
        )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_ddp = True

    cfg = MODELS_CONFIG[args.model]

    if cfg["type"] == "pixel":
        unet = DiffusionModelUNet(spatial_dims=3, in_channels=1, out_channels=1, num_channels=(32, 64, 128), attention_levels=(False, False, True), num_res_blocks=2, num_head_channels=32).to(device)
        ctrl = ControlNet(spatial_dims=3, in_channels=1, conditioning_embedding_in_channels=cfg["cond_ch"], conditioning_embedding_num_channels=(16,), num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)
        model = PixelControlSystem(unet, ctrl).to(device)
        vae = None
    elif cfg["type"] == "pixel_hist":
        unet = DiffusionModelUNet(spatial_dims=3, in_channels=1, out_channels=1, num_channels=(32, 64, 128), attention_levels=(False, False, True), num_res_blocks=2, num_head_channels=32).to(device)
        ctrl = ControlNet(spatial_dims=3, in_channels=1, conditioning_embedding_in_channels=cfg["cond_ch"], conditioning_embedding_num_channels=(16,), num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)

        # Extend time embedder to 144 dim for histogram
        unet.time_embed = HistTimeEmbedder(unet.time_embed)
        ctrl.time_embed = HistTimeEmbedder(ctrl.time_embed)

        # Adjust time_emb_proj mapping in ResnetBlock
        convert_to_adagn(unet, temb_channels=144)
        convert_to_adagn(ctrl, temb_channels=144)

        model = PixelControlSystemWithHist(unet, ctrl).to(device)
        vae = None
    elif cfg["type"] == "latent":
        unet = DiffusionModelUNet(spatial_dims=3, in_channels=8, out_channels=8, num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)
        ctrl = ControlNet(spatial_dims=3, in_channels=8, conditioning_embedding_in_channels=cfg["cond_ch"], conditioning_embedding_num_channels=(16, 32, 64), num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)
        model = LDMControlSystem(unet, ctrl).to(device)
        vae = AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1, num_channels=(64, 128, 256), latent_channels=8, num_res_blocks=2, norm_num_groups=32, attention_levels=(False, False, True)).to(device)
        vae.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only=False), strict=False)
        vae.eval()
    elif cfg["type"] == "vq_latent":
        unet = DiffusionModelUNet(spatial_dims=3, in_channels=8, out_channels=8, num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)
        ctrl = ControlNet(spatial_dims=3, in_channels=8, conditioning_embedding_in_channels=cfg["cond_ch"], conditioning_embedding_num_channels=(16, 32, 64), num_channels=(32, 64, 128), num_res_blocks=2, attention_levels=(False, False, True)).to(device)
        model = LDMControlSystem(unet, ctrl).to(device)
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

    ckpt = torch.load(cfg["path"], map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    loaded_mean = 0.0
    loaded_scale = 1.0

    if cfg["type"] in ["latent", "vq_latent"]:
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_unet."):
                fixed_state_dict[k.replace("base_unet.", "unet.")] = v
            else:
                fixed_state_dict[k] = v
        state_dict = fixed_state_dict

        # Load latent space statistics
        if "latent_stats" not in ckpt:
            raise KeyError(
                f"Error: Checkpoint '{cfg['path']}' is missing 'latent_stats'. "
                "Use a compatible latent diffusion checkpoint containing latent_stats."
            )

        loaded_mean = ckpt["latent_stats"]["mean"]
        loaded_scale = ckpt["latent_stats"]["scale"]
        if rank == 0:
            print(f"Loaded {args.model} Latent Stats from Checkpoint: Mean={loaded_mean:.6f}, Scale Factor={loaded_scale:.6f}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if rank == 0:
        print(f"[{args.model.upper()}] Evaluation on device {device} (DDP Rank {rank})")

    # 1. Feature extractor
    extractor = MedicalNetExtractor(RESNET_WEIGHTS, device)

    # 2. DataLoader
    val_dataset = ValDataset(VAL_DATA_DIR, model_type=args.model.upper())
    sampler = DistributedSampler(val_dataset, shuffle=False)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)

    # VAE scale factor
    vae_latent_mean = 0.0
    vae_scale_factor = 1.0
    vq_latent_mean = 0.0
    vq_scale_factor = 1.0

    if cfg["type"] == "latent":
        vae_latent_mean = loaded_mean
        vae_scale_factor = loaded_scale
    elif cfg["type"] == "vq_latent":
        vq_latent_mean = loaded_mean
        vq_scale_factor = loaded_scale

    scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=cfg["clip"])
    scheduler.set_timesteps(50)

    f_real_global, f_fused_global, f_real_roi, f_fused_roi = [], [], [], []

    # Initialize metrics
    ssim_calc = SSIMMetric(spatial_dims=3, data_range=2.0)
    global_ssims = []
    global_psnrs = []
    global_maes = []

    local_psnrs = []
    local_maes = []


    import time
    batch_indices = []
    rank_pure_infer_time = 0.0
    rank_sample_count = 0
    for batch in tqdm(loader, desc=f"Evaluating {args.model}", disable=(rank != 0)):
        batch_indices.append(batch['idx'].cpu().numpy())
        img_gt = batch['img'].to(device)
        m_dil = batch['m_dil'].to(device)
        masked_img = img_gt * (1.0 - m_dil) + LUNG_MEAN_PIXEL * m_dil

        nod_gt = batch['masks'][:, 0:1].to(device)
        if cfg["cond_ch"] == 3: cond = torch.cat([nod_gt, masked_img, m_dil], dim=1)
        else: cond = torch.cat([nod_gt, batch['masks'][:, 1:5].to(device), masked_img, m_dil], dim=1)

        # Timing start
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_start_time = time.time()

        with torch.no_grad():
            if cfg["type"] == "pixel":
                x_t = torch.empty_like(img_gt)
                for b in range(img_gt.shape[0]):
                    sample_idx = int(batch['idx'][b].item())
                    gen = torch.Generator(device=device).manual_seed(42 + sample_idx)
                    x_t[b].normal_(generator=gen)
                for i, t in enumerate(scheduler.timesteps):
                    t_b = torch.full((img_gt.shape[0],), t, device=device).long()
                    x_t_gen, _ = scheduler.step(model(x_t, t_b, cond), t, x_t)
                    if i + 1 < len(scheduler.timesteps):
                        t_p = scheduler.timesteps[i + 1]
                        noise = torch.empty_like(img_gt)
                        for b in range(img_gt.shape[0]):
                            sample_idx = int(batch['idx'][b].item())
                            gen = torch.Generator(device=device).manual_seed(42 + sample_idx + (i + 1) * 100000)
                            noise[b].normal_(generator=gen)
                        x_ref = scheduler.add_noise(img_gt, noise, torch.tensor([t_p], device=device).long())
                    else: x_ref = img_gt
                    x_t = x_t_gen * m_dil + x_ref * (1.0 - m_dil)
                final_fused = x_t_gen * m_dil + img_gt * (1.0 - m_dil)
            elif cfg["type"] == "pixel_hist":
                x_t = torch.empty_like(img_gt)
                for b in range(img_gt.shape[0]):
                    sample_idx = int(batch['idx'][b].item())
                    gen = torch.Generator(device=device).manual_seed(42 + sample_idx)
                    x_t[b].normal_(generator=gen)
                hist = batch['hist'].to(device)
                for i, t in enumerate(scheduler.timesteps):
                    t_b = torch.full((img_gt.shape[0],), t, device=device).long()
                    x_t_gen, _ = scheduler.step(model(x_t, t_b, cond, hist), t, x_t)
                    if i + 1 < len(scheduler.timesteps):
                        t_p = scheduler.timesteps[i + 1]
                        noise = torch.empty_like(img_gt)
                        for b in range(img_gt.shape[0]):
                            sample_idx = int(batch['idx'][b].item())
                            gen = torch.Generator(device=device).manual_seed(42 + sample_idx + (i + 1) * 100000)
                            noise[b].normal_(generator=gen)
                        x_ref = scheduler.add_noise(img_gt, noise, torch.tensor([t_p], device=device).long())
                    else: x_ref = img_gt
                    x_t = x_t_gen * m_dil + x_ref * (1.0 - m_dil)
                final_fused = x_t_gen * m_dil + img_gt * (1.0 - m_dil)
            elif cfg["type"] == "latent":
                z_gt = (vae.encode(img_gt)[0] - vae_latent_mean) * vae_scale_factor
                z_t = torch.empty_like(z_gt)
                for b in range(img_gt.shape[0]):
                    sample_idx = int(batch['idx'][b].item())
                    gen = torch.Generator(device=device).manual_seed(42 + sample_idx)
                    z_t[b].normal_(generator=gen)
                m_dil_l = F.interpolate(m_dil, size=z_gt.shape[2:], mode='nearest')
                for i, t in enumerate(scheduler.timesteps):
                    t_b = torch.full((img_gt.shape[0],), t, device=device).long()
                    z_t_gen, _ = scheduler.step(model(z_t, t_b, cond), t, z_t)
                    if i + 1 < len(scheduler.timesteps):
                        t_p = scheduler.timesteps[i + 1]
                        noise = torch.empty_like(z_gt)
                        for b in range(img_gt.shape[0]):
                            sample_idx = int(batch['idx'][b].item())
                            gen = torch.Generator(device=device).manual_seed(42 + sample_idx + (i + 1) * 100000)
                            noise[b].normal_(generator=gen)
                        z_ref = scheduler.add_noise(z_gt, noise, torch.tensor([t_p], device=device).long())
                        z_t = m_dil_l * z_t_gen + (1.0 - m_dil_l) * z_ref
                    else: z_t = m_dil_l * z_t_gen + (1.0 - m_dil_l) * z_gt
                raw_decoded_img = vae.decode((z_t / vae_scale_factor) + vae_latent_mean)
                clamped_decoded_img = torch.clamp(raw_decoded_img, min=-1.0, max=1.0)
                final_fused = clamped_decoded_img * m_dil + img_gt * (1.0 - m_dil)
            elif cfg["type"] == "vq_latent":
                z_gt_raw = vae.encode(img_gt)
                z_gt_quant, _ = vae.quantize(z_gt_raw)
                z_gt = (z_gt_quant - vq_latent_mean) * vq_scale_factor
                z_t = torch.empty_like(z_gt)
                for b in range(img_gt.shape[0]):
                    sample_idx = int(batch['idx'][b].item())
                    gen = torch.Generator(device=device).manual_seed(42 + sample_idx)
                    z_t[b].normal_(generator=gen)
                m_dil_l = F.interpolate(m_dil, size=z_gt.shape[2:], mode='nearest')
                for i, t in enumerate(scheduler.timesteps):
                    t_b = torch.full((img_gt.shape[0],), t, device=device).long()
                    z_t_gen, _ = scheduler.step(model(z_t, t_b, cond), t, z_t)
                    if i + 1 < len(scheduler.timesteps):
                        t_p = scheduler.timesteps[i + 1]
                        noise = torch.empty_like(z_gt)
                        for b in range(img_gt.shape[0]):
                            sample_idx = int(batch['idx'][b].item())
                            gen = torch.Generator(device=device).manual_seed(42 + sample_idx + (i + 1) * 100000)
                            noise[b].normal_(generator=gen)
                        z_ref = scheduler.add_noise(z_gt, noise, torch.tensor([t_p], device=device).long())
                        z_t = m_dil_l * z_t_gen + (1.0 - m_dil_l) * z_ref
                    else: z_t = m_dil_l * z_t_gen + (1.0 - m_dil_l) * z_gt
                raw_decoded_img = vae.decode((z_t / vq_scale_factor) + vq_latent_mean)
                clamped_decoded_img = torch.clamp(raw_decoded_img, min=-1.0, max=1.0)
                final_fused = clamped_decoded_img * m_dil + img_gt * (1.0 - m_dil)

        # Timing end
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_infer_time = time.time() - batch_start_time
        rank_pure_infer_time += batch_infer_time
        rank_sample_count += img_gt.shape[0]


        # Accumulate SSIM / PSNR / MAE
        for b in range(img_gt.shape[0]):
            # Calculate SSIM per sample
            ssim_calc(y_pred=final_fused[b:b+1], y=img_gt[b:b+1])
            ssim_val = ssim_calc.aggregate().item()
            ssim_calc.reset()
            global_ssims.append(ssim_val)

            # Global MSE and PSNR
            mse = F.mse_loss(final_fused[b], img_gt[b], reduction='mean').item()
            if mse > 0:
                psnr = 20 * np.log10(2.0) - 10 * np.log10(mse) # data range = 2.0
                global_psnrs.append(psnr)
            else:
                global_psnrs.append(float('inf'))

            # Global MAE
            mae = F.l1_loss(final_fused[b], img_gt[b], reduction='mean').item()
            global_maes.append(mae)

            # 1. Local PSNR inside dilated mask
            err_sq = (final_fused[b] - img_gt[b]) ** 2
            sum_mask = torch.sum(m_dil[b])
            if sum_mask > 0:
                mse_local = torch.sum(err_sq * m_dil[b]) / sum_mask
                mse_local_val = mse_local.item()
                if mse_local_val > 0:
                    psnr_local = 20 * np.log10(2.0) - 10 * np.log10(mse_local_val)
                    local_psnrs.append(psnr_local)
                else:
                    local_psnrs.append(float('inf'))
            else:
                local_psnrs.append(float('inf'))

            # 2. Local MAE inside dilated mask
            err_abs = torch.abs(final_fused[b] - img_gt[b])
            if sum_mask > 0:
                mae_local = torch.sum(err_abs * m_dil[b]) / sum_mask
                local_maes.append(mae_local.item())
            else:
                local_maes.append(0.0)

        # Extract features for FID
        f_real_global.append(extractor(img_gt).cpu().numpy())
        f_fused_global.append(extractor(final_fused).cpu().numpy())

        img_gt_roi = img_gt * m_dil + BG_FILL * (1.0 - m_dil)
        img_fused_roi = final_fused * m_dil + BG_FILL * (1.0 - m_dil)

        f_real_roi.append(extractor(img_gt_roi).cpu().numpy())
        f_fused_roi.append(extractor(img_fused_roi).cpu().numpy())

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fid_result")
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_ddp")

    if rank == 0:
        os.makedirs(temp_dir, exist_ok=True)

    if is_ddp:
        dist.barrier()

    # Save intermediate rank results
    rank_data = {
        "f_real_global": np.concatenate(f_real_global, axis=0) if len(f_real_global) > 0 else np.array([]),
        "f_fused_global": np.concatenate(f_fused_global, axis=0) if len(f_fused_global) > 0 else np.array([]),
        "f_real_roi": np.concatenate(f_real_roi, axis=0) if len(f_real_roi) > 0 else np.array([]),
        "f_fused_roi": np.concatenate(f_fused_roi, axis=0) if len(f_fused_roi) > 0 else np.array([]),
        "global_ssims": global_ssims,
        "global_psnrs": global_psnrs,
        "local_psnrs": local_psnrs,
        "global_maes": global_maes,
        "local_maes": local_maes,
        "pure_infer_time": rank_pure_infer_time,
        "sample_count": rank_sample_count,
        "indices": np.concatenate(batch_indices, axis=0) if len(batch_indices) > 0 else np.array([])
    }
    torch.save(rank_data, os.path.join(temp_dir, f"rank_{rank}.pth"))

    if is_ddp:
        dist.barrier()

    if rank == 0:
        all_f_real_global = []
        all_f_fused_global = []
        all_f_real_roi = []
        all_f_fused_roi = []
        all_global_ssims = []
        all_global_psnrs = []
        all_local_psnrs = []
        all_global_maes = []
        all_local_maes = []
        all_indices = []
        all_pure_infer_times = []
        all_sample_counts = []

        for r in range(world_size):
            # Allow loading numpy arrays
            p = torch.load(os.path.join(temp_dir, f"rank_{r}.pth"), map_location='cpu', weights_only=False)
            if p["indices"].size > 0:
                all_indices.append(p["indices"])
                all_f_real_global.append(p["f_real_global"])
                all_f_fused_global.append(p["f_fused_global"])
                all_f_real_roi.append(p["f_real_roi"])
                all_f_fused_roi.append(p["f_fused_roi"])
                all_global_ssims.append(p["global_ssims"])
                all_global_psnrs.append(p["global_psnrs"])
                all_local_psnrs.append(p["local_psnrs"])
                all_global_maes.append(p["global_maes"])
                all_local_maes.append(p["local_maes"])
                all_pure_infer_times.append(p["pure_infer_time"])
                all_sample_counts.append(p["sample_count"])

        merged_indices = np.concatenate(all_indices, axis=0)
        merged_f_real_global = np.concatenate(all_f_real_global, axis=0)
        merged_f_fused_global = np.concatenate(all_f_fused_global, axis=0)
        merged_f_real_roi = np.concatenate(all_f_real_roi, axis=0)
        merged_f_fused_roi = np.concatenate(all_f_fused_roi, axis=0)

        # Gather metrics from all GPUs
        merged_global_ssims = sum(all_global_ssims, [])
        merged_global_psnrs = sum(all_global_psnrs, [])
        merged_local_psnrs = sum(all_local_psnrs, [])
        merged_global_maes = sum(all_global_maes, [])
        merged_local_maes = sum(all_local_maes, [])

        # Remove padding samples added by DistributedSampler
        _, unique_idx = np.unique(merged_indices, return_index=True)
        unique_idx = np.sort(unique_idx)

        f_real_global_cat = merged_f_real_global[unique_idx]
        f_fused_global_cat = merged_f_fused_global[unique_idx]
        f_real_roi_cat = merged_f_real_roi[unique_idx]
        f_fused_roi_cat = merged_f_fused_roi[unique_idx]

        final_global_ssims = [merged_global_ssims[i] for i in unique_idx]
        final_global_psnrs = [merged_global_psnrs[i] for i in unique_idx if merged_global_psnrs[i] != float('inf')]
        final_local_psnrs = [merged_local_psnrs[i] for i in unique_idx if merged_local_psnrs[i] != float('inf')]
        final_global_maes = [merged_global_maes[i] for i in unique_idx]
        final_local_maes = [merged_local_maes[i] for i in unique_idx]

        avg_ssim = np.mean(final_global_ssims) if len(final_global_ssims) > 0 else 0.0
        avg_psnr = np.mean(final_global_psnrs) if len(final_global_psnrs) > 0 else 0.0
        avg_psnr_local = np.mean(final_local_psnrs) if len(final_local_psnrs) > 0 else 0.0
        avg_mae = np.mean(final_global_maes) if len(final_global_maes) > 0 else 0.0
        avg_mae_local = np.mean(final_local_maes) if len(final_local_maes) > 0 else 0.0

        def stats(f): return np.mean(f, axis=0), np.cov(f, rowvar=False)
        m_rg, s_rg = stats(f_real_global_cat); m_fg, s_fg = stats(f_fused_global_cat)
        m_rr, s_rr = stats(f_real_roi_cat); m_fr, s_fr = stats(f_fused_roi_cat)

        total_pure_infer_time = sum(all_pure_infer_times)
        total_sample_count = sum(all_sample_counts)
        avg_time_per_sample = total_pure_infer_time / total_sample_count if total_sample_count > 0 else 0.0

        results = {
            "Model": args.model,
            "FID_Global": calculate_fid(m_rg, s_rg, m_fg, s_fg),
            "mFID_ROI": calculate_fid(m_rr, s_rr, m_fr, s_fr),
            "SSIM_Global": avg_ssim,
            "PSNR_Global": avg_psnr,
            "MAE_Global": avg_mae,
            "PSNR_Local": avg_psnr_local,
            "MAE_Local": avg_mae_local,
            "Pure_Inference_Time_Sec": total_pure_infer_time,
            "Avg_Inference_Time_Per_Sample_Sec": avg_time_per_sample
        }

        out_path = os.path.join(out_dir, f"metrics_{args.model}.json")
        with open(out_path, "w") as f: json.dump(results, f, indent=4)
        print(f"\n[Success] {args.model} evaluation completed: {results}")

        # Cleanup temp files
        for r in range(world_size):
            try: os.remove(os.path.join(temp_dir, f"rank_{r}.pth"))
            except Exception: pass
        try: os.rmdir(temp_dir)
        except Exception: pass

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

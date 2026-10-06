# Anatomy-Constrained Voxel Diffusion (ACVD)

Implementation of **Anatomy-Constrained Voxel Diffusion for Controllable Synthesis of Complex Lung Nodules**, accepted at IEEE BIBM 2026 (publication forthcoming). ACVD synthesizes nodules directly in 48³ voxel space with anatomical conditioning and masked blended diffusion.

## Current release: supported workflows

| Workflow | Requirements | Supported by the processed data alone? |
|---|---|---|
| ACVD, voxel ablation and Voxel-Hist training | Paired 64³ CT/condition crops | Yes, after environment/path setup; no VAE required |
| Rebuild crops with `prepare_crops.py` | Original CT, full-volume nodule masks, coordinates and four anatomical masks | No; original CT/nodule masks must be obtained separately |
| VAE / VQ-VAE pretraining | Original CT and full-volume nodule masks, or a separately prepared autoencoder cache | No; these volumes/cache are not included |
| LDM / VQ-LDM training | Paired crops and compatible VAE / VQ-VAE weights | No; autoencoder weights are not included |
| Evaluation | Trained method checkpoints and MedicalNet weights; latent methods also require autoencoder weights | No; these weights must be supplied |

The current release supports starting voxel-space training. Additional assets are needed to reproduce latent baselines and reported evaluation results.

## Processed data

The data package contains **7,341 paired 64³ crops** (1,186 LUNA16 and 6,155 LUNA25), annotation tables and **four anatomical mask volumes covering the full CT grid**, for 888 LUNA16 and 4,069 LUNA25 series: airway, vessel/pulmonary artery, lung parenchyma and bone.

Beyond localized nodule synthesis, the full-volume masks support research on CT morphology, anatomical structures and spatial relationships over larger regions or entire scans. They are automatically generated predictions. Original full CT images must be obtained from LUNA16/LUNA25 separately.

Data repository: [lakelk/ACVD on Hugging Face](https://huggingface.co/datasets/lakelk/ACVD). Its data card describes formats; `SOURCES.md` includes source references/BibTeX, and `LICENSE.md` records component terms.

## Environment and paths

Use Linux with CUDA-enabled PyTorch and a suitable GPU. All training scripts support multi-GPU Distributed Data Parallel (DDP) with NCCL. Launch them with `torchrun`; the examples below use four GPUs. For one GPU, set `--nproc_per_node=1`. Install PyTorch for your CUDA environment, then:

```bash
python -m pip install monai monai-generative scipy numpy tqdm matplotlib scikit-learn
```

The scripts import `generative` from `monai-generative`. The original experimental environment has not yet been pinned.

Extract both data ZIPs into a common directory:

```text
ACVD-data/
  Dataset_LUNA16/
    annotations.csv
    ControlNet_Data/*.npy
    airway_masks/*.nii.gz
    vessel_masks/*.nii.gz
    lung_masks/*.nii.gz
    bone_masks/*.nii.gz
  Dataset_LUNA25/             # same structure
```

Edit `config.py`:

- `DATA_ROOT`: absolute path to `ACVD-data`.
- `BASE_MODEL_DIR`: writable diffusion checkpoint/output directory.
- `TRAINED_VAE_DIR`, `TRAINED_VQVAE_DIR`, `VAE_CACHE_DIR`: autoencoder paths if using those workflows.
- `RESNET_WEIGHTS`: actual location of the MedicalNet evaluation checkpoint.

Each `ControlNet_Data` folder contains NPY files directly, without another nested folder.

## A. Voxel-space training with the released crops

Run from the repository directory. This workflow does not require rebuilding crops, downloading original CT or pretraining a VAE.

```bash
# ACVD: full anatomical conditioning
torchrun --nproc_per_node=4 train_acvd.py

# Voxel-space ablation: basic conditioning
torchrun --nproc_per_node=4 train_voxel.py

# Voxel-Hist baseline
torchrun --nproc_per_node=4 train_voxel_hist.py
```

Set `--nproc_per_node` to your GPU count for multi-GPU training. Adjust the script's batch size to GPU memory.

### Format and conditioning

Each NPY stores a dictionary with `uid`, `nodule_idx`, `gt_image` (float32, 1×64×64×64) and `conditions` (binary float32, 5×64×64×64). Mask order: **nodule, vessel, airway, lung, bone**. Spatial order: **Z, Y, X**. CT is clipped to [-1000,400] HU and normalized to [-1,1].

The loaders extract 48³ crops. ACVD builds seven conditioning channels: the five masks, background CT with the synthesis region filled at -0.2, and a nodule mask dilated for three iterations. The basic voxel ablation uses nodule, masked CT and the dilated region.

Voxel training uses LUNA25 crops: sorted filenames, a `random.Random(42)` shuffle and an 80/20 nodule-instance split. Nodules from one CT can occur in different subsets. Training uses randomly shifted 48³ crops; validation uses centered crops. `test.py` reads LUNA16 crops.

## B. Workflows requiring additional preparation

### Rebuild paired crops

`prepare_crops.py` requires original CT in `imagesTr`, full-volume nodule masks in `labelsTr`, `annotations.csv` and the four anatomical mask folders for each subset. Released anatomical masks can be reused without retraining segmentation models. Legacy combined `anatomy_masks` are supported as a lung/bone fallback.

Additional dependencies and command:

```bash
python -m pip install pandas SimpleITK
python prepare_crops.py
```

Coordinates use `seriesuid`, `coordX`, `coordY`, `coordZ` (with aliases handled by the script). Keep CSV row order: `nodule_idx` refers to zero-based rows.

The script extracts an approximately 96 mm region around each physical annotation coordinate, applies CT-derived array bounds to every channel, interpolates to 96³, then extracts the central 64³. CT is clipped/normalized as above; masks are thresholded at >0.5. Masks must already share the CT voxel grid. Interpolation produces approximately 1 mm crops; it is not exact affine-aware resampling to (1,1,1) for every scan. Saved dictionaries contain no physical affine.

### VAE / VQ-VAE pretraining

VAE and VQ-VAE pretraining mine patches from the original LUNA25 `imagesTr` and full-volume nodule `labelsTr`. The mining pipeline orients volumes to RAS, resamples CT/masks to **(1,1,1) mm** using linear/nearest-neighbor interpolation, and normalizes CT.

`RandCropByPosNegLabeld(pos=1, neg=1, num_samples=8)` chooses foreground (nodule) or eligible background sampling centers with equal probability. This is an expected **1:1 foreground/background center sampling ratio**, not a guarantee of four positive and four negative patches per scan. A background-centered patch can still contain nearby nodule voxels.

Training patches are cached at 80³ and randomly cropped to 48³ during training; validation patches are mined at 48³. The released 64³ diffusion crops are centered around annotated nodules and cannot reproduce the original full-volume background sampling distribution. This sampling protocol is why these scripts use original volumes rather than directly using the released 64³ crops.

After obtaining these volumes, configuring paths and installing the NIfTI reader dependencies:

```bash
torchrun --nproc_per_node=4 train_vae.py
torchrun --nproc_per_node=4 train_vqvae.py
```

### Latent diffusion training

Supply compatible autoencoder weights at `VAE_PATH` or `VQVAE_PATH`, then use the paired crops:

```bash
# Full anatomical conditioning
torchrun --nproc_per_node=4 train_ldm.py
torchrun --nproc_per_node=4 train_vq_ldm.py

# Basic conditioning baselines
torchrun --nproc_per_node=4 train_ldm_raw.py
torchrun --nproc_per_node=4 train_vq_ldm_raw.py
```

## Evaluation

Supply trained method checkpoints and the Med3D ResNet-50 weights (`resnet_50_23dataset.pth`) from [MedicalNet](https://github.com/Tencent/MedicalNet). Set their actual paths in `config.py`; weights are not bundled.

```bash
torchrun --nproc_per_node=4 test.py --model ACVD
```

`--model` is required. Other voxel options are `VOXEL` and `Voxel_Hist`. The script computes 3D FID, masked PSNR/MAE and SSIM on LUNA16 crops.

Latent evaluation additionally requires autoencoder weights and diffusion checkpoints containing `latent_stats`. Legacy checkpoints may require conversion; the `convert_legacy_weights.py` mentioned in the evaluation error message is not included.

Autoencoder reconstruction evaluation supports VAE and VQ-VAE weights (at least one checkpoint must be available). `test_vae.py` uses multi-GPU `DataParallel`, rather than DDP, and is launched with plain Python:

```bash
python test_vae.py
```

## Citation

If you use this code, please cite **Anatomy-Constrained Voxel Diffusion for Controllable Synthesis of Complex Lung Nodules**, accepted at IEEE BIBM 2026. The final author list and proceedings/DOI citation will be added when available.

Data-source references and BibTeX are provided separately in the [dataset SOURCES.md](https://huggingface.co/datasets/lakelk/ACVD/blob/main/SOURCES.md).

## License

Copyright (c) 2026 ACVD authors. The authors' original code is licensed under the [MIT License](https://opensource.org/license/mit). Third-party code, dependencies and weights retain their respective licenses. Data licensing is documented separately in the [dataset LICENSE.md](https://huggingface.co/datasets/lakelk/ACVD/blob/main/LICENSE.md).

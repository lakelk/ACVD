import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")

# ================= 1. Configuration =================
from config import BASE_DIR, LUNA25_ROOT, LUNA25_CONTROLNET_DATA, LUNA16_ROOT, LUNA16_CONTROLNET_DATA

# Physical and model parameters
CROP_MM = 96.0 # 96mm physical block
FINAL_SIZE = 64 # Central crop after interpolation to 96 voxels per axis
NUM_WORKERS = 30 # Number of worker processes
CLIP_MIN = -1000.0
CLIP_MAX = 400.0 # VAE training range limit

# ================= 2. Utility Functions =================
def find_file(uid, folder):
    for ext in [".nii.gz", "_0000.nii.gz", ".nii", "_0000.nii"]:
        p = os.path.join(folder, f"{uid}{ext}")
        if os.path.exists(p): return p
    return None

def process_single_case(args):
    uid, csv_idx, wx, wy, wz, root, save_dir = args

    paths = {
        "image": os.path.join(root, "imagesTr"),
        "nodule": os.path.join(root, "labelsTr"),
        "vessel": os.path.join(root, "vessel_masks"),
        "airway": os.path.join(root, "airway_masks"),
        "lung": os.path.join(root, "lung_masks"),
        "bone": os.path.join(root, "bone_masks"),
        "anatomy": os.path.join(root, "anatomy_masks")
    }

    # Align files and masks
    img_p = find_file(uid, paths["image"])
    if not img_p:
        return ("MISSING_IMAGE", uid)

    nod_p = find_file(uid, paths["nodule"])
    if not nod_p:
        return ("MISSING_NODULE", uid)

    ves_p = find_file(uid, paths["vessel"])
    if not ves_p:
        return ("MISSING_VESSEL", uid)

    air_p = find_file(uid, paths["airway"])
    if not air_p:
        return ("MISSING_AIRWAY", uid)

    # Prefer separate binary masks; fall back to the original combined labels.
    lung_p = find_file(uid, paths["lung"])
    bone_p = find_file(uid, paths["bone"])
    ana_p = None
    if not (lung_p and bone_p):
        ana_p = find_file(uid, paths["anatomy"])
        if not ana_p:
            return ("MISSING_LUNG" if not lung_p else "MISSING_BONE", uid)

    try:
        # Read ITK image
        itk_img = sitk.ReadImage(img_p)
        itk_nod = sitk.ReadImage(nod_p)

        # 3. Physical alignment
        target_pos = (wx, wy, wz)

        # Map to voxel coordinates
        center_idx = itk_img.TransformPhysicalPointToContinuousIndex(target_pos)
        cx, cy, cz = center_idx

        # Calculate radius in voxels
        spacing = itk_img.GetSpacing()
        rad_vox = [(CROP_MM / 2) / s for s in spacing]

        # Perform cropping
        def get_crop(itk_obj, fill_val):
            arr = sitk.GetArrayFromImage(itk_obj) # (Z, Y, X)
            z, y, x = arr.shape
            # Coordinate bounds
            sz, ez = int(cz - rad_vox[2]), int(cz + rad_vox[2])
            sy, ey = int(cy - rad_vox[1]), int(cy + rad_vox[1])
            sx, ex = int(cx - rad_vox[0]), int(cx + rad_vox[0])
            # Padding
            pad = [(max(0, -sz), max(0, ez-z)), (max(0, -sy), max(0, ey-y)), (max(0, -sx), max(0, ex-x))]
            crop = arr[max(0,sz):min(z,ez), max(0,sy):min(y,ey), max(0,sx):min(x,ex)]
            if any(sum(p)>0 for p in pad):
                crop = np.pad(crop, pad, mode='constant', constant_values=fill_val)
            return crop

        raw_img = get_crop(itk_img, -1000)
        raw_nod = get_crop(itk_nod, 0)

        if raw_nod.sum() < 5: return ("EMPTY_NODULE", uid)

        # Load anatomical masks
        if lung_p and bone_p:
            lung_mask = (get_crop(sitk.ReadImage(lung_p), 0) > 0).astype(np.float32)
            bone_mask = (get_crop(sitk.ReadImage(bone_p), 0) > 0).astype(np.float32)
        else:
            raw_ana = get_crop(sitk.ReadImage(ana_p), 0)
            lung_mask = (raw_ana == 1).astype(np.float32)
            bone_mask = (raw_ana == 2).astype(np.float32)

        raw_ves = get_crop(sitk.ReadImage(ves_p), 0)
        raw_air = get_crop(sitk.ReadImage(air_p), 0)

        # Resample to 1.0mm spacing
        # Concatenate 6 channels
        stack = np.stack([raw_img, raw_nod, raw_ves, raw_air, lung_mask, bone_mask], axis=0)
        t_stack = torch.from_numpy(stack).unsqueeze(0).float() # (1, 6, D, H, W)

        # Resample to 96^3
        res = F.interpolate(t_stack, size=(96, 96, 96), mode='trilinear')
        # Crop center to 64^3
        d = (96 - 64) // 2
        final_6ch = res[0, :, d:d+64, d:d+64, d:d+64]

        # Normalize and binarize
        img_final = torch.clamp((final_6ch[0:1] - CLIP_MIN) / (CLIP_MAX - CLIP_MIN), 0.0, 1.0) * 2.0 - 1.0
        masks_final = (final_6ch[1:] > 0.5).float() # [Nod, Ves, Air, Lung, Bone]

        return {
            "img": img_final,   # (1, 64, 64, 64)
            "masks": masks_final, # (5, 64, 64, 64)
            "meta": {"uid": uid, "idx": csv_idx},
            "save_dir": save_dir
        }
    except Exception as e:
        return ("CRASH", f"{uid}: {str(e)}")

# ================= 3. Main (Multiprocessing) =================
def main():
    datasets = [
        {
            "name": "LUNA25",
            "root": LUNA25_ROOT,
            "csv": os.path.join(LUNA25_ROOT, "annotations.csv"),
            "save_dir": LUNA25_CONTROLNET_DATA
        },
        {
            "name": "LUNA16",
            "root": LUNA16_ROOT,
            "csv": os.path.join(LUNA16_ROOT, "annotations.csv"),
            "save_dir": LUNA16_CONTROLNET_DATA
        }
    ]

    for ds in datasets:
        name = ds["name"]
        root = ds["root"]
        csv_path = ds["csv"]
        save_dir = ds["save_dir"]

        print(f"\n======== Processing Dataset: {name} ========")
        if not os.path.exists(csv_path):
            print(f" CSV file not found: {csv_path}. Skipping dataset {name}.")
            continue

        os.makedirs(save_dir, exist_ok=True)
        vis_dir = os.path.join(save_dir, "vis_debug")
        os.makedirs(vis_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        tasks = []
        for idx, row in df.iterrows():
            uid = row.get('seriesuid', row.get('SeriesInstanceUID'))
            cx = float(row.get('coordX', row.get('CoordX')))
            cy = float(row.get('coordY', row.get('CoordY')))
            cz = float(row.get('coordZ', row.get('CoordZ')))
            tasks.append((uid, idx, cx, cy, cz, root, save_dir))

        success = 0
        errors = {}
        vis_data = []

        with Pool(NUM_WORKERS) as p:
            for res in tqdm(p.imap_unordered(process_single_case, tasks), total=len(tasks), desc=name):
                if isinstance(res, tuple):
                    reason = res[0]
                    errors[reason] = errors.get(reason, 0) + 1
                    continue

                # Package data dictionary
                save_packet = {
                    "uid": res['meta']['uid'],
                    "nodule_idx": res['meta']['idx'],
                    "gt_image": res['img'].numpy(),     # (1, 64, 64, 64)
                    "conditions": res['masks'].numpy()  # (5, 64, 64, 64) -> [Nod, Ves, Air, Lung, Bone]
                }

                fname = f"case_{res['meta']['idx']}_{res['meta']['uid']}.npy"
                np.save(os.path.join(res['save_dir'], fname), save_packet)

                if success < 3: vis_data.append(save_packet)
                success += 1

        print(f"\n {name} : {success} |  : {errors}")

        # ================= 4. Verification =================
        if success > 0:
            print(f"\n Generating {name} verification image...")
            for i, item in enumerate(vis_data):
                gt = item['gt_image'][0]
                masks = item['conditions']

                # Find nodule slice
                cz = np.argmax(np.sum(masks[0], axis=(1,2)))

                fig, axs = plt.subplots(1, 6, figsize=(24, 4))
                axs[0].imshow(gt[cz], cmap='gray', vmin=-1, vmax=1); axs[0].set_title("Original CT")
                labels = ["Nodule", "Vessel", "Airway", "Lung", "Bone"]
                for k in range(5):
                    axs[k+1].imshow(masks[k, cz], cmap='gray', vmin=0, vmax=1)
                    axs[k+1].set_title(f"{labels[k]} Mask")

                plt.savefig(os.path.join(vis_dir, f"check_{i}.png"))
                plt.close()
            print(f" Report saved to: {vis_dir}")

if __name__ == "__main__":
    main()

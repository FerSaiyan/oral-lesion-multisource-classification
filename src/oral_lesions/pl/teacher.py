# src/oral_lesions/pl/teacher.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.datasets import UnlabeledDatasetCSV
from ..data.transforms import build_transforms
# Prefer the modular effnet if present; otherwise fallback to legacy src.models
try:
    from ..models.effnet import enetv2  # type: ignore
except Exception:
    from ...models import enetv2  # fallback to src/models.py

def _softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    T = 1.0 if (T is None or T <= 0) else float(T)
    return torch.softmax(logits / T, dim=1)

@torch.no_grad()
def run_teacher_inference_if_needed(
    teacher_ckpt_path: str,
    unlabeled_csv_path: str,
    unlabeled_images_folder: str,
    device: torch.device,
    out_infer_path: str,
    num_classes: int,
    backbone_name: str,
    load_geffnet_pretrained_in_model: bool,
    temperature: float = 1.0,
    batch_size: int = 4,
    num_workers: int = 0,
    img_size: int = 512
) -> str:
    """
    Runs a (EfficientNet) teacher on an unlabeled CSV and writes:
      filename, pred_idx, p_max
    Skips work if out_infer_path exists.
    """
    if os.path.exists(out_infer_path):
        print(f"[PL] Using cached teacher inference: {out_infer_path}")
        return out_infer_path

    # Transforms: use 'val' pipeline of effnet (normalization + resize)
    _, val_t = build_transforms("effnet", img_size)
    ds_u = UnlabeledDatasetCSV(unlabeled_csv_path, unlabeled_images_folder, transform=val_t)
    dl_u = DataLoader(ds_u, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Build teacher model (EffNet)
    teacher = enetv2(backbone=backbone_name, out_dim=num_classes, load_pretrained_geffnet=load_geffnet_pretrained_in_model).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device), strict=False)
    teacher.eval()

    fnames, preds, pmaxs = [], [], []
    amp_enabled = (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        for images, batch_fnames in dl_u:
            images = images.to(device, non_blocking=True)
            logits = teacher(images)
            probs = _softmax_with_temperature(logits, temperature)
            p_max, pred = probs.max(dim=1)
            fnames.extend(batch_fnames)
            preds.extend(pred.detach().cpu().tolist())
            pmaxs.extend(p_max.detach().cpu().tolist())

    df_out = pd.DataFrame({'filename': fnames, 'pred_idx': preds, 'p_max': pmaxs})
    os.makedirs(os.path.dirname(out_infer_path), exist_ok=True)
    df_out.to_csv(out_infer_path, index=False)
    print(f"[PL] Saved teacher inference to {out_infer_path} ({len(df_out)} rows)")
    return out_infer_path

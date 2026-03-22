"""
Unified model factory that wraps existing implementations in src.models.
Adds robust resolution of custom checkpoint paths via environment variables
before falling back to config-provided paths, and emits clear logs when
falling back to random weights.
"""
import os
import torch

# Use legacy implementations
from ...models import enetv2, ViTFineTuner
from ...utils import getenv_path


def _resolve_custom_path(primary_path: str | None, env_explicit_key: str | None, env_dir_key: str | None) -> tuple[str | None, str]:
    """Resolve a custom checkpoint path with the following precedence:
    1) Explicit env path (env_explicit_key), if set and exists
    2) If env_dir_key points to a directory, try joining it with the basename of primary_path
    3) primary_path as-is
    Returns (resolved_path_or_None, explanation_string)
    """
    # 1) Explicit env override (absolute path)
    if env_explicit_key:
        env_path = getenv_path(env_explicit_key)
        if env_path and os.path.exists(env_path):
            return env_path, f"env {env_explicit_key}"
    # 2) Directory from env + basename of primary_path
    if env_dir_key:
        env_dir = getenv_path(env_dir_key)
        if env_dir and primary_path:
            candidate = os.path.join(env_dir, os.path.basename(primary_path))
            if os.path.exists(candidate):
                return candidate, f"env {env_dir_key}+basename"
    # 3) Primary path
    if primary_path and os.path.exists(primary_path):
        return primary_path, "config path"
    return None, "not found"


def create_model(cfg: dict, num_classes: int, device: torch.device = torch.device('cpu')):
    mt = cfg.get("model_type")
    if mt == "effnet":
        backbone = cfg.get("enet_backbone", "efficientnet-b0")
        pretrained_source = cfg.get("pretrained_source", "imagenet")  # imagenet | custom_skin
        load_geffnet = (pretrained_source == "imagenet")
        m = enetv2(backbone=backbone, out_dim=num_classes, load_pretrained_geffnet=load_geffnet)
        # Dropout if provided
        if "dropout_rate" in cfg and hasattr(m, "dropout"):
            m.dropout.p = float(cfg["dropout_rate"])  # set externally too in study

        # Load full weights if provided/desired
        desired = cfg.get("custom_effnet_path") or cfg.get("custom_model_path")
        resolved, reason = _resolve_custom_path(
            primary_path=desired,
            env_explicit_key="CUSTOM_EFFNET_PATH",
            env_dir_key="PRETRAINED_MODELS_DIR",
        )
        if resolved:
            try:
                print(f"[EffNet] Loading custom weights from: {resolved} (via {reason})")
                sd = torch.load(resolved, map_location=device)
                m.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[EffNet] Failed to load custom weights from {resolved}: {e}. Continuing with current weights.")
        else:
            if pretrained_source == "custom_skin":
                # Custom requested but file not found → explicit notice
                bn = os.path.basename(desired) if desired else "<unspecified>"
                print(f"[EffNet] WARNING: pretrained_source=custom_skin but checkpoint not found (wanted '{bn}'). "
                      f"Falling back to random-initialized head/backbone.")
        return m

    if mt == "vit":
        vit_name = cfg.get("vit_model_name", "vit_base_patch16_224")
        pretrained_source = cfg.get("pretrained_source", "imagenet_timm")  # imagenet_timm | custom_skin | custom_backbone_timm
        use_timm_imagenet = (pretrained_source == "imagenet_timm")

        # Resolve custom backbone (for custom_backbone_timm)
        backbone_path_cfg = cfg.get("custom_vit_path") if pretrained_source == "custom_backbone_timm" else None
        backbone_resolved, backbone_reason = _resolve_custom_path(
            primary_path=backbone_path_cfg,
            env_explicit_key="CUSTOM_VIT_BACKBONE_PATH",
            env_dir_key="PRETRAINED_MODELS_DIR",
        )
        m = ViTFineTuner(
            model_name=vit_name,
            out_dim=num_classes,
            dropout_rate=float(cfg.get("dropout_rate", 0.5)),
            custom_pretrained_model_path=backbone_resolved,
            timm_imagenet_pretrained=use_timm_imagenet,
        )
        if backbone_path_cfg and not backbone_resolved and pretrained_source == "custom_backbone_timm":
            bn = os.path.basename(backbone_path_cfg)
            print(f"[ViT] WARNING: custom_backbone_timm requested but backbone checkpoint '{bn}' not found. Using TIMM init.")

        # Full model weights (custom_skin)
        if pretrained_source == "custom_skin":
            desired = cfg.get("custom_vit_path") or cfg.get("custom_model_path")
            resolved, reason = _resolve_custom_path(
                primary_path=desired,
                env_explicit_key="CUSTOM_VIT_PATH",
                env_dir_key="PRETRAINED_MODELS_DIR",
            )
            if resolved:
                try:
                    print(f"[ViT] Loading custom FULL weights from: {resolved} (via {reason})")
                    sd = torch.load(resolved, map_location=device)
                    if isinstance(sd, dict) and any(k.startswith('module.') for k in sd.keys()):
                        sd = {k.replace('module.', ''): v for k, v in sd.items()}
                    m.load_state_dict(sd, strict=False)
                except Exception as e:
                    print(f"[ViT] Failed to load custom FULL weights from {resolved}: {e}. Continuing with current weights.")
            else:
                bn = os.path.basename(desired) if desired else "<unspecified>"
                print(f"[ViT] WARNING: pretrained_source=custom_skin but checkpoint not found (wanted '{bn}'). "
                      f"Falling back to TIMM/random init.")
        return m

    raise ValueError(f"Unknown model_type: {mt}")

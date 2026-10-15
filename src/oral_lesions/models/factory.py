"""
Unified model factory that wraps existing implementations in src.models.
Adds robust resolution of custom checkpoint paths via environment variables
before falling back to config-provided paths, and emits clear logs when
falling back to random weights.

Supported pretrained_source values by model type:

  EffNet:
    - "imagenet"     (default) — TIMM ImageNet weights
    - "custom_skin"  — custom skin-pretrained checkpoint
    - "checkpoint" — generic classifier checkpoint finetuning

  ViT:
    - "imagenet_timm"    (default) — TIMM ImageNet weights
    - "custom_skin"      — custom skin-pretrained full checkpoint
    - "custom_backbone_timm" — custom backbone weights passed to TIMM
    - "checkpoint"     — generic classifier checkpoint finetuning

When pretrained_source="checkpoint":
  - The checkpoint path is taken from cfg["custom_model_path"] (or env override).
  - The checkpoint MUST exist or a FileNotFoundError is raised (fail-fast).
  - A number-of-classes compatibility check is performed.
  - Use this OPTUNA-DRIVEN path for finetuning an existing classifier .pth.
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


def _validate_checkpoint_compatibility(sd: dict, model: torch.nn.Module, model_type: str) -> list[str]:
    """
    Check that the checkpoint's final-layer weight shape matches the model's head.
    Returns a list of warning/info messages (empty list = fully compatible).
    """
    msgs = []
    # Determine classifier weight names for each architecture
    if model_type == "effnet":
        classifier_keys = ["myfc.weight", "myfc.bias"]
    elif model_type == "vit":
        classifier_keys = ["vit_model.head.weight", "vit_model.head.bias",
                           "head.weight", "head.bias"]
    else:
        return msgs

    for ck in classifier_keys:
        if ck not in sd:
            continue
        # Try to resolve attribute path safely on the model.
        parts = ck.split(".")
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "shape"):
            sd_shape = sd[ck].shape
            model_shape = obj.shape
            if sd_shape != model_shape:
                msgs.append(
                    f"  Head layer '{ck}': checkpoint shape {tuple(sd_shape)} "
                    f"vs model shape {tuple(model_shape)}. "
                    "This is expected if num_classes differs between source and target."
                )
    return msgs


def _load_checkpoint_weights(model: torch.nn.Module, sd_path: str, device: torch.device,
                             model_type: str, num_classes: int) -> None:
    """
    Load a checkpoint state_dict into *model* with validation and clear messaging.
    Raises FileNotFoundError if the checkpoint path does not exist.
    """
    if not os.path.isfile(sd_path):
        raise FileNotFoundError(
            f"[{model_type.upper()}] pretrained_source='checkpoint' but checkpoint file "
            f"not found at: {sd_path}\n"
            f"  Set a valid path via 'custom_model_path' in your model config YAML, "
            f"or via the CUSTOM_MODEL_PATH environment variable."
        )

    print(f"[{model_type.upper()}] Loading classifier checkpoint from: {sd_path}")
    sd = torch.load(sd_path, map_location='cpu')

    # Handle common wrapped checkpoint formats.
    if isinstance(sd, dict):
        for state_key in ("state_dict", "model_state_dict"):
            nested = sd.get(state_key)
            if isinstance(nested, dict):
                sd = nested
                break

    if not isinstance(sd, dict):
        raise RuntimeError(
            f"[{model_type.upper()}] Unsupported checkpoint format at {sd_path}: "
            f"expected state_dict-like dict, got {type(sd).__name__}."
        )

    # Warn about DataParallel wrapper
    if any(k.startswith('module.') for k in sd.keys()):
        print(f"  Detected 'module.' prefix in checkpoint keys — stripping for direct load.")
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # Compatibility validation
    warnings = _validate_checkpoint_compatibility(sd, model, model_type)
    if warnings:
        print(f"  [CHECKPOINT COMPATIBILITY]")
        for w in warnings:
            print(f"    {w}")

    # Drop shape-mismatched tensors so strict=False can load the rest.
    # This is expected for classifier head tensors when target split has fewer classes.
    model_sd = model.state_dict()
    dropped_keys: list[str] = []
    for k in list(sd.keys()):
        if k not in model_sd:
            continue
        try:
            ck_shape = tuple(sd[k].shape)
            md_shape = tuple(model_sd[k].shape)
        except Exception:
            continue
        if ck_shape != md_shape:
            dropped_keys.append(k)
            sd.pop(k, None)
    if dropped_keys:
        print(
            "  Dropping shape-mismatched checkpoint keys: "
            f"{dropped_keys[:5]}{'...' if len(dropped_keys) > 5 else ''}"
        )

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys in checkpoint (will use model init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys in checkpoint (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"  Checkpoint loaded successfully (strict=False).")
    except Exception as e:
        raise RuntimeError(
            f"[{model_type.upper()}] Failed to load checkpoint state_dict from {sd_path}: {e}"
        )


def create_model(cfg: dict, num_classes: int, device: torch.device = torch.device('cpu')):
    mt = cfg.get("model_type")
    if mt == "effnet":
        backbone = cfg.get("enet_backbone", "efficientnet-b0")
        pretrained_source = cfg.get("pretrained_source", "imagenet")  # imagenet | custom_skin | checkpoint
        load_geffnet = (pretrained_source == "imagenet")
        m = enetv2(backbone=backbone, out_dim=num_classes, load_pretrained_geffnet=load_geffnet)
        # Dropout if provided
        if "dropout_rate" in cfg and hasattr(m, "dropout"):
            m.dropout.p = float(cfg["dropout_rate"])  # set externally too in study

        # --- pretrained_source: "checkpoint" (explicit Optuna checkpoint-finetune path) ---
        if pretrained_source == "checkpoint":
            desired = cfg.get("custom_model_path")
            resolved, reason = _resolve_custom_path(
                primary_path=desired,
                env_explicit_key="CUSTOM_MODEL_PATH",
                env_dir_key="PRETRAINED_MODELS_DIR",
            )
            if not resolved:
                bn = os.path.basename(desired) if desired else "<unspecified>"
                raise FileNotFoundError(
                    f"[EffNet] pretrained_source='checkpoint' but checkpoint not found "
                    f"(wanted '{bn}'). Searched via {reason}.\n"
                    f"  Set 'custom_model_path' in your model config, or export CUSTOM_MODEL_PATH."
                )
            _load_checkpoint_weights(m, resolved, device, model_type="effnet", num_classes=num_classes)
            return m

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
                sd = torch.load(resolved, map_location='cpu')
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
        pretrained_source = cfg.get("pretrained_source", "imagenet_timm")  # imagenet_timm | custom_skin | custom_backbone_timm | checkpoint
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

        # --- pretrained_source: "checkpoint" (explicit Optuna checkpoint-finetune path) ---
        if pretrained_source == "checkpoint":
            desired = cfg.get("custom_model_path")
            resolved, reason = _resolve_custom_path(
                primary_path=desired,
                env_explicit_key="CUSTOM_MODEL_PATH",
                env_dir_key="PRETRAINED_MODELS_DIR",
            )
            if not resolved:
                bn = os.path.basename(desired) if desired else "<unspecified>"
                raise FileNotFoundError(
                    f"[ViT] pretrained_source='checkpoint' but checkpoint not found "
                    f"(wanted '{bn}'). Searched via {reason}.\n"
                    f"  Set 'custom_model_path' in your model config, or export CUSTOM_MODEL_PATH."
                )
            _load_checkpoint_weights(m, resolved, device, model_type="vit", num_classes=num_classes)
            return m

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
                    sd = torch.load(resolved, map_location='cpu')
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

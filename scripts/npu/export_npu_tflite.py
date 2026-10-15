#!/usr/bin/env python3
"""Export an oral-lesion classifier checkpoint to LiteRT for the Qualcomm NPU.

Writes, for one checkpoint:
  <name>_fp32.tflite   float graph (HTP runs it in fp16 after AOT)
  <name>_w8a8.tflite   static int8 weights + int8 activations (ai-edge-quantizer)
  <name>_w8a16.tflite  static int8 weights + int16 activations
  <name>_w8a16dwfp.tflite  W8A16 with float depthwise conv / pad (opt-in, EfficientNet)
  <name>_w8a16lnfp.tflite  W8A16 with float LayerNorm (opt-in, ViT)
  <name>_torch_test_logits.npy  PyTorch fp32 reference on the cached test set

Graph contract (all variants): input float32 [1, S, S, 3], RGB in [0, 1] after the
project's letterbox resize (prepare_npu_inputs.py); ImageNet normalisation is
in-graph; output float32 [1, 4] logits in class order benign_lesion, cancer,
healthy, opmd.

Run in the export venv (litert-torch 0.9.4, ai-edge-quantizer 0.9.0, torch 2.13 CPU):
    ~/nvme_tmp/oral-lesions-npu/env/bin/python scripts/npu/export_npu_tflite.py \
        --arch b3 --ckpt results/.../best_model_study.pth --name b3_global_val \
        --inputs ~/nvme_tmp/oral-lesions-npu/work/inputs --out ~/nvme_tmp/oral-lesions-npu/work/models
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ARCHS = {
    "b3": dict(kind="effnet", backbone="tf_efficientnet-b3_ns", size=384),
    "b7": dict(kind="effnet", backbone="tf_efficientnet-b7_ns", size=512),
    "vit_b": dict(kind="vit", backbone="vit_base_patch16_224", size=224),
    "vit_l": dict(kind="vit", backbone="vit_large_patch16_224", size=224),
}
MEAN = (0.485, 0.456, 0.406)  # A.Normalize() defaults == IMAGENET_MEAN/STD used for ViT
STD = (0.229, 0.224, 0.225)


STAGED_MEAN_MAX = 256


def staged_mean(x):
    """Spatial mean as repeated 4x4 average pools, then a mean over <= 256 values.

    Same value as x.mean((2, 3), keepdim=True). The Qualcomm HTP computes one large
    spatial mean (e.g. 192x192 in the first SE block of B3 at 384 px) inaccurately;
    see docs/NPU_EXPORT.md.
    """
    while x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0 and x.shape[2] * x.shape[3] > STAGED_MEAN_MAX:
        x = torch.nn.functional.avg_pool2d(x, 4)
    return x.mean((2, 3), keepdim=True)


def _staged_se_forward(self, x):
    x_se = staged_mean(x)
    x_se = self.act1(self.conv_reduce(x_se))
    return x * self.gate_fn(self.conv_expand(x_se))


def _rank4_attention_forward(self, x, attn_mask=None):
    """timm Attention.forward with the qkv split done by chunk + rank-4 permutes.

    timm reshapes qkv to [B, N, 3, heads, dim] and permutes that rank-5 tensor. QAIRT
    2.50 rejects int16 rank-5 TRANSPOSE (ViT-B W8A16: 12 ops on CPU, 13 partitions;
    ViT-L W8A16: compilation fails). Same q, k, v values; the rest is unchanged.
    """
    B, N, _ = x.shape
    q, k, v = (t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
               for t in self.qkv(x).chunk(3, dim=-1))
    q, k = self.q_norm(q), self.k_norm(k)
    if self.fused_attn:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        x = attn.softmax(dim=-1) @ v
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    return self.proj_drop(self.proj(self.norm(x)))


class NpuClassifier(torch.nn.Module):
    """NHWC [0, 1] RGB in, logits out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("inv_std", 1.0 / torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) * self.inv_std
        if hasattr(self.model, "enet"):
            # Same math as enetv2.forward in eval mode. Pooling with mean + reshape avoids
            # the as_strided that flatten(1) leaves after the NHWC permute, which LiteRT
            # lowers to a GATHER_ND that QNN does not take (2 NPU partitions).
            features = self.model.enet.features(x).mean(dim=(2, 3))
            return self.model.myfc(features)
        return self.model(x)


def build_model(arch, ckpt, se_mean="staged", attention="timm"):
    import geffnet
    from geffnet.efficientnet_builder import SqueezeExcite
    from src.models import ViTFineTuner, enetv2
    spec = ARCHS[arch]
    if spec["kind"] == "effnet":
        # Exportable layers: static "same" padding and plain (non-autograd.Function) Swish.
        with geffnet.config.set_exportable(True), geffnet.config.set_no_jit(True):
            model = enetv2(spec["backbone"], out_dim=4, load_pretrained_geffnet=False)
    else:
        model = ViTFineTuner(model_name=spec["backbone"], out_dim=4, timm_imagenet_pretrained=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if attention == "rank4":
        from timm.layers.attention import Attention
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = _rank4_attention_forward.__get__(module)
    if se_mean == "staged":
        for module in model.modules():
            if isinstance(module, SqueezeExcite):
                module.forward = _staged_se_forward.__get__(module)
    return NpuClassifier(model).eval()


def to_float(images):
    return images.astype(np.float32) / 255.0


def torch_logits(model, images, batch=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out.append(model(torch.from_numpy(to_float(images[i:i + batch]))).numpy())
    return np.concatenate(out)


def w8a16_with_float(operations=(), regexes=()):
    """W8A16 recipe that leaves some ops unquantized (float, i.e. fp16 on the HTP)."""
    from ai_edge_quantizer import recipe
    base = recipe.static_wi8_ai16()
    keep_float = [{"operation": op, "regex": ".*"} for op in operations] + [{"operation": "*", "regex": r} for r in regexes]
    return base + [dict(json.loads(json.dumps(base[0])), **entry, algorithm_key="no_quantize") for entry in keep_float]


def w8a16_float_depthwise():
    """EfficientNet: QAIRT 2.50 leaves int16 DEPTHWISE_CONV_2D and PAD on the CPU (B3: 27
    ops, 28 partitions), and int8 activations there wreck accuracy (B3 top-1 0.61 -> 0.29).
    The SE blocks' large spatial MEAN is also kept in float (see w8a16_float_layernorm)."""
    return w8a16_with_float(operations=("DEPTHWISE_CONV_2D", "PAD", "MEAN"))


def w8a16_float_layernorm():
    """ViT: the HTP computes the int16 LayerNorm decomposition wrongly (ViT-B block 0
    norm1: cos 0.90 vs LiteRT CPU on the same int16 graph)."""
    return w8a16_with_float(regexes=(".*LayerNorm.*",))


def quantize(fp32_path, out_path, recipe_fn, calib_images):
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer
    signature = list(Interpreter(model_path=str(fp32_path)).get_signature_list().items())[0]
    sig_key, sig = signature
    input_name = sig["inputs"][0]
    qt = quantizer.Quantizer(str(fp32_path))
    qt.load_quantization_recipe(recipe_fn())
    calib = {sig_key: [{input_name: to_float(x[None])} for x in calib_images]}
    started = time.time()
    result = qt.quantize(qt.calibrate(calib))
    result.export_model(str(out_path), overwrite=True)
    return time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="dir with test_<S>.npz and train_<S>.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default="fp32,w8a8,w8a16")
    parser.add_argument("--calib-count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--reference-logits", type=Path, help="Original-checkpoint FP32 logits on these exact cached test rows")
    parser.add_argument("--verify-reference-count", type=int, default=24,
                        help="Class-stratified CPU adapter checks when supplying reference logits")
    parser.add_argument("--se-mean", choices=("staged", "plain"), default="staged",
                        help="EfficientNet SE spatial mean: staged (HTP-safe, same math) or plain x.mean")
    parser.add_argument("--attention", choices=("timm", "rank4"), default="timm",
                        help="ViT qkv split: timm (rank-5 permute) or rank4 (needed for int16 on QAIRT 2.50)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    size = ARCHS[args.arch]["size"]
    test = np.load(args.inputs / f"test_{size}.npz")
    calib = np.load(args.inputs / f"train_{size}.npz")["images"][:args.calib_count]
    model = build_model(args.arch, args.ckpt, args.se_mean, args.attention)
    report = {"name": args.name, "arch": args.arch, "ckpt": str(args.ckpt.resolve()), "size": size, "seMean": args.se_mean, "attention": args.attention,
              "classes": test["classes"].tolist(), "calibCount": len(calib)}

    ref_path = args.out / f"{args.name}_torch_test_logits.npy"
    if args.reference_logits and not ref_path.exists():
        reference = np.load(args.reference_logits)
        if reference.shape != (len(test["images"]), 4) or not np.isfinite(reference).all():
            raise ValueError("Reference logits have incorrect shape or non-finite values")
        count = min(args.verify_reference_count, len(reference))
        if count < 4:
            raise ValueError("At least four reference checks are required")
        groups = [np.flatnonzero(test["labels"] == label).tolist() for label in np.unique(test["labels"])]
        indices = []
        while len(indices) < count:
            for group in groups:
                if group and len(indices) < count:
                    indices.append(group.pop(0))
        actual = torch_logits(model, test["images"][indices], batch=1)
        error = float(np.abs(actual - reference[indices]).max())
        agreement = float((actual.argmax(1) == reference[indices].argmax(1)).mean())
        report["referenceSource"] = str(args.reference_logits.resolve())
        report["adapterReferenceCheck"] = {"images": count, "top1Agreement": agreement, "maxAbsLogitError": error}
        if error > 0.01 or agreement < 1.0:
            raise ValueError(f"Export adapter differs from the original checkpoint: {report['adapterReferenceCheck']}")
        np.save(ref_path, reference)
    if not ref_path.exists():
        started = time.time()
        np.save(ref_path, torch_logits(model, test["images"]))
        report["torchSeconds"] = round(time.time() - started, 1)
    ref = np.load(ref_path)
    report["torchTestAcc"] = float((ref.argmax(1) == test["labels"]).mean())

    import litert_torch
    fp32_path = args.out / f"{args.name}_fp32.tflite"
    sample = torch.from_numpy(to_float(test["images"][:1]))
    if not fp32_path.exists():
        started = time.time()
        litert_torch.convert(model, (sample,)).export(str(fp32_path))
        report["convertSeconds"] = round(time.time() - started, 1)
    from ai_edge_quantizer import recipe
    recipes = {"w8a8": recipe.static_wi8_ai8, "w8a16": recipe.static_wi8_ai16, "w8a16dwfp": w8a16_float_depthwise,
               "w8a16lnfp": w8a16_float_layernorm}
    for variant in args.variants.split(","):
        if variant in recipes:
            path = args.out / f"{args.name}_{variant}.tflite"
            if not path.exists():
                report[f"{variant}QuantizeSeconds"] = round(quantize(fp32_path, path, recipes[variant], calib), 1)
    report["files"] = {p.name: p.stat().st_size for p in sorted(args.out.glob(f"{args.name}_*.tflite"))}
    (args.out / f"{args.name}_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

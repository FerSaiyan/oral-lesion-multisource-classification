#!/usr/bin/env python3
"""AOT-compile an exported classifier for a Qualcomm SoC and score it on the test set.

For one plain .tflite (from export_npu_tflite.py):
  1. AOT compile with the installed QAIRT (ai-edge-litert-sdk-qualcomm) and record
     the partition stats (full offload = all ops, one partition).
  2. Read the QNN context's build, SoC and DSP architecture (qnn-context-binary-utility).
  3. LiteRT CPU logits on the cached test set (the quantized graph's own numerics).
  4. QNN HTP emulation (the SDK's x86 libQnnHtp.so) on a class-stratified subset,
     or raw outputs dumped from a phone (--npu-raws) on any subset.
  5. Task metrics (top-1, macro/weighted F1, suspicious-vs-not F1, top-2) per
     backend, and argmax agreement with the PyTorch fp32 reference.

Run in the QAIRT 2.50 venv (~/nvme_tmp/npu-litert230/env: ai-edge-litert 2.3.0,
tflite 2.18):
    python scripts/npu/npu_aot_eval.py work/models/b3_global_val_fp32.tflite \
        --test work/inputs/test_384.npz --torch-logits work/models/b3_global_val_torch_test_logits.npy \
        --qairt-root ~/nvme_tmp/npu-litert230/qairt/2.50.0.260828 --emu-count 100 --out work/eval
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tflite
from flatbuffers import flexbuffers

SUSPICIOUS = ("opmd", "cancer")


def dispatch_contexts(data):
    model = tflite.Model.GetRootAsModel(data, 0)
    found = []
    for s in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(s)
        for o in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(o)
            if model.OperatorCodes(op.OpcodeIndex()).CustomCode() == b"DISPATCH_OP":
                options = flexbuffers.Loads(bytes(op.CustomOptionsAsNumpy()))
                found.append(data[options["bytecode_offset"]:options["bytecode_offset"] + options["bytecode_size"]])
    return found


def aot_compile(model_path, out_dir, soc):
    from ai_edge_litert.aot import aot_compile as aot
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
    started = time.time()
    result = aot.aot_compile(str(model_path), output_dir=str(out_dir),
                             target=qnn_target.Target(qnn_target.SocModel[soc]), keep_going=False)
    assert len(result.models_with_backend) == 1, result.compilation_report()
    _, compiled = result.models_with_backend[0]
    stats = [dict(offloaded=s.num_ops_offloaded, total=s.num_total_ops, partitions=s.num_partitions_offloaded)
             for s in compiled.partition_stats.subgraph_stats]
    return Path(compiled.path), stats, round(time.time() - started, 1)


def find(info, key):
    if isinstance(info, dict):
        if key in info:
            return info[key]
        values = info.values()
    elif isinstance(info, list):
        values = info
    else:
        return None
    for value in values:
        found = find(value, key)
        if found is not None:
            return found
    return None


def context_info(context, qairt_root, work):
    binary, info_path = work / "ctx.bin", work / "ctx.json"
    binary.write_bytes(context)
    subprocess.run(["/lib64/ld-linux-x86-64.so.2", str(qairt_root / "bin/x86_64-linux-clang/qnn-context-binary-utility"),
                    "--context_binary", str(binary), "--json_file", str(info_path)], check=True, capture_output=True,
                   env={"LD_LIBRARY_PATH": str(qairt_root / "lib/x86_64-linux-clang")})
    info = json.loads(info_path.read_text())
    return {key: find(info, key) for key in ("buildId", "socModel", "dspArch")} | {"bytes": len(context)}


def litert_cpu(model_path, images):
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(model_path), num_threads=16)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    # Statically quantized graphs (ai-edge-quantizer) have integer I/O; quantize the
    # input and dequantize the output with the tensors' own parameters, as qnn-net-run does.
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    logits = []
    for x in images:
        value = x[None].astype(np.float32) / 255.0
        if in_scale:
            info = np.iinfo(inp["dtype"])
            value = np.clip(np.round(value / in_scale) + in_zero, info.min, info.max).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], value)
        interpreter.invoke()
        result = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        logits.append((result - out_zero) * out_scale if out_scale else result)
    return np.stack(logits)


def htp_emulate(context, images, qairt_root, work):
    work.mkdir(parents=True, exist_ok=True)
    (work / "ctx.bin").write_bytes(context)
    lines = []
    for i, x in enumerate(images):
        path = work / f"in{i}.raw"
        (x.astype(np.float32) / 255.0).tofile(path)
        lines.append(str(path))
    (work / "inputs.txt").write_text("\n".join(lines) + "\n")
    bin_dir, lib_dir = qairt_root / "bin/x86_64-linux-clang", qairt_root / "lib/x86_64-linux-clang"
    started = time.time()
    # The dynamic loader runs the tool even where the SDK lost its executable bits.
    subprocess.run(["/lib64/ld-linux-x86-64.so.2", str(bin_dir / "qnn-net-run"), "--backend", str(lib_dir / "libQnnHtp.so"),
                    "--retrieve_context", str(work / "ctx.bin"), "--input_list", str(work / "inputs.txt"),
                    "--output_dir", str(work / "out")], check=True, capture_output=True, env={"LD_LIBRARY_PATH": str(lib_dir)})
    seconds = time.time() - started
    return read_raws(work / "out", len(images)), seconds


def read_raws(out_dir, count):
    logits = []
    for i in range(count):
        raws = sorted((out_dir / f"Result_{i}").glob("*.raw"))
        assert len(raws) == 1, f"expected one output tensor, got {raws}"
        logits.append(np.fromfile(raws[0], np.float32))
    return np.stack(logits)


def stratified_subset(labels, count, seed=0):
    rng = np.random.default_rng(seed)
    per_class = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in np.unique(labels)]
    order = []
    while any(per_class):  # round-robin so small classes are represented
        for queue in per_class:
            if queue:
                order.append(queue.pop(0))
    return np.sort(np.array(order[:count]))


def f1_score(y_true, y_pred, average=None):
    """sklearn-compatible F1 (macro / weighted / binary with positive label True)."""
    if average is None:
        y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
        tp, fp, fn = (y_true & y_pred).sum(), (~y_true & y_pred).sum(), (y_true & ~y_pred).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = np.array([f1_score(y_true == c, y_pred == c) for c in labels])
    if average == "macro":
        return scores.mean()
    support = np.array([(y_true == c).sum() for c in labels])
    return (scores * support).sum() / support.sum()


def metrics(logits, labels, classes, reference=None):
    pred = logits.argmax(1)
    suspicious = np.isin(np.array(classes), SUSPICIOUS)
    top2 = np.argsort(-logits, axis=1)[:, :2]
    result = {"n": int(len(labels)), "top1": float((pred == labels).mean()),
              "top2": float((top2 == labels[:, None]).any(1).mean()),
              "f1Macro": float(f1_score(labels, pred, average="macro")),
              "f1Weighted": float(f1_score(labels, pred, average="weighted")),
              "f1Suspicious": float(f1_score(suspicious[labels], suspicious[pred]))}
    if reference is not None:
        result["agreeTorchTop1"] = float((pred == reference.argmax(1)).mean())
        result["maxAbsLogitErr"] = float(np.abs(logits - reference).max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="plain (not yet AOT-compiled) .tflite")
    parser.add_argument("--test", type=Path, required=True, help="test_<S>.npz from prepare_npu_inputs.py")
    parser.add_argument("--torch-logits", type=Path, required=True)
    parser.add_argument("--qairt-root", type=Path, required=True)
    parser.add_argument("--soc", default="SM8650")
    parser.add_argument("--emu-count", type=int, default=100, help="HTP-emulated test images (0 = skip)")
    parser.add_argument("--cpu-count", type=int, default=0,
                        help="LiteRT CPU on a stratified subset (0 = all); int16 graphs use slow reference kernels")
    parser.add_argument("--npu-raws", type=Path, help="phone qnn-net-run output dir (Result_i/*.raw) for --subset")
    parser.add_argument("--subset", type=Path, help=".npy of test indices the phone ran")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem
    test = np.load(args.test)
    images, labels, classes = test["images"], test["labels"], test["classes"].tolist()
    reference = np.load(args.torch_logits)
    report = {"model": str(args.model), "soc": args.soc, "qairtRoot": str(args.qairt_root)}

    aot_dir = args.out / "aot"
    compiled, report["partitionStats"], report["aotSeconds"] = aot_compile(args.model, aot_dir, args.soc)
    report["aotModel"] = str(compiled)
    report["aotBytes"] = compiled.stat().st_size
    contexts = dispatch_contexts(compiled.read_bytes())
    report["contexts"] = len(contexts)
    with tempfile.TemporaryDirectory() as tmp:
        report["context"] = context_info(contexts[0], args.qairt_root, Path(tmp)) if contexts else None

    cpu_path = args.out / f"{stem}_litert_cpu_logits.npy"
    if not cpu_path.exists():
        # Rows not run stay NaN; stratified prefixes nest, so the emulation subset is covered.
        cpu_rows = stratified_subset(labels, args.cpu_count) if args.cpu_count else np.arange(len(labels))
        cpu_all = np.full_like(reference, np.nan, dtype=np.float32)
        cpu_all[cpu_rows] = litert_cpu(args.model, images[cpu_rows])
        np.save(cpu_path, cpu_all)
    cpu = np.load(cpu_path)
    ran = np.flatnonzero(np.isfinite(cpu).all(1))
    report["litertCpu"] = metrics(cpu[ran], labels[ran], classes, reference[ran])
    report["torch"] = metrics(reference, labels, classes)

    subset = None
    if args.npu_raws:
        subset = np.load(args.subset)
        npu = read_raws(args.npu_raws, len(subset))
        report["npuDevice"] = metrics(npu, labels[subset], classes, reference[subset])
        report["npuDevice"]["agreeLitertCpuTop1"] = float((npu.argmax(1) == cpu[subset].argmax(1)).mean())
    elif args.emu_count and len(contexts) == 1:
        subset = stratified_subset(labels, args.emu_count)
        np.save(args.out / f"{stem}_emu_subset.npy", subset)
        with tempfile.TemporaryDirectory() as tmp:
            htp, seconds = htp_emulate(contexts[0], images[subset], args.qairt_root, Path(tmp))
        np.save(args.out / f"{stem}_htp_emu_logits.npy", htp)
        report["htpEmulation"] = metrics(htp, labels[subset], classes, reference[subset])
        report["htpEmulation"]["agreeLitertCpuTop1"] = float((htp.argmax(1) == cpu[subset].argmax(1)).mean())
        report["htpEmulation"]["allFinite"] = bool(np.isfinite(htp).all())
        report["htpEmulation"]["secondsPerImage"] = round(seconds / len(subset), 2)
    if subset is not None:
        report["subsetTorch"] = metrics(reference[subset], labels[subset], classes)
        report["subsetLitertCpu"] = metrics(cpu[subset], labels[subset], classes, reference[subset])
    (args.out / f"{stem}_npu_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

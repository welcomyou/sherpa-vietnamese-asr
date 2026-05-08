#!/usr/bin/env python3
"""
Export asteroid Conv-TasNet 16k to ONNX (FP32).

Source: JorisCos/ConvTasNet_Libri2Mix_sepclean_16k (https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)
Use case: 2-speaker overlap separation, 16 kHz, mono input -> 2 source streams.

Why FP32 only? INT8 dynamic quantization of this model produces poor SNR
(~17 dB vs PyTorch) and runs ~5x SLOWER than FP32 on ORT CPU EP — Conv-TasNet's
many small Conv1d layers don't benefit from dynamic quantization. So: skip INT8.

Why ONNX? The PyTorch+asteroid runtime drags ~2 GB of deps (torch, asteroid,
asteroid_filterbanks, julius, einops). Exporting to ONNX lets the runtime use
~50 MB onnxruntime instead — same accuracy (SNR > 60 dB vs PyTorch on speech),
~1.5x faster cold start, no torch needed at inference time.

Usage:
    python convert_onnx/export_convtasnet_onnx.py \\
        --output models/convtasnet-libri2mix-16k/convtasnet_16k.onnx \\
        --verify

Requires (only at export time): torch, asteroid, asteroid_filterbanks, onnxruntime, numpy
"""

import argparse
import os
import sys
import time
from pathlib import Path


def export(output_path: str, opset: int = 17, verify: bool = False) -> str:
    import numpy as np
    import torch
    from asteroid.models import ConvTasNet

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading asteroid Conv-TasNet (JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)...")
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    model.eval()

    SR = 16000
    dummy = torch.randn(1, SR * 5)  # 5s seed for trace

    print(f"[2/3] Exporting ONNX (opset {opset}) -> {output_path}")
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["mixture"],
        output_names=["sources"],
        dynamic_axes={
            "mixture": {0: "batch", 1: "time"},
            "sources": {0: "batch", 2: "time"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"      Done in {elapsed:.1f}s. File size: {size_mb:.1f} MB")

    if verify:
        print("[3/3] Verifying ONNX matches PyTorch (5s random input)...")
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(str(output_path), opts, providers=["CPUExecutionProvider"])

        np.random.seed(0)
        x = np.random.randn(1, SR * 5).astype(np.float32) * 0.3
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x)).numpy()
        ort_out = sess.run(None, {"mixture": x})[0]

        T = min(pt_out.shape[-1], ort_out.shape[-1])
        pt_out, ort_out = pt_out[..., :T], ort_out[..., :T]
        sig_pow = (pt_out ** 2).mean()
        err_pow = ((pt_out - ort_out) ** 2).mean()
        snr_db = 10 * np.log10(sig_pow / err_pow) if err_pow > 0 else float("inf")
        max_diff = np.abs(pt_out - ort_out).max()
        print(f"      max|PyTorch - ONNX| = {max_diff:.4e}")
        print(f"      SNR vs PyTorch      = {snr_db:.1f} dB  (>40 dB = near-identical)")
        if snr_db < 40:
            print("      WARNING: low SNR — verify export integrity")
            return str(output_path)
        print("      PASS")
    else:
        print("[3/3] Skipped verification (--verify to enable)")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="models/convtasnet-libri2mix-16k/convtasnet_16k.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch (random 5s input, SNR check)",
    )
    args = parser.parse_args()
    export(args.output, opset=args.opset, verify=args.verify)


if __name__ == "__main__":
    main()

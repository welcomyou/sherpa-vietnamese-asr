#!/usr/bin/env python3
"""
Split pyannote Community-1 embedding ONNX into encoder + projection weights.

Input:  embedding_model.onnx (full pipeline: fbank -> encoder -> stats_pool -> Gemm -> 256d)
Output: embedding_encoder.onnx     - everything before stats_pool, output frame features (B, 256, F)
        resnet_seg_1_weight.npy    - final Gemm weight (256, 5120)
        resnet_seg_1_bias.npy      - final Gemm bias (256,)

Why split? Stats pooling in the full graph operates over ALL frames of the input.
For diarization with overlapping speakers, we need a MASKED stats pool (weighted by
per-speaker activity mask). Doing this in NumPy after running the encoder once per
batch of chunks is ~30x faster than running the full model once per (chunk, speaker).
See core/speaker_diarization_pure_ort.py for usage.

Usage:
    python convert_onnx/split_pyannote_embedding.py \\
        --input  models/pyannote-onnx/embedding_model.onnx \\
        --output_dir models/pyannote-onnx/

Requires: onnx>=1.14, numpy
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ENCODER_OUTPUT_TENSOR = "/resnet/pool/Reshape_output_0"
GEMM_WEIGHT_NAME = "resnet.seg_1.weight"
GEMM_BIAS_NAME = "resnet.seg_1.bias"


def split(input_path: str, output_dir: str) -> None:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_path}")

    print(f"[1/3] Loading {input_path.name}...")
    model = onnx.load(str(input_path))
    input_name = model.graph.input[0].name

    print(f"[2/3] Extracting encoder subgraph (output: {ENCODER_OUTPUT_TENSOR})...")
    encoder_path = output_dir / "embedding_encoder.onnx"
    onnx.utils.extract_model(
        str(input_path),
        str(encoder_path),
        input_names=[input_name],
        output_names=[ENCODER_OUTPUT_TENSOR],
    )
    enc_size_mb = encoder_path.stat().st_size / 1024 / 1024
    print(f"      -> {encoder_path.name} ({enc_size_mb:.1f} MB)")

    print(f"[3/3] Extracting Gemm projection weights -> .npy...")
    found = {}
    for init in model.graph.initializer:
        if init.name == GEMM_WEIGHT_NAME:
            found["weight"] = numpy_helper.to_array(init)
        elif init.name == GEMM_BIAS_NAME:
            found["bias"] = numpy_helper.to_array(init)

    if "weight" not in found or "bias" not in found:
        raise RuntimeError(
            f"Could not find Gemm initializers ({GEMM_WEIGHT_NAME}, {GEMM_BIAS_NAME}) "
            f"in {input_path.name}. Is this the right model?"
        )

    weight_path = output_dir / "resnet_seg_1_weight.npy"
    bias_path = output_dir / "resnet_seg_1_bias.npy"
    np.save(weight_path, found["weight"])
    np.save(bias_path, found["bias"])
    print(f"      -> {weight_path.name} (shape={found['weight'].shape}, "
          f"{weight_path.stat().st_size / 1024:.1f} KB)")
    print(f"      -> {bias_path.name} (shape={found['bias'].shape}, "
          f"{bias_path.stat().st_size:.0f} B)")

    print("\nDone. Verify with core/speaker_diarization_pure_ort.py — it auto-detects")
    print("embedding_encoder.onnx + resnet_seg_1_*.npy and uses the fast batched path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default="models/pyannote-onnx/embedding_model.onnx",
        help="Path to full embedding_model.onnx (download from "
             "https://huggingface.co/altunenes/speaker-diarization-community-1-onnx)",
    )
    parser.add_argument(
        "--output_dir",
        default="models/pyannote-onnx/",
        help="Output directory for encoder + .npy files",
    )
    args = parser.parse_args()
    split(args.input, args.output_dir)


if __name__ == "__main__":
    main()

"""
Export ViBERT Seq2Labels (punctuation + capitalization) sang ONNX.

File này tự chứa toàn bộ model definition (Seq2LabelsConfig, Seq2LabelsModel)
nên không cần import từ core/. Chỉ cần PyTorch + transformers để chạy export.
Sau khi export xong, runtime chỉ cần onnxruntime + numpy.

Cách dùng:
    python convert_onnx/export_vibert_onnx.py
    python convert_onnx/export_vibert_onnx.py --model_dir models/vibert-capu --output models/vibert-capu/vibert-capu.onnx
    python convert_onnx/export_vibert_onnx.py --model_dir models/vibert-capu --verify
"""

import argparse
import gc
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput


# ============================================================
# Seq2LabelsConfig (inline — tự chứa, không cần file riêng)
# ============================================================
class Seq2LabelsConfig(PretrainedConfig):
    """Config cho Seq2Labels model (ViBERT punctuation + capitalization)."""
    model_type = "bert"

    def __init__(
        self,
        pretrained_name_or_path="bert-base-cased",
        vocab_size=15,
        num_detect_classes=4,
        load_pretrained=False,
        initializer_range=0.02,
        pad_token_id=0,
        use_cache=True,
        predictor_dropout=0.0,
        special_tokens_fix=False,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.num_detect_classes = num_detect_classes
        self.pretrained_name_or_path = pretrained_name_or_path
        self.load_pretrained = load_pretrained
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.predictor_dropout = predictor_dropout
        self.special_tokens_fix = special_tokens_fix
        self.label_smoothing = label_smoothing


# ============================================================
# Seq2LabelsModel (inline — tự chứa, không cần file riêng)
# ============================================================
def _get_range_vector(size, device):
    return torch.arange(0, size, dtype=torch.long, device=device)


@dataclass
class Seq2LabelsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    detect_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    max_error_probability: Optional[torch.FloatTensor] = None


class Seq2LabelsModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_detect_classes = config.num_detect_classes
        self.label_smoothing = config.label_smoothing

        if config.load_pretrained:
            self.bert = AutoModel.from_pretrained(config.pretrained_name_or_path)
            bert_config = self.bert.config
        else:
            if hasattr(config, "hidden_size") and config.hidden_size:
                bert_config = deepcopy(config)
                bert_config.vocab_size = getattr(config, "bert_vocab_size", 38168)
            else:
                bert_config = AutoConfig.from_pretrained(config.pretrained_name_or_path)
            self.bert = AutoModel.from_config(bert_config)

            try:
                param = next(self.bert.parameters())
                if param.device.type == "meta":
                    self.bert.to_empty(device="cpu")
                    self.bert.to(torch.float32)
                    self.bert.init_weights()
            except Exception:
                pass

        if config.special_tokens_fix:
            if hasattr(self.bert, "embeddings"):
                self.bert.embeddings.to(torch.float32)
            if hasattr(self.bert, "word_embedding"):
                self.bert.word_embedding.to(torch.float32)
            try:
                vocab_size = self.bert.embeddings.word_embeddings.num_embeddings
            except AttributeError:
                vocab_size = self.bert.word_embedding.num_embeddings + 5
            self.bert.resize_token_embeddings(vocab_size + 1)

        predictor_dropout = config.predictor_dropout if config.predictor_dropout is not None else 0.0
        self.dropout = nn.Dropout(predictor_dropout)
        self.classifier = nn.Linear(bert_config.hidden_size, config.vocab_size)
        self.detector = nn.Linear(bert_config.hidden_size, config.num_detect_classes)

        if self.classifier.weight.device.type == "meta":
            self.classifier.to_empty(device="cpu")
            self.detector.to_empty(device="cpu")

        self.post_init()

    def forward(
        self,
        input_ids=None,
        input_offsets=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        d_tags=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if input_offsets is not None:
            range_vector = _get_range_vector(input_offsets.size(0), device=sequence_output.device).unsqueeze(1)
            sequence_output = sequence_output[range_vector, input_offsets]

        logits = self.classifier(self.dropout(sequence_output))
        logits_d = self.detector(sequence_output)

        loss = None
        if labels is not None and d_tags is not None:
            from torch.nn import CrossEntropyLoss
            loss_labels_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss_d_fct = CrossEntropyLoss()
            loss = loss_labels_fct(logits.view(-1, self.num_labels), labels.view(-1)) + \
                   loss_d_fct(logits_d.view(-1, self.num_detect_classes), d_tags.view(-1))

        if not return_dict:
            output = (logits, logits_d) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Seq2LabelsOutput(
            loss=loss,
            logits=logits,
            detect_logits=logits_d,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            max_error_probability=torch.ones(logits.size(0), device=logits.device),
        )


# ============================================================
# Wrapper cho export (chỉ trả logits + detect_logits)
# ============================================================
class _ViBERTForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids, input_offsets):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_offsets=input_offsets,
        )
        return out.logits, out.detect_logits


# ============================================================
# Main export logic
# ============================================================
START_TOKEN = "$START"


def export(model_dir, output_path, opset_version=14, verify=False):
    """
    Export ViBERT Seq2Labels model sang ONNX.

    Args:
        model_dir: Thư mục chứa config.json + pytorch_model.bin + vocab.txt
        output_path: Đường dẫn file .onnx output
        opset_version: ONNX opset (mặc định 14)
        verify: Nếu True, so sánh output PyTorch vs ONNX sau khi export
    """
    print(f"Model dir:  {model_dir}")
    print(f"Output:     {output_path}")

    # --- 1. Load PyTorch model ---
    print("\n[1/4] Loading PyTorch model...")
    config = AutoConfig.from_pretrained(model_dir)
    model = Seq2LabelsModel(config)

    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"pytorch_model.bin not found at {bin_path}")

    state_dict = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    model.eval()
    print("       OK")

    # --- 2. Load tokenizer + build dummy input ---
    print("[2/4] Building dummy input...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, do_basic_tokenize=False, do_lower_case=False, model_max_length=1024
    )
    tokenizer.add_tokens([START_TOKEN])

    dummy_text = "kính thưa các đồng chí tiếp nối chương trình chất vấn"
    words = dummy_text.split()
    token_batch = [[START_TOKEN] + words]

    batch = tokenizer(
        token_batch,
        return_tensors="pt",
        padding=True,
        is_split_into_words=True,
        truncation=True,
        add_special_tokens=False,
    )

    # Build input_offsets (word → subword mapping)
    word_ids = batch.word_ids(batch_index=0)
    offsets = [0]
    for j in range(1, len(word_ids)):
        if word_ids[j] != word_ids[j - 1]:
            offsets.append(j)
    input_offsets = torch.tensor([offsets], dtype=torch.long)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]
    print(f"       input_ids: {input_ids.shape}, offsets: {input_offsets.shape}")

    # --- 3. Export to ONNX ---
    print(f"[3/4] Exporting ONNX (opset {opset_version})...")
    export_model = _ViBERTForExport(model)
    export_model.eval()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    torch.onnx.export(
        export_model,
        (input_ids, attention_mask, token_type_ids, input_offsets),
        output_path,
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask", "token_type_ids", "input_offsets"],
        output_names=["logits", "detect_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "input_offsets": {0: "batch", 1: "num_words"},
            "logits": {0: "batch", 1: "num_words"},
            "detect_logits": {0: "batch", 1: "num_words"},
        },
    )
    dt = time.time() - t0

    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    bin_size = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"       Done in {dt:.1f}s")
    print(f"       pytorch_model.bin: {bin_size:.1f} MB")
    print(f"       vibert-capu.onnx:  {onnx_size:.1f} MB")

    # --- 4. Verify (optional) ---
    if verify:
        print("[4/4] Verifying ONNX output...")
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "token_type_ids": token_type_ids.numpy(),
            "input_offsets": input_offsets.numpy(),
        }

        with torch.no_grad():
            pt_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_offsets=input_offsets,
            )
        pt_logits = pt_out.logits.numpy()

        ort_logits, ort_detect = sess.run(None, ort_inputs)

        max_diff = np.abs(pt_logits - ort_logits).max()
        labels_match = np.array_equal(pt_logits.argmax(axis=-1), ort_logits.argmax(axis=-1))

        print(f"       Max logits diff:  {max_diff:.6f}")
        print(f"       Labels match:     {labels_match}")

        if labels_match and max_diff < 0.01:
            print("       VERIFY OK")
        else:
            print("       VERIFY WARNING: outputs differ!")
    else:
        print("[4/4] Skipping verification (use --verify to enable)")

    print(f"\nExport complete: {output_path}")


def quantize_int8(fp32_path, int8_path=None):
    """Quantize ONNX fp32 → int8 (dynamic quantization).
    Giảm ~75% kích thước (439MB → 111MB), tốc độ tương đương trên CPU.

    Args:
        fp32_path: Đường dẫn file .onnx fp32
        int8_path: Output path (default: thay .onnx thành .int8.onnx)
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if int8_path is None:
        int8_path = fp32_path.replace(".onnx", ".int8.onnx")

    print(f"\n[Quantize] {fp32_path}")
    print(f"           → {int8_path}")

    t0 = time.time()
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
    )
    elapsed = time.time() - t0

    fp32_size = os.path.getsize(fp32_path) / 1024 / 1024
    int8_size = os.path.getsize(int8_path) / 1024 / 1024
    print(f"           fp32: {fp32_size:.0f} MB → int8: {int8_size:.0f} MB "
          f"({int8_size/fp32_size*100:.0f}%) in {elapsed:.1f}s")

    return int8_path


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Export ViBERT Seq2Labels model to ONNX (fp32 + int8)"
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "vibert-capu"),
        help="Directory containing config.json + pytorch_model.bin + vocab.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx path (default: <model_dir>/vibert-capu.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch after export",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip int8 quantization (chỉ export fp32)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.model_dir, "vibert-capu.onnx")

    # Export fp32
    export(
        model_dir=args.model_dir,
        output_path=args.output,
        opset_version=args.opset,
        verify=args.verify,
    )

    # Quantize int8
    if not args.no_int8:
        quantize_int8(args.output)


if __name__ == "__main__":
    main()

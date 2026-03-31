"""Wrapper of Seq2Labels model. Fixes errors based on model predictions.
Uses ONNX Runtime for inference — no PyTorch dependency."""
from collections import defaultdict
from difflib import SequenceMatcher
import logging
import os
import re
from time import time
from typing import List, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from core.vocabulary import Vocabulary
from core.gec_utils import PAD, UNK, START_TOKEN, get_target_sent_by_edits

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _pad_sequences(sequences, padding_value=0):
    """Pad list of 1D int arrays to same length, return 2D int64 ndarray."""
    max_len = max(len(s) for s in sequences)
    result = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
    for i, s in enumerate(sequences):
        result[i, :len(s)] = s
    return result


class GecBERTModel:
    def __init__(
        self,
        vocab_path=None,
        model_paths=None,
        weights=None,
        device=None,
        max_len=64,
        min_len=3,
        lowercase_tokens=False,
        log=False,
        iterations=None,
        min_error_probability=0.0,
        confidence=0,
        resolve_cycles=False,
        split_chunk=False,
        chunk_size=48,
        overlap_size=12,
        min_words_cut=6,
        punc_dict={':', ".", ",", "?"},
        case_confidence=0.0,
    ):
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.model_weights = list(map(float, weights)) if weights else [1] * len(model_paths)

        # device param giữ lại cho tương thích API, nhưng ONNX tự chọn provider
        self.device = device

        if iterations is None:
            self.iterations = 3
        else:
            self.iterations = iterations
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.incorr_index = self.vocab.get_token_index("INCORRECT", "d_tags")
        self.log = log
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        assert (
            chunk_size > 0 and chunk_size // 2 >= overlap_size
        ), "Chunk merging required overlap size must be smaller than half of chunk size"
        self.split_chunk = split_chunk
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_words_cut = min_words_cut
        self.stride = chunk_size - overlap_size
        self.punc_dict = punc_dict
        self.case_confidence = case_confidence
        self.punc_str = '[' + ''.join([f'\\{x}' for x in punc_dict]) + ']'
        self.noop_index = self.vocab.get_token_index("$KEEP", "labels")

        self.case_indices = []
        for i in range(self.vocab.get_vocab_size("labels")):
            token = self.vocab.get_token_from_index(i, namespace="labels")
            if token.startswith("$TRANSFORM_CASE_"):
                self.case_indices.append(i)

        # Indices cho pause-based boost
        self.append_period_index = self.vocab.get_token_index("$APPEND_.", "labels")
        self.append_comma_index = self.vocab.get_token_index("$APPEND_,", "labels")

        # Load ONNX sessions + tokenizers
        self.indexers = []
        self.sessions = []
        for model_path in model_paths:
            logger.info(f"[GecBERT] Loading ONNX model from: {model_path}")

            # Tìm file ONNX
            onnx_path = os.path.join(model_path, "vibert-capu.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"ONNX model not found at {onnx_path}. "
                    f"Run temp/export_vibert_onnx.py to export first."
                )

            # Load ONNX session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.inter_op_num_threads = 1
            from core.config import compute_ort_threads, PHYSICAL_CORES
            sess_options.intra_op_num_threads = compute_ort_threads(PHYSICAL_CORES)
            session = ort.InferenceSession(
                onnx_path, sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.sessions.append(session)
            logger.info(f"[GecBERT] ONNX session loaded OK")

            # Load tokenizer
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "vocab.txt")):
                tokenizer_path = model_path
            else:
                # Fallback: đọc config.json để lấy pretrained_name_or_path
                import json
                config_file = os.path.join(model_path, "config.json")
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                tokenizer_path = cfg.get("pretrained_name_or_path", model_path)

            # special_tokens_fix luôn True cho vibert-capu
            logger.info(f"[GecBERT] Loading tokenizer from {tokenizer_path}")
            self.indexers.append(self._get_indexer(tokenizer_path, special_tokens_fix=True))
            logger.info(f"[GecBERT] Model loaded OK from {model_path}")

    def _get_indexer(self, weights_name, special_tokens_fix):
        tokenizer = AutoTokenizer.from_pretrained(
            weights_name, do_basic_tokenize=False, do_lower_case=self.lowercase_tokens, model_max_length=1024
        )
        # to adjust all tokenizers
        if hasattr(tokenizer, 'encoder'):
            tokenizer.vocab = tokenizer.encoder
        if hasattr(tokenizer, 'sp_model'):
            tokenizer.vocab = defaultdict(lambda: 1)
            for i in range(tokenizer.sp_model.get_piece_size()):
                tokenizer.vocab[tokenizer.sp_model.id_to_piece(i)] = i

        if special_tokens_fix:
            tokenizer.add_tokens([START_TOKEN])
            tokenizer.vocab[START_TOKEN] = len(tokenizer) - 1
        return tokenizer

    def forward(self, text: Union[str, List[str], List[List[str]]], is_split_into_words=False, progress_callback=None, pause_hints=None):
        # Giữ nguyên giao diện gọi __call__ như cũ
        def _is_valid_text_input(t):
            if isinstance(t, str):
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                elif isinstance(t[0], str):
                    return True
                elif isinstance(t[0], (list, tuple)):
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if is_split_into_words:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))
            if is_batched:
                text = [x.split() for x in text]
            else:
                text = text.split()

        if not is_batched:
            text = [text]
            if pause_hints is not None:
                pause_hints = [pause_hints]

        return self.handle_batch(text, progress_callback=progress_callback, pause_hints=pause_hints)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def split_chunks(self, batch, pause_hints=None):
        result = []
        indices = []
        hints_result = [] if pause_hints is not None else None
        for batch_idx, tokens in enumerate(batch):
            start = len(result)
            num_token = len(tokens)
            hints = pause_hints[batch_idx] if pause_hints is not None else None
            if num_token <= self.chunk_size:
                result.append(tokens)
                if hints is not None:
                    hints_result.append(hints[:num_token])
            elif num_token > self.chunk_size and num_token < (self.chunk_size * 2 - self.overlap_size):
                split_idx = (num_token + self.overlap_size + 1) // 2
                result.append(tokens[:split_idx])
                result.append(tokens[split_idx - self.overlap_size :])
                if hints is not None:
                    hints_result.append(hints[:split_idx])
                    hints_result.append(hints[split_idx - self.overlap_size :])
            else:
                for i in range(0, num_token - self.overlap_size, self.stride):
                    result.append(tokens[i : i + self.chunk_size])
                    if hints is not None:
                        hints_result.append(hints[i : i + self.chunk_size])

            indices.append((start, len(result)))

        return result, indices, hints_result

    def apply_chunk_merging(self, tokens, next_tokens):
        if not tokens:
            return next_tokens

        source_token_idx = []
        target_token_idx = []
        source_tokens = []
        target_tokens = []
        num_keep = self.overlap_size - self.min_words_cut
        i = 0
        while len(source_token_idx) < self.overlap_size and -i < len(tokens):
            i -= 1
            if tokens[i] not in self.punc_dict:
                source_token_idx.insert(0, i)
                source_tokens.insert(0, tokens[i].lower())

        i = 0
        while len(target_token_idx) < self.overlap_size and i < len(next_tokens):
            if next_tokens[i] not in self.punc_dict:
                target_token_idx.append(i)
                target_tokens.append(next_tokens[i].lower())
            i += 1

        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == "equal":
                if i1 >= num_keep:
                    tail_idx = source_token_idx[i1]
                    head_idx = target_token_idx[j1]
                    break
                elif i2 > num_keep:
                    tail_idx = source_token_idx[num_keep]
                    head_idx = target_token_idx[j2 - i2 + num_keep]
                    break
            elif tag == "delete" and i1 == 0:
                num_keep += i2 // 2

        tokens = tokens[:tail_idx] + next_tokens[head_idx:]
        return tokens

    def merge_chunks(self, batch):
        result = []
        if len(batch) == 1 or self.overlap_size == 0:
            for sub_tokens in batch:
                result.extend(sub_tokens)
        else:
            for _, sub_tokens in enumerate(batch):
                try:
                    result = self.apply_chunk_merging(result, sub_tokens)
                except Exception as e:
                    print(e)

        result = " ".join(result)
        return result

    def predict(self, batch_inputs, progress_callback=None, pause_hints_batch=None, orig_tokens_batch=None):
        """Run ONNX inference on preprocessed inputs."""
        t11 = time()
        all_logits_list = []
        all_detect_list = []

        for inputs, session in zip(batch_inputs, self.sessions):
            batch_size = inputs["input_ids"].shape[0]
            mini_batch_size = 32

            if batch_size > mini_batch_size:
                logits_parts = []
                detect_parts = []
                for i in range(0, batch_size, mini_batch_size):
                    end_idx = min(i + mini_batch_size, batch_size)
                    mini = {
                        "input_ids": inputs["input_ids"][i:end_idx],
                        "attention_mask": inputs["attention_mask"][i:end_idx],
                        "token_type_ids": inputs["token_type_ids"][i:end_idx],
                        "input_offsets": inputs["input_offsets"][i:end_idx],
                    }
                    logits, detect_logits = session.run(None, mini)
                    logits_parts.append(logits)
                    detect_parts.append(detect_logits)

                    if progress_callback is not None:
                        progress_callback(end_idx, batch_size)

                all_logits = np.concatenate(logits_parts, axis=0)
                all_detect = np.concatenate(detect_parts, axis=0)
            else:
                all_logits, all_detect = session.run(None, inputs)
                if progress_callback is not None:
                    progress_callback(batch_size, batch_size)

            all_logits_list.append(all_logits)
            all_detect_list.append(all_detect)

        preds, idx, error_probs = self._convert(
            all_logits_list, all_detect_list,
            pause_hints_batch=pause_hints_batch,
            orig_tokens_batch=orig_tokens_batch
        )
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get list of suggested actions for token."""
        if prob < self.min_error_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        # CHỈ CẤP QUYỀN: Thêm dấu câu ($APPEND_. , ? !) và Viết hoa ($TRANSFORM_CASE_)
        # CẤM QUYỀN: Thay thế chữ ($REPLACE_), Xóa chữ ($DELETE), Đính chữ ($APPEND_chữ)
        if sugg_token == '$DELETE' or sugg_token.startswith('$REPLACE_'):
            return None

        if sugg_token.startswith('$APPEND_'):
            added_text = sugg_token.replace('$APPEND_', '')
            if added_text not in self.punc_dict:
                return None
            start_pos = index + 1
            end_pos = index + 1
        elif sugg_token.startswith('$TRANSFORM_CASE_'):
            start_pos = index
            end_pos = index + 1
        else:
            return None

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1 :]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def preprocess(self, token_batch):
        """Tokenize + compute word offsets → numpy arrays cho ONNX."""
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            prefixed_batch = [[START_TOKEN] + sequence[:max_len] for sequence in token_batch]
            batch = indexer(
                prefixed_batch,
                return_tensors="np",
                padding=True,
                is_split_into_words=True,
                truncation=True,
                add_special_tokens=False,
            )
            # Build input_offsets
            offset_batch = []
            for i in range(len(prefixed_batch)):
                word_ids = batch.word_ids(batch_index=i)
                offsets = [0]
                for j in range(1, len(word_ids)):
                    if word_ids[j] != word_ids[j - 1]:
                        offsets.append(j)
                offset_batch.append(offsets)

            padded_offsets = _pad_sequences(offset_batch, padding_value=0)

            batches.append({
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
                "token_type_ids": batch["token_type_ids"].astype(np.int64),
                "input_offsets": padded_offsets,
            })

        return batches

    def _convert(self, logits_list, detect_list, pause_hints_batch=None, orig_tokens_batch=None):
        """Softmax + confidence adjustment + pause nudge → predicted label indices."""
        # Weighted average of model outputs (hỗ trợ ensemble)
        total_weight = sum(self.model_weights)
        all_class_probs = np.zeros_like(logits_list[0], dtype=np.float32)
        error_probs = np.zeros(logits_list[0].shape[:1], dtype=np.float32)

        for logits, detect_logits, weight in zip(logits_list, detect_list, self.model_weights):
            class_probs = _softmax(logits, axis=-1)
            all_class_probs += (weight / total_weight) * class_probs

            class_probs_d = _softmax(detect_logits, axis=-1)
            error_probs_d = class_probs_d[:, :, self.incorr_index]
            incorr_prob = error_probs_d.max(axis=-1)
            error_probs += (weight / total_weight) * incorr_prob

        if self.confidence != 0.0:
            all_class_probs[:, :, self.noop_index] += self.confidence

        if self.case_confidence != 0.0:
            for idx in self.case_indices:
                all_class_probs[:, :, idx] += self.case_confidence

        # Pause-based nudge: chỉ can thiệp nhẹ khi model predict $KEEP tại vị trí có pause
        if pause_hints_batch is not None:
            before_idx = all_class_probs.argmax(axis=-1)

            for b_idx, hints in enumerate(pause_hints_batch):
                if hints is None:
                    continue
                for w_idx, gap in enumerate(hints):
                    t_idx = w_idx + 1  # +1 vì START_TOKEN ở vị trí 0
                    if t_idx >= all_class_probs.shape[1]:
                        break

                    current_label_idx = int(all_class_probs[b_idx, t_idx].argmax())
                    current_label = self.vocab.get_token_from_index(current_label_idx, "labels")

                    if gap >= 1.0:
                        if current_label == "$KEEP":
                            all_class_probs[b_idx, t_idx, self.noop_index] -= 0.2
                            all_class_probs[b_idx, t_idx, self.append_period_index] += 0.2
                    elif gap >= 0.2:
                        if current_label == "$KEEP":
                            all_class_probs[b_idx, t_idx, self.append_comma_index] += 0.2

            # Log pause changes
            after_idx = all_class_probs.argmax(axis=-1)
            for b_idx, hints in enumerate(pause_hints_batch):
                if hints is None:
                    continue
                for w_idx, gap in enumerate(hints):
                    if gap < 0.2:
                        continue
                    t_idx = w_idx + 1
                    if t_idx >= all_class_probs.shape[1]:
                        break
                    old_label = self.vocab.get_token_from_index(int(before_idx[b_idx, t_idx]), "labels")
                    new_label = self.vocab.get_token_from_index(int(after_idx[b_idx, t_idx]), "labels")
                    word = ""
                    if orig_tokens_batch and b_idx < len(orig_tokens_batch) and w_idx < len(orig_tokens_batch[b_idx]):
                        word = orig_tokens_batch[b_idx][w_idx]
                    if old_label != new_label:
                        print(f"  [Pause CHANGED] \"{word}\" gap={gap:.2f}s: {old_label} -> {new_label}")
                    else:
                        print(f"  [Pause kept]    \"{word}\" gap={gap:.2f}s: {new_label} (unchanged)")

        probs = all_class_probs.max(axis=-1).tolist()
        idx = all_class_probs.argmax(axis=-1).tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch, prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs, error_probs):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch, all_probabilities, all_idxs, error_probs):
            length = min(len(tokens), self.max_len)
            edits = []

            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')
                action = self.get_token_action(token, i, probabilities[i], sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch, merge_punc=True, progress_callback=None, pause_hints=None):
        """
        Handle batch of requests.
        pause_hints: list of list of floats, mỗi phần tử là gap (giây) sau word tương ứng.
                     Dùng để boost logits thêm dấu chấm/phẩy tại vị trí có pause trong speech.
        """
        if self.split_chunk:
            full_batch, indices, pause_hints_chunks = self.split_chunks(full_batch, pause_hints=pause_hints)
        else:
            indices = None
            pause_hints_chunks = pause_hints
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch)) if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            # Pause hints chỉ dùng ở iteration đầu tiên
            if n_iter == 0 and pause_hints_chunks is not None:
                cur_pause_hints = [pause_hints_chunks[i] for i in pred_ids]
            else:
                cur_pause_hints = None

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(
                sequences, progress_callback=progress_callback,
                pause_hints_batch=cur_pause_hints,
                orig_tokens_batch=orig_batch if cur_pause_hints else None
            )

            pred_batch = self.postprocess_batch(orig_batch, probabilities, idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = self.update_final_batch(final_batch, pred_ids, pred_batch, prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        if self.split_chunk:
            final_batch = [self.merge_chunks(final_batch[start:end]) for (start, end) in indices]
        else:
            final_batch = [" ".join(x) for x in final_batch]
        if merge_punc:
            final_batch = [re.sub(r'\s+(%s)' % self.punc_str, r'\1', x) for x in final_batch]

        return final_batch

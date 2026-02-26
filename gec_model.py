"""Wrapper of Seq2Labels model. Fixes errors based on model predictions"""
from collections import defaultdict
from difflib import SequenceMatcher
import logging
import re
from time import time
from typing import List, Union
import warnings

import torch
from transformers import AutoTokenizer
from modeling_seq2labels import Seq2LabelsModel
from vocabulary import Vocabulary
from utils import PAD, UNK, START_TOKEN, get_target_sent_by_edits

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class GecBERTModel(torch.nn.Module):
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
        iterations=None,  # None = auto: 1 cho CPU, 3 cho GPU
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
        r"""
        Args:
            vocab_path (`str`):
                Path to vocabulary directory.
            model_paths (`List[str]`):
                List of model paths.
            weights (`int`, *Optional*, defaults to None):
                Weights of each model. Only relevant if `is_ensemble is True`.
            device (`int`, *Optional*, defaults to None):
                Device to load model. If not set, device will be automatically choose.
            max_len (`int`, defaults to 64):
                Max sentence length to be processed (all longer will be truncated).
            min_len (`int`, defaults to 3):
                Min sentence length to be processed (all shorted will be returned w/o changes).
            lowercase_tokens (`bool`, defaults to False):
                Whether to lowercase tokens.
            log (`bool`, defaults to False):
                Whether to enable logging.
            iterations (`int`, defaults to 3):
                Max iterations to run during inference.
            special_tokens_fix (`bool`, defaults to True):
               Whether to fix problem with [CLS], [SEP] tokens tokenization.
            min_error_probability (`float`, defaults to `0.0`):
                Minimum probability for each action to apply.
            confidence (`float`, defaults to `0.0`):
                How many probability to add to $KEEP token.
            split_chunk (`bool`, defaults to False):
                Whether to split long sentences to multiple segments of `chunk_size`.
                !Warning: if `chunk_size > max_len`, each segment will be truncate to `max_len`.
            chunk_size (`int`, defaults to 48):
                Length of each segment (in words). Only relevant if `split_chunk is True`.
            overlap_size (`int`, defaults to 12):
                Overlap size (in words) between two consecutive segments. Only relevant if `split_chunk is True`.
            min_words_cut (`int`, defaults to 6):
                Minimun number of words to be cut while merging two consecutive segments.
                Only relevant if `split_chunk is True`.
            punc_dict (List[str], defaults to `{':', ".", ",", "?"}`):
                List of punctuations.
        """
        super().__init__()
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.model_weights = list(map(float, weights)) if weights else [1] * len(model_paths)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        
        # Auto-determine iterations: Force 3 cho chất lượng tốt nhất dù trên CPU
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
        # Note: self.iterations đã được set ở trên (line 88-92)
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
                
        # set training parameters and operations

        self.indexers = []
        self.models = []
        for model_path in model_paths:
            # DEBUG: Manual Load
            try:
                # 1. Load Config
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path)
                
                # 2. Init Model
                model = Seq2LabelsModel(config)
                
                # 3. Load Weights
                import os
                bin_path = os.path.join(model_path, "pytorch_model.bin")
                if not os.path.exists(bin_path):
                     # If file doesn't exist (e.g. model_path is a repo ID), raise to trigger fallback
                     raise FileNotFoundError(f"Binary not found at {bin_path}")
                     
                print(f"Loading weights from {bin_path}")
                state_dict = torch.load(bin_path, map_location="cpu")
                
                # Load weights đúng cách bằng load_state_dict
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded weights successfully from {bin_path}")
                except Exception as e:
                    print(f"ERROR loading state_dict: {e}")
                    # Fallback: manual copy từng param nhưng vào model gốc
                    model_dict = model.state_dict()
                    for k, v in state_dict.items():
                        if k in model_dict:
                            try:
                                model_dict[k].copy_(v)
                            except Exception as copy_e:
                                print(f"ERROR copying {k}: {copy_e}")
                
                del state_dict
                import gc
                gc.collect()
                
            except Exception as e:
                if "Binary not found" in str(e):
                     print(f"Model binary not found locally at {model_path}. Downloading from Hub...")
                     # Fallback: Download snapshot and use the local path
                     try:
                         from huggingface_hub import snapshot_download
                         # If model_path is a path, this might fail, but if it's a repo_id it works.
                         # Assuming model_path is repo_id if we are here.
                         local_path = snapshot_download(repo_id=model_path)
                         print(f"Downloaded to {local_path}. Retrying manual load...")
                         
                         # Retry logic by recursive call or just copy-paste manual load? 
                         # Copy-paste is safer to avoid infinite recursion if something is wrong.
                         # 1. Load Config
                         from transformers import AutoConfig
                         config = AutoConfig.from_pretrained(local_path)
                         # 2. Init Model
                         model = Seq2LabelsModel(config)
                         # 3. Load Weights
                         bin_path = os.path.join(local_path, "pytorch_model.bin")
                         state_dict = torch.load(bin_path, map_location="cpu")
                         try:
                             model.load_state_dict(state_dict, strict=False)
                         except Exception as load_e:
                             print(f"ERROR in fallback load: {load_e}")
                             # Manual copy với dict tham chiếu đúng
                             model_dict = model.state_dict()
                             for k, v in state_dict.items():
                                 if k in model_dict:
                                     model_dict[k].copy_(v)
                         del state_dict
                         import gc
                         gc.collect()
                         
                     except Exception as download_e:
                         print(f"Download/Retry failed: {download_e}")
                         # Last resort: from_pretrained (which might crash, but we tried)
                         model = Seq2LabelsModel.from_pretrained(model_path, low_cpu_mem_usage=False)
                         config = model.config
                else:
                     print(f"Manual load failed: {e}")
                     # Fallback
                     model = Seq2LabelsModel.from_pretrained(model_path, low_cpu_mem_usage=False)
                     config = model.config
            
            model_name = config.pretrained_name_or_path
            special_tokens_fix = config.special_tokens_fix
            self.indexers.append(self._get_indexer(model_name, special_tokens_fix))
            model.eval().to(self.device)
            self.models.append(model)

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
    
    def forward(self, text: Union[str, List[str], List[List[str]]], is_split_into_words=False, progress_callback=None):
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
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
        
        return self.handle_batch(text, progress_callback=progress_callback)

    def split_chunks(self, batch):
        # return batch pairs of indices
        result = []
        indices = []
        for tokens in batch:
            start = len(result)
            num_token = len(tokens)
            if num_token <= self.chunk_size:
                result.append(tokens)
            elif num_token > self.chunk_size and num_token < (self.chunk_size * 2 - self.overlap_size):
                split_idx = (num_token + self.overlap_size + 1) // 2
                result.append(tokens[:split_idx])
                result.append(tokens[split_idx - self.overlap_size :])
            else:
                for i in range(0, num_token - self.overlap_size, self.stride):
                    result.append(tokens[i : i + self.chunk_size])

            indices.append((start, len(result)))

        return result, indices

    def check_alnum(self, s):
        if len(s) < 2:
            return False
        return not (s.isalpha() or s.isdigit())

    def apply_chunk_merging(self, tokens, next_tokens):
        # Return next tokens if current tokens list is empty
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

    def predict(self, batches, progress_callback=None):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch_size = len(batch['input_ids']) if 'input_ids' in batch else 0
            mini_batch_size = 32
            
            if batch_size > mini_batch_size:
                all_logits = []
                all_detect_logits = []
                all_max_error_probability = []
                
                for i in range(0, batch_size, mini_batch_size):
                    end_idx = min(i + mini_batch_size, batch_size)
                    mini_batch = {k: v[i:end_idx].to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    with torch.no_grad():
                        pred = model.forward(**mini_batch)
                        all_logits.append(pred['logits'].cpu())
                        all_detect_logits.append(pred['detect_logits'].cpu())
                        all_max_error_probability.append(pred['max_error_probability'].cpu())
                    
                    if progress_callback is not None:
                        progress_callback(end_idx, batch_size)
                
                prediction = {
                    'logits': torch.cat(all_logits, dim=0).to(self.device),
                    'detect_logits': torch.cat(all_detect_logits, dim=0).to(self.device),
                    'max_error_probability': torch.cat(all_max_error_probability, dim=0).to(self.device)
                }
            else:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    prediction = model.forward(**batch)
                if progress_callback is not None:
                    progress_callback(batch_size, batch_size)
                    
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_error_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None
            
        # CHỈ CẤP QUYỀN: Thêm dấu câu ($APPEND_. , ? !) và Viết hoa ($TRANSFORM_CASE_)
        # CẤM QUYỀN: Thay thế chữ ($REPLACE_), Xóa chữ ($DELETE), Đính chữ ($APPEND_chữ)
        if sugg_token == '$DELETE' or sugg_token.startswith('$REPLACE_'):
            return None
        
        if sugg_token.startswith('$APPEND_'):
            # Lấy ký tự/từ muốn đính kèm để kiểm tra
            added_text = sugg_token.replace('$APPEND_', '')
            if added_text not in self.punc_dict:
                # Nếu không phải đính kèm dấu câu hợp lệ -> Bỏ qua
                return None
            start_pos = index + 1
            end_pos = index + 1
        elif sugg_token.startswith('$TRANSFORM_CASE_'):
            start_pos = index
            end_pos = index + 1
        else:
            return None # Block everything else just to be safe

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1 :]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            token_batch = [[START_TOKEN] + sequence[:max_len] for sequence in token_batch]
            batch = indexer(
                token_batch,
                return_tensors="pt",
                padding=True,
                is_split_into_words=True,
                truncation=True,
                add_special_tokens=False,
            )
            offset_batch = []
            for i in range(len(token_batch)):
                word_ids = batch.word_ids(batch_index=i)
                offsets = [0]
                for i in range(1, len(word_ids)):
                    if word_ids[i] != word_ids[i - 1]:
                        offsets.append(i)
                offset_batch.append(torch.LongTensor(offsets))

            batch["input_offsets"] = torch.nn.utils.rnn.pad_sequence(
                offset_batch, batch_first=True, padding_value=0
            ).to(torch.long)

            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['logits'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            class_probabilities_labels = torch.softmax(output['logits'], dim=-1)
            all_class_probs += weight * class_probabilities_labels / sum(self.model_weights)
            class_probabilities_d = torch.softmax(output['detect_logits'], dim=-1)
            error_probs_d = class_probabilities_d[:, :, self.incorr_index]
            incorr_prob = torch.max(error_probs_d, dim=-1)[0]
            error_probs += weight * incorr_prob / sum(self.model_weights)

        if self.confidence != 0.0:
            all_class_probs[:, :, self.noop_index] += self.confidence
            
        if self.case_confidence != 0.0 and hasattr(self, 'case_indices'):
            for idx in self.case_indices:
                all_class_probs[:, :, idx] += self.case_confidence

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
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
                # update final batch, but stop iterations
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

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')
                action = self.get_token_action(token, i, probabilities[i], sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch, merge_punc=True, progress_callback=None):
        """
        Handle batch of requests.
        """
        try:
            from debug_utils import log_memory
            log_memory("GecBERTModel.handle_batch start")
        except ImportError:
            log_memory = lambda x: None
        
        if self.split_chunk:
            full_batch, indices = self.split_chunks(full_batch)
        else:
            indices = None
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch)) if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            try:
                log_memory(f"GecBERTModel loop iteration {n_iter} start")
            except:
                pass
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences, progress_callback=progress_callback)

            pred_batch = self.postprocess_batch(orig_batch, probabilities, idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = self.update_final_batch(final_batch, pred_ids, pred_batch, prev_preds_dict)
            total_updates += cnt
            
            try:
                log_memory(f"GecBERTModel loop iteration {n_iter} end")
            except:
                pass

            if not pred_ids:
                break
        
        if self.split_chunk:
            final_batch = [self.merge_chunks(final_batch[start:end]) for (start, end) in indices]
        else:
            final_batch = [" ".join(x) for x in final_batch]
        if merge_punc:
            final_batch = [re.sub(r'\s+(%s)' % self.punc_str, r'\1', x) for x in final_batch]
        
        try:
            log_memory("GecBERTModel.handle_batch end")
        except:
            pass
        return final_batch
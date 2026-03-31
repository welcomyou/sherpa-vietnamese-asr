# core/hotword_context.py — Aho-Corasick Context Graph cho hotword boosting
# Port 1:1 từ sherpa-onnx/csrc/context-graph.cc (k2-fsa/sherpa-onnx)
#
# Cách dùng:
#   graph = build_context_graph(hotwords_file, bpe_model_path, default_score=1.5)
#   # Trong beam search, mỗi hypothesis giữ 1 context_state:
#   state = graph.root
#   score_delta, new_state = graph.forward_one_step(state, token_id)
#   # Cuối decode: trừ partial score chưa hoàn thành
#   finalize_score = graph.finalize(state)

import os
import unicodedata
from collections import deque


class ContextState:
    """Node trong Aho-Corasick trie. Giống sherpa-onnx ContextState."""
    __slots__ = ('token', 'token_score', 'node_score', 'output_score',
                 'is_end', 'next', 'fail', 'output')

    def __init__(self, token=-1, token_score=0.0, node_score=0.0,
                 output_score=0.0, is_end=False):
        self.token = token
        self.token_score = token_score  # score CỦA edge đến node này (= phrase score, KHÔNG chia)
        self.node_score = node_score    # tổng score từ root đến node này
        self.output_score = output_score  # score output khi hoàn thành phrase (+ accumulated từ output link)
        self.is_end = is_end
        self.next = {}                  # {token_id: ContextState}
        self.fail = None                # Aho-Corasick failure link
        self.output = None              # nearest ancestor qua fail chain mà is_end


class ContextGraph:
    """Aho-Corasick automaton cho hotword boosting.

    Port 1:1 từ sherpa-onnx/csrc/context-graph.cc.
    Build, ForwardOneStep (non-strict), Finalize.
    """

    def __init__(self):
        self.root = ContextState(token=-1)
        self.root.fail = self.root
        self.n_phrases = 0

    def build(self, token_sequences, scores):
        """Build trie + Aho-Corasick failure links.

        Port từ sherpa-onnx ContextGraph::Build().

        Quan trọng: token_score = score NGUYÊN (không chia cho len),
        node_score = parent.node_score + token_score.
        Khi 2 phrase share prefix, lấy max(score) cho node chung.
        """
        for seq, score in zip(token_sequences, scores):
            if not seq:
                continue
            node = self.root

            for j, tid in enumerate(seq):
                is_last = (j == len(seq) - 1)

                if tid not in node.next:
                    # Tạo node mới
                    child = ContextState(
                        token=tid,
                        token_score=score,
                        node_score=node.node_score + score,
                        output_score=(node.node_score + score) if is_last else 0.0,
                        is_end=is_last,
                    )
                    node.next[tid] = child
                else:
                    # Node đã tồn tại (shared prefix): lấy max score
                    existing = node.next[tid]
                    existing.token_score = max(score, existing.token_score)
                    existing.node_score = node.node_score + existing.token_score
                    if is_last:
                        existing.is_end = True
                        existing.output_score = existing.node_score
                    elif existing.is_end:
                        # Giữ output_score từ phrase trước
                        existing.output_score = existing.node_score

                node = node.next[tid]

            self.n_phrases += 1

        # BFS: fill failure links + output links
        self._fill_fail_output()

    def _fill_fail_output(self):
        """Port từ sherpa-onnx ContextGraph::FillFailOutput()."""
        queue = deque()

        # Root children: fail → root
        for child in self.root.next.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()

            for tid, child in current.next.items():
                # Tìm failure link cho child
                fail = current.fail

                if tid in fail.next:
                    fail = fail.next[tid]
                else:
                    fail = fail.fail
                    while tid not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:  # root
                            break
                    if tid in fail.next:
                        fail = fail.next[tid]

                child.fail = fail

                # Output link: tìm nearest ancestor is_end qua fail chain
                output = fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:  # root
                        output = None
                        break

                child.output = output

                # Accumulate output_score qua output link
                # Giống sherpa: output_score += output.output_score
                if output is not None:
                    child.output_score += output.output_score

                queue.append(child)

    def forward_one_step(self, state, token_id):
        """Port từ sherpa-onnx ForwardOneStep(state, token, strict_mode=false).

        Non-strict mode (dùng trong beam search):
        - Khi output_score != 0 (phrase match): reset về root, trả delta score
        - Khi mismatch: follow fail links, trả delta score (có thể âm)

        Returns: (score_delta, new_state)
        """
        node = None
        score = 0.0

        if token_id in state.next:
            # Direct match
            node = state.next[token_id]
            score = node.token_score
        else:
            # Follow fail links
            node = state.fail
            while token_id not in node.next:
                node = node.fail
                if node.token == -1:  # root
                    break

            if token_id in node.next:
                node = node.next[token_id]
            # else: node = root (no match at all)

            score = node.node_score - state.node_score

        # Non-strict mode: khi output_score != 0, reset về root
        if node.output_score != 0:
            # Xác định matched node
            if node.is_end:
                output_score = node.node_score
            elif node.output is not None:
                output_score = node.output.node_score
            else:
                output_score = node.node_score

            return score + output_score - node.node_score, self.root

        return score, node

    def finalize(self, state):
        """Port từ sherpa-onnx Finalize: trừ partial score chưa hoàn thành."""
        return -state.node_score


# ══════════════════════════════════════════════════════════════
# HOTWORDS FILE PARSING + GRAPH BUILDING
# ══════════════════════════════════════════════════════════════

def parse_hotwords_file(hotwords_path, default_score=1.5):
    """Parse hotwords.txt giống sherpa-onnx utils.cc EncodeHotwords.

    Format: "phrase :score" hoặc "phrase" (dùng default_score).
    Dòng # = comment, dòng trống bỏ qua.

    Returns: list of (phrase_text_uppercase, score)
    """
    if not hotwords_path or not os.path.exists(hotwords_path):
        return []

    result = []
    with open(hotwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            score = default_score
            if ':' in line:
                parts = line.rsplit(':', 1)
                try:
                    score = float(parts[1].strip())
                    line = parts[0].strip()
                except ValueError:
                    pass

            phrase = unicodedata.normalize('NFC', line.strip().upper())
            if phrase:
                result.append((phrase, score))

    return result


def build_context_graph(hotwords_path, bpe_model_path, default_score=1.5):
    """Build ContextGraph từ hotwords file + BPE model.

    Returns: ContextGraph hoặc None nếu không có hotwords
    """
    phrases = parse_hotwords_file(hotwords_path, default_score)
    if not phrases:
        return None

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)

    token_sequences = []
    scores = []
    skipped = 0

    for phrase_text, score in phrases:
        tids = sp.encode(phrase_text, out_type=int)
        if not tids:
            skipped += 1
            continue
        token_sequences.append(tids)
        scores.append(score)

    if not token_sequences:
        return None

    graph = ContextGraph()
    graph.build(token_sequences, scores)

    print(f"[Hotwords] Built context graph: {graph.n_phrases} phrases "
          f"(skipped {skipped}), default_score={default_score}")

    return graph

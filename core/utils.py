# core/utils.py - Tiện ích xử lý văn bản, tìm kiếm fuzzy
# KHÔNG import PyQt6 - pure Python

import unicodedata
from difflib import SequenceMatcher


def normalize_vietnamese(text):
    """Chuyển text về dạng không dấu, lowercase để search không dấu"""
    if not text:
        return ""
    text = text.lower()
    # Thay thế 'đ' thủ công vì Unicode NFD không decompose 'đ' thành 'd'
    text = text.replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text


def fuzzy_score(query, text):
    """Tính độ tương đồng fuzzy giữa query và text (0.0 - 1.0)"""
    if not query or not text:
        return 0.0
    if query.lower() in text.lower():
        return 1.0
    query_norm = normalize_vietnamese(query)
    text_norm = normalize_vietnamese(text)
    if query_norm in text_norm:
        return 0.9
    return SequenceMatcher(None, query_norm, text_norm).ratio()


def find_fuzzy_matches(query, text, threshold=0.6):
    """Tìm tất cả các vị trí fuzzy match trong text"""
    matches = []
    if not query or not text:
        return matches

    query_lower = query.lower()
    text_lower = text.lower()
    query_len = len(query)

    # First: try exact match (case insensitive)
    start = 0
    while True:
        idx = text_lower.find(query_lower, start)
        if idx == -1:
            break
        matches.append((idx, idx + query_len, text[idx:idx + query_len], 1.0))
        start = idx + 1

    # Second: try normalized (no accent) match
    query_norm = normalize_vietnamese(query)
    text_norm = normalize_vietnamese(text)
    start = 0
    while True:
        idx = text_norm.find(query_norm, start)
        if idx == -1:
            break
        orig_start = idx
        orig_end = idx + query_len
        matches.append((orig_start, orig_end, text[orig_start:orig_end], 0.9))
        start = idx + 1

    # Remove duplicate positions (keep highest score)
    seen = set()
    unique_matches = []
    for start, end, matched_text, score in matches:
        key = (start, end)
        if key not in seen:
            seen.add(key)
            unique_matches.append((start, end, matched_text, score))

    return unique_matches

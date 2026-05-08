# core/asr_json.py - JSON serialize/deserialize kết quả ASR
# KHÔNG import PyQt6 - pure Python

import json
import os
from datetime import datetime


def serialize_segments(segments, speaker_name_mapping=None, speaker_colors=None,
                       model_name='unknown', model_type='file',
                       duration_sec=0.0, timing=None,
                       overlap_segments=None):
    """
    Chuyển đổi segments nội bộ thành cấu trúc JSON chuẩn.

    Args:
        segments: List[dict] - segments nội bộ với keys: text, start/start_time, speaker, speaker_id, partials, end
        speaker_name_mapping: Dict[str, str] - mapping speaker_id -> display name
        model_name: str - tên model ASR đã dùng
        model_type: str - 'file' hoặc 'online'
        duration_sec: float - thời lượng audio (giây)

    Returns:
        dict - JSON-serializable data
    """
    if speaker_name_mapping is None:
        speaker_name_mapping = {}
    if speaker_colors is None:
        speaker_colors = {}

    json_segments = []
    current_speaker = None

    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', '')
        speaker_id = seg.get('speaker_id', 0)

        # Check speaker name mapping (keys are str)
        display_name = speaker
        sid_str = str(speaker_id)
        if sid_str in speaker_name_mapping:
            display_name = speaker_name_mapping[sid_str]

        # Add speaker separator when speaker changes
        if display_name != current_speaker and display_name:
            json_segments.append({
                'type': 'speaker',
                'speaker': display_name,
                'speaker_id': int(speaker_id) if isinstance(speaker_id, (int, float)) or (isinstance(speaker_id, str) and speaker_id.isdigit()) else speaker_id,
                'start_time': seg.get('start', seg.get('start_time', 0))
            })
            current_speaker = display_name

        # Clean partials
        clean_partials = []
        for p in seg.get('partials', []):
            clean_partials.append({
                'text': p.get('text', ''),
                'timestamp': p.get('timestamp', 0)
            })

        # If no partials, create single partial
        if not clean_partials:
            seg_end = seg.get('end', seg.get('start', 0) + 1.0)
            clean_partials.append({
                'text': seg.get('text', ''),
                'timestamp': seg_end
            })

        # Serialize raw_words (suspect highlighting)
        raw_words_data = None
        if seg.get('raw_words'):
            raw_words_data = []
            for w in seg['raw_words']:
                wd = {'text': w.get('text', '')}
                if w.get('gap_after_ms'):
                    wd['gap_after_ms'] = w['gap_after_ms']
                if w.get('gap_before_ms'):
                    wd['gap_before_ms'] = w['gap_before_ms']
                if w.get('_suspect_level'):
                    wd['suspect'] = w['_suspect_level']
                raw_words_data.append(wd)

        text_seg = {
            'type': 'text',
            'text': seg.get('text', ''),
            'start_time': seg.get('start', seg.get('start_time', 0)),
            'segment_id': i,
            'partials': clean_partials
        }
        if raw_words_data:
            text_seg['raw_words'] = raw_words_data

        json_segments.append(text_seg)

    json_data = {
        'version': 1,
        'model': model_name,
        'model_type': model_type,
        'created_at': datetime.now().isoformat(),
        'duration_sec': round(duration_sec, 2),
        'timing': timing or {},
        'speaker_names': dict(speaker_name_mapping) if speaker_name_mapping else {},
        'speaker_colors': dict(speaker_colors) if speaker_colors else {},
        'segments': json_segments
    }

    # Optional: overlap segments (parallel entries cho vùng 2-speaker overlap).
    # Additive field — reader cũ không biết về overlap_segments sẽ bỏ qua, vẫn
    # đọc segments như bình thường.
    if overlap_segments:
        ov_out = []
        for ov in overlap_segments:
            spk_id = ov.get('speaker_id', 0)
            sid_str = str(spk_id)
            display_name = ov.get('speaker', f"Người nói {spk_id + 1}")
            if speaker_name_mapping and sid_str in speaker_name_mapping:
                display_name = speaker_name_mapping[sid_str]
            ov_entry = {
                'speaker': display_name,
                'speaker_id': int(spk_id) if isinstance(spk_id, (int, float)) else spk_id,
                'start_time': round(float(ov.get('start', 0)), 3),
                'end_time': round(float(ov.get('end', 0)), 3),
                'text': ov.get('text', ''),
            }
            if ov.get('raw_words'):
                rw_out = []
                for w in ov['raw_words']:
                    rw_out.append({
                        'text': w.get('word') or w.get('text') or '',
                        'start': round(float(w.get('start', 0)), 3),
                        'end': round(float(w.get('end', 0)), 3),
                    })
                ov_entry['raw_words'] = rw_out
            ov_out.append(ov_entry)
        json_data['overlap_segments'] = ov_out

    return json_data


def deserialize_segments(data):
    """
    Chuyển đổi JSON data thành segments nội bộ.

    Args:
        data: dict - JSON data đã parse

    Returns:
        tuple: (segments, speaker_mapping, has_speakers)
            - segments: List[dict] - segments nội bộ
            - speaker_mapping: Dict[str, str] - speaker_id -> name
            - has_speakers: bool - có speaker diarization hay không
    """
    if 'segments' not in data:
        raise ValueError("Invalid JSON: no 'segments' key")

    json_segments = data['segments']
    speaker_mapping = data.get('speaker_names', {})
    speaker_colors = data.get('speaker_colors', {})

    segments = []
    current_speaker = ''
    current_speaker_id = 0
    has_speakers = False
    seg_counter = 0

    for seg in json_segments:
        seg_type = seg.get('type', 'text')

        if seg_type == 'speaker':
            current_speaker = seg.get('speaker', '')
            raw_id = seg.get('speaker_id', 0)
            try:
                current_speaker_id = int(raw_id)
            except (ValueError, TypeError):
                current_speaker_id = raw_id
            has_speakers = True
            continue

        if seg_type == 'text':
            original_text = seg.get('text', '')
            partials = seg.get('partials', [])
            partials = [p for p in partials if p.get('text', '').strip()]

            if not partials and original_text:
                partials = [{'text': original_text}]

            internal_seg = {
                'text': original_text,
                'start': seg.get('start_time', 0),
                'start_time': seg.get('start_time', 0),
                'index': seg_counter,
                'speaker': current_speaker,
                'speaker_id': current_speaker_id,
            }

            if partials:
                internal_seg['partials'] = partials
                internal_seg['end'] = partials[-1].get('timestamp', internal_seg['start'] + 1.0)
            else:
                internal_seg['end'] = internal_seg['start'] + 1.0
                internal_seg['partials'] = [{
                    'text': internal_seg['text'],
                    'timestamp': internal_seg['end']
                }]

            segments.append(internal_seg)
            seg_counter += 1

    return segments, speaker_mapping, speaker_colors, has_speakers


def deserialize_overlap_segments(data):
    """Đọc overlap_segments từ JSON data (trả [] nếu không có).

    Mỗi entry có: speaker, speaker_id, start_time, end_time, text, raw_words.
    """
    ov = data.get('overlap_segments') or []
    out = []
    for o in ov:
        try:
            out.append({
                'speaker': o.get('speaker', ''),
                'speaker_id': int(o.get('speaker_id', 0)),
                'start': float(o.get('start_time', 0)),
                'end': float(o.get('end_time', 0)),
                'text': o.get('text', ''),
                'raw_words': o.get('raw_words', []),
                'overlap': True,
            })
        except (ValueError, TypeError):
            continue
    return out


def load_asr_json(json_path):
    """Đọc file JSON và parse thành data dict."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_asr_json(json_path, data):
    """Ghi data dict ra file JSON."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

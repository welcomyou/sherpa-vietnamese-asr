# core/asr_json.py - JSON serialize/deserialize kết quả ASR
# KHÔNG import PyQt6 - pure Python

import json
import os
from datetime import datetime


def serialize_segments(segments, speaker_name_mapping=None, model_name='unknown',
                       model_type='file', duration_sec=0.0, timing=None):
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

        json_segments.append({
            'type': 'text',
            'text': seg.get('text', ''),
            'start_time': seg.get('start', seg.get('start_time', 0)),
            'segment_id': i,
            'partials': clean_partials
        })

    json_data = {
        'version': 1,
        'model': model_name,
        'model_type': model_type,
        'created_at': datetime.now().isoformat(),
        'duration_sec': round(duration_sec, 2),
        'timing': timing or {},
        'speaker_names': dict(speaker_name_mapping) if speaker_name_mapping else {},
        'segments': json_segments
    }

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

    return segments, speaker_mapping, has_speakers


def load_asr_json(json_path):
    """Đọc file JSON và parse thành data dict."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_asr_json(json_path, data):
    """Ghi data dict ra file JSON."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

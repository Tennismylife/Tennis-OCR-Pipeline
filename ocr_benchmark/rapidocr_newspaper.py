#!/usr/bin/env python3
from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion

PROFILES = {
    'STANDARD': {'Det.limit_side_len': 1536, 'Det.limit_type': 'min', 'Det.box_thresh': 0.35, 'Det.max_candidates': 3000},
    'HQ': {'Det.limit_side_len': 2048, 'Det.limit_type': 'min', 'Det.box_thresh': 0.30, 'Det.max_candidates': 4000},
    'TILED': {'Det.limit_side_len': 736, 'Det.limit_type': 'min', 'Det.box_thresh': 0.50, 'Det.max_candidates': 1000},
}
BASE = {
    'Global.log_level': 'error',
    'EngineConfig.onnxruntime.intra_op_num_threads': 1,
    'EngineConfig.onnxruntime.inter_op_num_threads': 1,
    'Rec.lang_type': LangRec.LATIN,
    'Rec.model_type': ModelType.MOBILE,
    'Rec.ocr_version': OCRVersion.PPOCRV5,
    'Det.model_type': ModelType.SMALL,
    'Det.ocr_version': OCRVersion.PPOCRV6,
}

def build_engine(profile='STANDARD'):
    profile = profile.upper()
    if profile not in PROFILES:
        raise ValueError(f'Unknown newspaper OCR profile: {profile}')
    params = dict(BASE)
    params.update(PROFILES[profile])
    return RapidOCR(params=params)

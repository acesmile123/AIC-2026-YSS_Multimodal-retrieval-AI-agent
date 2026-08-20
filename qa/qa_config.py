import os

try:
    import torch
except ImportError:
    torch = None


def _env_int(name, default, min_value=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if min_value is not None and value < min_value:
        return default if default >= min_value else min_value
    return value


def _env_float(name, default, min_value=None):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if min_value is not None and value < min_value:
        return default if default >= min_value else min_value
    return value


def _env_bool(name, default):
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = os.environ.get("QA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = os.environ.get("QA_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
KEYFRAMES_DIR = os.environ.get("QA_KEYFRAMES_DIR", DATA_DIR)
VLM_MODEL_ID = os.environ.get("QA_VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

TEMPORAL_WINDOW_SIZE = _env_int("QA_TEMPORAL_WINDOW_SIZE", 1, 0)
FRAME_STEP = _env_int("QA_FRAME_STEP", 5, 1)
MOTION_TEMPORAL_WINDOW_SIZE = _env_int("QA_MOTION_TEMPORAL_WINDOW_SIZE", 2, 0)
MOTION_FRAME_STEP = _env_int("QA_MOTION_FRAME_STEP", 3, 1)

FRAME_CACHE_SIZE = _env_int("QA_FRAME_CACHE_SIZE", 128, 1)
RAW_FRAME_CACHE_SIZE = _env_int("QA_RAW_FRAME_CACHE_SIZE", 256, 1)
ANSWER_CACHE_SIZE = _env_int("QA_ANSWER_CACHE_SIZE", 256, 1)
QA_FRAME_PREFETCH_WORKERS = _env_int("QA_FRAME_PREFETCH_WORKERS", 4, 1)

# Retrieval/VLM budgets are intentionally separate from submission K.
# This prevents a top-100 submission request from triggering 100 VLM calls.
QA_RETRIEVAL_POOL_K = _env_int("QA_RETRIEVAL_POOL_K", 50, 1)
QA_VLM_CANDIDATE_BUDGET = _env_int("QA_VLM_CANDIDATE_BUDGET", 10, 1)

VLM_MAX_NEW_TOKENS = _env_int("QA_VLM_MAX_NEW_TOKENS", 32, 1)
VLM_DO_SAMPLE = _env_bool("QA_VLM_DO_SAMPLE", False)
VLM_TEMPERATURE = _env_float("QA_VLM_TEMPERATURE", 0.0, 0.0)
VLM_MAX_INPUT_FRAMES = _env_int("QA_VLM_MAX_INPUT_FRAMES", 3, 1)
VLM_MIN_PIXELS = _env_int("QA_VLM_MIN_PIXELS", 224 * 28 * 28, 28 * 28)
VLM_MAX_PIXELS = _env_int("QA_VLM_MAX_PIXELS", 512 * 28 * 28, 28 * 28)
VLM_PROFILE = os.environ.get("QA_VLM_PROFILE", "rtx4060_8gb")
VLM_MAX_RETRIES = _env_int("QA_VLM_MAX_RETRIES", 2, 0)
VLM_UNKNOWN_TOKEN = "UNKNOWN"
VLM_MAX_ANSWER_WORDS = _env_int("QA_VLM_MAX_ANSWER_WORDS", 12, 1)

ANSWER_AGGREGATION_ENABLED = _env_bool("QA_ANSWER_AGGREGATION_ENABLED", True)
ANSWER_SIMILARITY_THRESHOLD = _env_float("QA_ANSWER_SIMILARITY_THRESHOLD", 0.6, 0.0)

QA_KIS_FUSION_WEIGHT = _env_float("QA_KIS_FUSION_WEIGHT", 1.3, 0.0)
QA_QUESTION_CLIP_WEIGHT = _env_float("QA_QUESTION_CLIP_WEIGHT", 1.0, 0.0)
QA_QUESTION_CAPTION_WEIGHT = _env_float("QA_QUESTION_CAPTION_WEIGHT", 0.7, 0.0)

RANK_RETRIEVAL_WEIGHT = _env_float("QA_RANK_RETRIEVAL_WEIGHT", 0.40, 0.0)
RANK_CONSENSUS_WEIGHT = _env_float("QA_RANK_CONSENSUS_WEIGHT", 0.20, 0.0)
RANK_CONFIDENCE_WEIGHT = _env_float("QA_RANK_CONFIDENCE_WEIGHT", 0.18, 0.0)
RANK_DIVERSIFY_BY_VIDEO = _env_bool("QA_RANK_DIVERSIFY_BY_VIDEO", True)
RANK_VIDEO_REPETITION_DECAY = _env_float("QA_RANK_VIDEO_REPETITION_DECAY", 0.85, 0.0)
MAX_SUBMISSION_CANDIDATES = _env_int("QA_MAX_SUBMISSION_CANDIDATES", 100, 1)
LOG_LEVEL = os.environ.get("QA_LOG_LEVEL", "INFO")


def resolve_vlm_dtype(device: str):
    if torch is None:
        raise RuntimeError("PyTorch is required only when loading a VLM")
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    override = os.environ.get("QA_VLM_DTYPE_OVERRIDE", "").lower().strip()
    if override == "float16":
        return torch.float16
    if override == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


VLM_DTYPE = resolve_vlm_dtype("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch else None

# Advanced evidence/reasoning controls
# Semantic parsing is a fallback, not the primary path. It stays enabled
# by default for ambiguous/compound questions.
QUESTION_SEMANTIC_ENABLED = _env_bool("QA_QUESTION_SEMANTIC_ENABLED", True)
OBJECT_GROUNDING_ENABLED = _env_bool("QA_OBJECT_GROUNDING_ENABLED", True)
COUNTING_ENABLED = _env_bool("QA_COUNTING_ENABLED", True)
VISUAL_RERANKING_ENABLED = _env_bool("QA_VISUAL_RERANKING_ENABLED", True)
ANSWER_VALIDATION_ENABLED = _env_bool("QA_ANSWER_VALIDATION_ENABLED", True)
EVIDENCE_MEMORY_SIZE = _env_int("QA_EVIDENCE_MEMORY_SIZE", 512, 1)
DEDUP_IOU_THRESHOLD = _env_float("QA_DEDUP_IOU_THRESHOLD", 0.50, 0.0)
TEMPORAL_REASONING_WINDOW_SIZE = _env_int("QA_TEMPORAL_REASONING_WINDOW_SIZE", 4, 1)
TEMPORAL_REASONING_STEP = _env_int("QA_TEMPORAL_REASONING_STEP", 3, 1)
VLM_EVIDENCE_FRAME_LIMIT = _env_int("QA_VLM_EVIDENCE_FRAME_LIMIT", 5, 1)
RANK_VISUAL_WEIGHT = _env_float("QA_RANK_VISUAL_WEIGHT", 0.07, 0.0)
RANK_VALIDATION_WEIGHT = _env_float("QA_RANK_VALIDATION_WEIGHT", 0.08, 0.0)
RANK_EVIDENCE_WEIGHT = _env_float("QA_RANK_EVIDENCE_WEIGHT", 0.07, 0.0)

# Advanced frame/evidence settings
ADAPTIVE_FRAME_SELECTION_ENABLED = _env_bool("QA_ADAPTIVE_FRAME_SELECTION_ENABLED", True)
ENABLE_OCR = _env_bool("QA_ENABLE_OCR", True)
# Bilingual by default: on-screen text (captions, signs, subtitles) may be
# Vietnamese or English regardless of which language the question was asked
# in. TesseractOCR gracefully drops to whichever packs are actually installed.
OCR_LANG = os.environ.get("QA_OCR_LANG", "vie+eng")
# How many sampled frames in the temporal window to run OCR over before
# keeping the highest-confidence read. Bounded low: OCR is comparatively slow
# and this only needs to beat "first frame with any text" for reliability.
OCR_MAX_FRAMES = _env_int("QA_OCR_MAX_FRAMES", 3, 1)
OCR_MIN_CONFIDENCE = _env_float("QA_OCR_MIN_CONFIDENCE", 0.30, 0.0)
TRACK_IOU_THRESHOLD = _env_float("QA_TRACK_IOU_THRESHOLD", 0.25, 0.0)
COUNT_MIN_AGREEMENT = _env_float("QA_COUNT_MIN_AGREEMENT", 0.67, 0.0)
COUNT_DETERMINISTIC_CONFIDENCE = _env_float("QA_COUNT_DETERMINISTIC_CONFIDENCE", 0.90, 0.0)
COUNT_RELATION_MIN_CONFIDENCE = _env_float("QA_COUNT_RELATION_MIN_CONFIDENCE", 0.60, 0.0)

VLM_VERIFY_ENABLED = _env_bool("QA_VLM_VERIFY_ENABLED", True)

# Reliability controls
EVIDENCE_SUFFICIENCY_MIN = _env_float("QA_EVIDENCE_SUFFICIENCY_MIN", 0.45, 0.0)
RANK_DIVERSITY_MIN_VIDEO = _env_int("QA_RANK_DIVERSITY_MIN_VIDEO", 2, 1)

# BTC object metadata lives on the frame vector records. KIS is not modified;
# QA reads the metadata only after KIS returns candidate frame IDs.
OBJECT_METADATA_VECTOR_ENABLED = _env_bool("QA_OBJECT_METADATA_VECTOR_ENABLED", True)
OBJECT_METADATA_MILVUS_URI = os.environ.get("QA_MILVUS_URI", "http://localhost:19530")
OBJECT_METADATA_COLLECTION = os.environ.get("QA_MILVUS_COLLECTION", "clip_keyframes_qa_enriched")

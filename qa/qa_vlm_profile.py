from dataclasses import dataclass
import os

@dataclass(frozen=True)
class VLMProfile:
    name: str
    model_id: str
    backend: str
    max_input_frames: int
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    dtype: str
    device_map: str

# The default profile remains 8GB-safe while allowing two temporal evidence frames.
# Qwen2-VL-2B-Instruct is smaller than Qwen2.5-VL-3B and its published HF
# repository is about 4.43 GB of weights before runtime activations.
# See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
PROFILES = {
    "rtx4060_8gb": VLMProfile(
        "rtx4060_8gb",
        "Qwen/Qwen2-VL-2B-Instruct",
        "native",
        3,
        224 * 28 * 28,
        512 * 28 * 28,
        32,
        "float16",
        "auto",
    ),
    "balanced": VLMProfile(
        "balanced",
        "Qwen/Qwen2-VL-2B-Instruct",
        "native",
        3,
        224 * 28 * 28,
        672 * 28 * 28,
        40,
        "float16",
        "auto",
    ),
    "high_gpu": VLMProfile(
        "high_gpu",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "bitsandbytes_4bit",
        3,
        256 * 28 * 28,
        1024 * 28 * 28,
        48,
        "float16",
        "auto",
    ),
}

def get_profile() -> VLMProfile:
    name = os.environ.get("QA_VLM_PROFILE", "rtx4060_8gb").strip().lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown QA_VLM_PROFILE={name!r}. Available: {sorted(PROFILES)}")
    base = PROFILES[name]
    return VLMProfile(
        name=name,
        model_id=os.environ.get("QA_VLM_MODEL_ID", base.model_id),
        backend=os.environ.get("QA_VLM_BACKEND", base.backend),
        max_input_frames=max(1, int(os.environ.get("QA_VLM_MAX_INPUT_FRAMES", base.max_input_frames))),
        min_pixels=max(28 * 28, int(os.environ.get("QA_VLM_MIN_PIXELS", base.min_pixels))),
        max_pixels=max(base.min_pixels, int(os.environ.get("QA_VLM_MAX_PIXELS", base.max_pixels))),
        max_new_tokens=max(1, int(os.environ.get("QA_VLM_MAX_NEW_TOKENS", base.max_new_tokens))),
        dtype=os.environ.get("QA_VLM_PROFILE_DTYPE", base.dtype),
        device_map=os.environ.get("QA_VLM_DEVICE_MAP", base.device_map),
    )

"""Opt-in real VLM smoke test for the RTX 4060 QA runtime.

Run:
    python -m qa.vlm_runtime_smoke

This is intentionally the only QA test kept in the project. It checks that the
configured VLM can load and execute one real end-to-end query. It does not use
mock answers or a deterministic fake model.
"""
from __future__ import annotations

import importlib.util
import os
import time


def _missing(names):
    return [name for name in names if importlib.util.find_spec(name) is None]


def main() -> int:
    os.environ.setdefault("QA_VLM_PROFILE", "rtx4060_8gb")
    os.environ.setdefault("QA_VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

    required = ["torch", "transformers", "accelerate", "qwen_vl_utils"]
    missing = _missing(required)
    if missing:
        raise RuntimeError("Missing VLM packages: " + ", ".join(missing))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this smoke test on the RTX 4060 host.")

    from .pipeline import QASystem
    from .qa_vlm_profile import get_profile

    profile = get_profile()
    print("VLM profile:", profile)
    print("GPU:", torch.cuda.get_device_name(0))

    qa = QASystem.from_shared_kis()
    query = "What is visible in the video?"
    t0 = time.perf_counter()
    result = qa.query(query, top_k=1)
    elapsed = time.perf_counter() - t0

    if not result:
        raise RuntimeError("Real VLM query returned no result.")
    row = result[0]
    required = {"video_id", "frame_id", "answer"}
    if not required.issubset(row):
        raise RuntimeError(f"Invalid QA output: {row}")
    if not str(row["answer"]).strip():
        raise RuntimeError(f"Empty answer: {row}")

    print("PASS: real VLM query executed")
    print(f"video_id = {row['video_id']}")
    print(f"frame_id = {row['frame_id']}")
    print(f"answer = {row['answer']!r}")
    print(f"latency = {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

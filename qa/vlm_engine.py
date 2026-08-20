import abc
import json
import re
from typing import List, Union
import torch
from PIL import Image
from . import qa_config
from .qa_answer_normalizer import parse_vlm_output
from .qa_logging import get_logger
from .qa_vlm_profile import get_profile
logger=get_logger("vlm_engine")

def _is_oom_error(err: Exception)->bool:
    return isinstance(err,getattr(torch.cuda,"OutOfMemoryError",())) or "out of memory" in str(err).lower()

class BaseVLM(abc.ABC):
    @abc.abstractmethod
    def generate_answer(self, frames:Union[Image.Image,List[Image.Image]], question:str)->"tuple[str,float]":
        """Injected VLMs only need to support the 2-arg form. Implementations
        MAY additionally accept an optional `evidence_context: str = ""`
        keyword for richer prompting (see QwenVLEngine) - pipeline.py probes
        for this via inspect.signature before passing it, so 2-arg-only
        implementations keep working unmodified."""
        raise NotImplementedError

def _load_transformers_classes():
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("Transformers is required for the real VLM runtime.") from exc
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
    return AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration


def _is_qwen2_vl(model_id: str) -> bool:
    mid = str(model_id).lower()
    return "qwen2-vl" in mid and "qwen2.5-vl" not in mid


class QwenVLEngine(BaseVLM):
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.profile = get_profile()
        self.model_id = model_id or self.profile.model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.dtype = torch.float16 if self.device == "cuda" and self.profile.dtype == "float16" else (torch.bfloat16 if self.device == "cuda" else torch.float32)

        kwargs = {"low_cpu_mem_usage": True}
        self.quantization_config = None
        if self.device == "cuda" and self.profile.backend == "bitsandbytes_4bit":
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "BitsAndBytes 4-bit backend requires a recent transformers build. "
                    "Install transformers and bitsandbytes in the QA environment."
                ) from exc
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs.update({
                "quantization_config": self.quantization_config,
                "device_map": self.profile.device_map,
            })
        elif self.device == "cuda":
            kwargs.update({
                "device_map": self.profile.device_map,
                "torch_dtype": self.dtype,
            })
        else:
            kwargs.update({"torch_dtype": torch.float32})

        logger.info(
            "Booting %s profile=%s backend=%s device=%s max_frames=%d max_pixels=%d",
            self.model_id, self.profile.name, self.profile.backend, self.device,
            self.profile.max_input_frames, self.profile.max_pixels,
        )

        try:
            AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration = _load_transformers_classes()
            model_cls = Qwen2VLForConditionalGeneration if _is_qwen2_vl(self.model_id) else Qwen2_5_VLForConditionalGeneration
            if model_cls is None:
                family = "Qwen2-VL" if _is_qwen2_vl(self.model_id) else "Qwen2.5-VL"
                raise RuntimeError(f"Installed transformers does not provide the {family} model class.")
            self.model = model_cls.from_pretrained(self.model_id, **kwargs)
            if self.device == "cpu":
                self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=self.profile.min_pixels,
                max_pixels=self.profile.max_pixels,
            )
            self.model.eval()
        except Exception as exc:
            backend_note = (
                " The 8GB-safe default is Qwen/Qwen2-VL-2B-Instruct with native FP16; "
                "Qwen2.5-VL-3B 4-bit remains available only as the high_gpu profile."
                if self.profile.name == "rtx4060_8gb" else ""
            )
            raise RuntimeError(
                f"Failed to load VLM model {self.model_id!r} using backend "
                f"{self.profile.backend!r}.{backend_note} Original error: {exc}"
            ) from exc

    def _select_frames(self, frames):
        limit = max(1, self.profile.max_input_frames)
        if len(frames) <= limit:
            return frames
        if limit == 1:
            return [frames[len(frames) // 2]]
        idx = torch.linspace(0, len(frames) - 1, steps=limit).round().long().tolist()
        return [frames[i] for i in idx]

    def _build_prompt(self, n, q, evidence_context: str = ""):
        note = f"You are given {n} frames from a video, in temporal order. " if n > 1 else ""
        evidence_block = (
            f"\n\nSupporting evidence from automated analysis (may be incomplete or "
            f"wrong - treat as hints only, the frame(s) are the ground truth):\n{evidence_context}\n"
            if evidence_context else ""
        )
        return (
            "You are a visual question answering system for an OPEN-DOMAIN video retrieval "
            "competition. The question may be about any visible or inferable entity: people, "
            "animals, vehicles, machines, objects, text/signs, actions, spatial relationships, "
            "or events - never assume the topic in advance. "
            "Answer only from the provided frame(s) and evidence below, not from unstated "
            "assumptions. Treat the structured question constraints as HARD requirements: "
            "never substitute a global object count for a filtered count, never confuse the "
            "reference object with the target, and respect before/after/next temporal wording. "
            "For spatial phrases such as in front of, behind, left/right of, near, or next to, "
            "first identify the target entity and any reference entity used only to locate it. "
            "A reference object's attributes are never the answer unless the question explicitly asks about the reference. "
            "For role/action questions, count or select only people visibly satisfying the role/action when evidence supports it. "
            "If the evidence is insufficient, use the best-supported answer but mark CONFIDENCE=LOW. "
            "If the question has multiple parts, satisfy all constraints before answering. "
            "For counting questions, return the number and nothing else. "
            f"{note}{evidence_block}"
            f"Return exactly:\nANSWER: <short answer>\nCONFIDENCE: <HIGH|MEDIUM|LOW>\n\nQuestion: {q}"
        )

    def _run_generation(self, frames, q, evidence_context: str = ""):
        from qwen_vl_utils import process_vision_info
        frames = self._select_frames(frames)
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": im} for im in frames],
            {"type": "text", "text": self._build_prompt(len(frames), q, evidence_context)},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.profile.max_new_tokens,
                do_sample=False,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def parse_question_semantic(self, question: str) -> dict:
        """Semantic fallback for ambiguous/unseen questions.

        This is text-only: no image is needed. It asks the same Qwen-VL model
        to normalize the question into an executable semantic program. This
        path is used only when the fast keyword classifier cannot safely express
        the intent, so easy questions do not pay this extra inference cost.
        """
        prompt = (
            "You are an English-only video-question semantic parser. Convert the English question "
            "into a compact JSON program. Do not answer the question and do not invent visual facts. "
            "Resolve paraphrases by meaning, not by exact keywords. Return ONLY valid JSON with this schema: "
            "{\"primary_type\": string, \"types\": [string], \"operation\": string, "
            "\"expected_answer_type\": string, \"target\": string, \"target_terms\": [string], "
            "\"relations\": [{\"type\": string, \"subject\": string, \"reference\": string}], "
            "\"attributes\": [{\"entity\": string, \"name\": string, \"value\": string}], "
            "\"count_target\": string, \"count_constraints\": [object], \"reference_entities\": [object], "
            "\"temporal_constraints\": [object], \"needs_temporal\": boolean, \"temporal_direction\": string, \"confidence\": number}. "
            "Use types only from [COUNTING, OBJECT, ATTRIBUTE, ACTION, TEMPORAL, SPATIAL, RELATIONSHIP, "
            "COMPARISON, YES_NO, OCR, GENERAL]. operation examples: count, find, describe_attribute, "
            "describe_action, compare, verify, locate, temporal_next, temporal_before, answer. "
            "Important: phrases such as 'in front of the podium', 'near the red car', 'the second person from the left', "
            "or any paraphrase must be represented as relations/attributes instead of hard-coded keywords. "
            "For 'what color is TARGET in the frame with REFERENCE', set target=TARGET and reference_entities=[REFERENCE] and never put the reference color on the target. "
            f"\nQuestion: {question}\nJSON:"
        )
        try:
            raw = self._run_text_generation(prompt)
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"semantic parser returned non-JSON: {raw[:200]}")
            data = json.loads(raw[start:end+1])
            if not isinstance(data, dict):
                raise ValueError("semantic parser output must be an object")
            return data
        except Exception as exc:
            logger.warning("Semantic question parsing failed: %s", exc)
            return {}

    def _run_text_generation(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def gpu_stats(self):
        if self.device != "cuda":
            return {"device": self.device}
        p = torch.cuda.get_device_properties(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "total_gb": p.total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
            "model_id": self.model_id,
            "profile": self.profile.name,
            "backend": self.profile.backend,
        }

    def generate_answer(self, frames, q, evidence_context: str = ""):
        if not isinstance(frames, list):
            frames = [frames] if frames is not None else []
        if not frames:
            return "", 0.0
        working = self._select_frames(list(frames))
        retries = 0
        while True:
            try:
                raw = self._run_generation(working, q, evidence_context)
                if not raw:
                    return "", 0.0
                return parse_vlm_output(
                    raw,
                    unknown_token=qa_config.VLM_UNKNOWN_TOKEN,
                    max_words=qa_config.VLM_MAX_ANSWER_WORDS,
                )
            except Exception as exc:
                if self.device == "cuda" and _is_oom_error(exc):
                    torch.cuda.empty_cache()
                    if len(working) > 1:
                        working = [working[len(working) // 2]] if len(working) <= 2 else working[:max(1, len(working) // 2)]
                        logger.warning("CUDA OOM; reducing evidence to %d frame(s)", len(working))
                        continue
                    logger.error("VLM OOM even with one frame: %s", exc)
                    return "", 0.0
                retries += 1
                if retries > qa_config.VLM_MAX_RETRIES:
                    raise RuntimeError(
                        f"VLM generation failed after {retries} attempts: {exc}"
                    ) from exc

    def verify_answer(self, frames, question: str, answer: str, evidence_context: str = "") -> float:
        """Ask the VLM whether the given answer is actually supported by the
        frame(s), independent of how it was produced (VLM guess or
        specialized evidence like the counting engine). Returns a 0..1
        support score. This is what makes `QA_VLM_VERIFY_ENABLED` actually
        do something - previously `pipeline._answer_one_candidate` called
        this method but it did not exist on QwenVLEngine, so the call always
        raised AttributeError and was silently swallowed (dead-code flag)."""
        if not isinstance(frames, list):
            frames = [frames] if frames is not None else []
        if not frames or not answer:
            return 0.0
        prompt = (
            "You are verifying a candidate answer to a visual question. "
            "Look only at the provided frame(s). "
            f"Question: {question}\nCandidate answer: {answer}\n"
            "Check every constraint in the question, not just whether the answer is plausible. "
            "For counting questions, verify the COUNTED SET is correct (target + spatial/role/action/time constraints), "
            "not merely the total number of people or objects. Reply with exactly one line:\nVERDICT: <YES|PARTIAL|NO>"
        )
        try:
            from qwen_vl_utils import process_vision_info
            working = self._select_frames(list(frames))
            messages = [{"role": "user", "content": [
                *[{"type": "image", "image": im} for im in working],
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
            raw = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().upper()
            if "YES" in raw:
                return 1.0
            if "PARTIAL" in raw:
                return 0.5
            return 0.0
        except Exception as exc:
            if self.device == "cuda" and _is_oom_error(exc):
                torch.cuda.empty_cache()
            logger.warning("verify_answer failed, treating as unverified: %s", exc)
            return 0.5


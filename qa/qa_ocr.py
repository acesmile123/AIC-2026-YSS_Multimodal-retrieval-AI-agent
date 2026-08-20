from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR output.

    ``confidence`` is normalized to [0, 1]. ``0.0`` means "no usable text was
    recognized", not "recognition failed with certainty zero" -- callers that
    need to distinguish "OCR unavailable" from "OCR ran, found nothing" should
    consult ``available()`` / the evidence layer's ``ocr_available`` flag
    instead of this value.
    """

    text: str = ""
    confidence: float = 0.0
    variant: str = ""
    word_count: int = 0


class OCRProvider:
    """Small OCR interface used by the evidence builder."""

    def available(self) -> bool:
        return True

    def extract(self, image: Any) -> str:
        return ""

    def extract_detailed(self, image: Any) -> OCRResult:
        return OCRResult(text=self.extract(image))


class TesseractOCR(OCRProvider):
    """Tesseract backend with preprocessing, bilingual support, and confidence
    scoring, and graceful availability detection.

    OCR must never make the QA pipeline crash when the executable, the optional
    Python package, or a requested language pack is missing. In that case
    ``available()`` is False and the evidence layer simply records that OCR was
    unavailable.

    The frames in this pipeline are video keyframes, not scanned documents: text
    is often small, low-contrast overlay/subtitle/sign text rather than a dense
    printed page. A single default Tesseract call on the raw frame handles that
    case poorly, so this backend:

    - Upscales small frames and boosts contrast before recognition, since
      Tesseract's accuracy degrades sharply below ~300 DPI-equivalent text size.
    - Tries more than one Tesseract page-segmentation mode (a block of text vs.
      sparse scattered text) and keeps whichever run scores higher confidence.
    - Recognizes with combined Vietnamese+English models by default, matching
      the bilingual VN/EN nature of the source queries/content, and falls back
      to whichever language packs are actually installed rather than failing.
    - Reports a genuine mean-confidence score (from Tesseract's per-word data,
      not merely "did I get a string back") and drops low-confidence word noise
      before returning text.
    """

    # Uniform block of text (subtitles/captions), then sparse text with no
    # particular layout (signs, labels, scattered on-screen text).
    _DEFAULT_PSM_MODES: Sequence[int] = (6, 11)
    _MIN_WORD_CONFIDENCE = 35.0  # Tesseract's 0-100 scale; below this is noise.
    _UPSCALE_TARGET_MIN_DIM = 800

    def __init__(self, lang: str = "vie+eng", psm_modes: Sequence[int] = None):
        self.lang = lang
        self.psm_modes = tuple(psm_modes) if psm_modes else self._DEFAULT_PSM_MODES
        self._ocr = None
        self._available = False
        self._resolved_lang = lang
        try:
            import pytesseract
            # This validates both the Python package and the system binary.
            pytesseract.get_tesseract_version()
            self._ocr = pytesseract
            self._resolved_lang = self._resolve_lang(pytesseract, lang)
            self._available = True
        except Exception:
            self._ocr = None
            self._available = False

    @staticmethod
    def _resolve_lang(pytesseract_module, requested: str) -> str:
        """Fall back to whichever requested language packs are installed.

        A deployment missing the Vietnamese tessdata pack must still run
        English OCR instead of raising on every call.
        """
        requested_codes = [c for c in requested.split("+") if c]
        try:
            installed = set(pytesseract_module.get_languages(config=""))
        except Exception:
            return requested
        usable = [c for c in requested_codes if c in installed]
        if usable:
            return "+".join(usable)
        return "eng" if "eng" in installed else requested

    def available(self) -> bool:
        return self._available

    def _preprocess(self, image: Any):
        """Grayscale, upscale small frames, and boost contrast for OCR.

        Returns the original object unchanged if PIL isn't available or the
        input isn't a PIL image (e.g. a test double); Tesseract/pytesseract
        will then simply see whatever it was given.
        """
        try:
            from PIL import Image, ImageOps, ImageFilter
        except Exception:
            return image
        if not isinstance(image, Image.Image):
            return image
        try:
            gray = image.convert("L")
            width, height = gray.size
            min_dim = min(width, height) if width and height else 0
            if 0 < min_dim < self._UPSCALE_TARGET_MIN_DIM:
                scale = min(4.0, self._UPSCALE_TARGET_MIN_DIM / max(1, min_dim))
                gray = gray.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.LANCZOS,
                )
            gray = ImageOps.autocontrast(gray, cutoff=1)
            gray = gray.filter(ImageFilter.SHARPEN)
            return gray
        except Exception:
            return image

    def _run_once(self, image: Any, psm: int) -> OCRResult:
        from pytesseract import Output

        config = f"--psm {psm}"
        data = self._ocr.image_to_data(
            image, lang=self._resolved_lang, config=config, output_type=Output.DICT
        )
        words: List[str] = []
        confidences: List[float] = []
        n = len(data.get("text", []))
        for i in range(n):
            token = str(data["text"][i]).strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                # Tesseract uses -1 for structural (non-word) rows.
                continue
            if conf < self._MIN_WORD_CONFIDENCE:
                continue
            words.append(token)
            confidences.append(conf)
        text = " ".join(words).strip()
        mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        return OCRResult(text=text, confidence=mean_conf, variant=f"psm{psm}", word_count=len(words))

    def extract_detailed(self, image: Any) -> OCRResult:
        if not self._available or self._ocr is None or image is None:
            return OCRResult()
        processed = self._preprocess(image)
        best = OCRResult()
        for psm in self.psm_modes:
            try:
                result = self._run_once(processed, psm)
            except Exception:
                continue
            if result.text and (
                not best.text
                or result.confidence > best.confidence
                or (result.confidence == best.confidence and result.word_count > best.word_count)
            ):
                best = result
        return best

    def extract(self, image: Any) -> str:
        if not self._available or self._ocr is None or image is None:
            return ""
        try:
            return self.extract_detailed(image).text
        except Exception:
            return ""


def aggregate_across_frames(
    provider: OCRProvider, images: Sequence[Any], max_frames: int = 3
) -> OCRResult:
    """Run OCR over several candidate frames and keep the best result.

    Video keyframes are noisy: a caption or sign may be blurred, occluded, or
    mid-transition in any single frame. Stopping at the first frame that
    happens to have *some* recognizable text (the previous behavior) means the
    evidence layer often keeps a low-quality, low-confidence read even when a
    better one is a frame or two away. This scores every sampled frame and
    keeps the highest-confidence non-empty result instead.
    """
    best = OCRResult()
    if not provider.available():
        return best
    for image in list(images)[:max_frames]:
        try:
            result = provider.extract_detailed(image)
        except Exception:
            continue
        if result.text and (not best.text or result.confidence > best.confidence):
            best = result
        if best.confidence >= 0.90 and best.text:
            break  # Good enough; skip scanning further frames.
    return best

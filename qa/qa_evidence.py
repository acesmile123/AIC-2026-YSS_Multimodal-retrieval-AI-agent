from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import qa_config
from .qa_counting import CountingEngine
from .qa_evidence_memory import EvidenceMemory, EvidenceRecord
from .qa_grounding import Detection, GroundingEngine
from .qa_ocr import OCRProvider, OCRResult, TesseractOCR, aggregate_across_frames
from .qa_scene import SceneUnderstanding
from .qa_spatial import SpatialReasoner
from .qa_temporal import TemporalReasoner


@dataclass
class EvidenceBundle:
    video_id: str
    center_frame_id: int
    frames: list = field(default_factory=list)
    records: List[EvidenceRecord] = field(default_factory=list)
    scene: dict = field(default_factory=dict)
    count_summary: dict = field(default_factory=dict)
    count_value: Optional[int] = None
    count_confidence: float = 0.0
    ocr_text: str = ""
    qualified_count_value: Optional[int] = None
    qualified_count_confidence: float = 0.0
    qualified_count_notes: str = ""
    count_agreement: float = 0.0
    ocr_confidence: float = 0.0


class EvidenceBuilder:
    """Build structured visual evidence after KIS has located the candidate.

    Sources are layered and optional: supplied object metadata, captions, OCR,
    and an external grounding detector. No KIS internals are duplicated here.
    """
    def __init__(self, frame_loader, object_lookup=None, grounding=None, ocr=None, memory=None):
        self.frame_loader = frame_loader
        self.object_lookup = object_lookup
        self.grounding = grounding or GroundingEngine()
        # OCR is optional and self-disables when Tesseract is unavailable.
        self.ocr = ocr if ocr is not None else (
            TesseractOCR(lang=qa_config.OCR_LANG) if qa_config.ENABLE_OCR else None
        )
        self.scene = SceneUnderstanding()
        self.spatial = SpatialReasoner()
        self.temporal = TemporalReasoner()
        self.counting = CountingEngine()
        self.memory = memory or EvidenceMemory(qa_config.EVIDENCE_MEMORY_SIZE)
        self._captions = self._load_captions()
        self._metadata_cache: Dict[tuple[str, int], dict] = {}

    def _load_captions(self) -> Dict[tuple[str,int], str]:
        path = Path(qa_config.DATA_DIR) / "captions.json"
        if not path.exists(): return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out={}
            for row in data:
                try: out[(row["video_id"], int(row["frame_id"]))] = str(row.get("caption",""))
                except Exception: continue
            return out
        except Exception:
            return {}

    def _object_metadata(self, video_id: str, frame_id: int) -> dict:
        key = (video_id, int(frame_id))
        if key in self._metadata_cache:
            return self._metadata_cache[key]
        if self.object_lookup is None:
            self._metadata_cache[key] = {}
            return {}
        data = {}
        try:
            if hasattr(self.object_lookup, "get"):
                data = self.object_lookup.get(key) or {}
        except Exception:
            data = {}
        if not data and isinstance(self.object_lookup, dict):
            data = self.object_lookup.get(key, {}) or {}
        self._metadata_cache[key] = data
        return data

    @staticmethod
    def _infer_count_target(analysis=None) -> str:
        terms = list(getattr(analysis, "target_terms", ()) if analysis else ())
        if terms:
            # Keep the noun phrase intact. The BTC detector decides whether a
            # deterministic count is possible; unknown role/category words
            # must fall through to the VLM instead of being guessed as animals.
            return str(getattr(analysis, "count_target", "") or terms[0]).strip().lower()
        return str(getattr(analysis, "target", "") or "").strip().lower()

    def build(self, video_id: str, center_frame_id: int, window: int, step: int, analysis=None) -> EvidenceBundle:
        records_raw = self.frame_loader.load_temporal_records(video_id, center_frame_id, window=window, step=step)
        if not records_raw:
            return EvidenceBundle(video_id, center_frame_id)
        bundles=[]
        detections_per_frame=[]
        all_detections=[]
        scene=None
        best_ocr = OCRResult()

        types = set(getattr(analysis, "types", ()) or ()) if analysis is not None else set()
        expected = str(getattr(analysis, "expected_answer_type", "TEXT") or "TEXT").upper() if analysis is not None else "TEXT"
        needs_count = qa_config.COUNTING_ENABLED and expected == "COUNT"
        needs_spatial = bool(getattr(analysis, "relations", ()) or ()) or "SPATIAL" in types
        needs_temporal = bool(getattr(analysis, "needs_temporal", False)) or "TEMPORAL" in types
        needs_ocr = "OCR" in types
        # Grounding is useful for target/entity questions, but unnecessary for
        # generic descriptive or OCR-only questions. This keeps evidence cheap.
        needs_grounding = qa_config.OBJECT_GROUNDING_ENABLED and bool(types & {
            "COUNTING", "OBJECT", "ATTRIBUTE", "ACTION", "SPATIAL",
            "RELATIONSHIP", "COMPARISON", "YES_NO",
        })

        # Keep the full temporal evidence window here. Frame selection is a
        # separate stage; truncating first would make the selector unable to
        # recover onset/peak/completion frames for temporal questions.
        # Run OCR once, up front, across a bounded sample of frames rather than
        # stopping at the first frame that happens to yield any text. Subtitle
        # or sign text is often blurred, occluded, or mid-transition in a given
        # frame; scoring several candidates and keeping the best-confidence
        # read is far more reliable than a first-hit heuristic.
        if self.ocr is not None and qa_config.ENABLE_OCR and needs_ocr and self.ocr.available():
            try:
                candidate_images = [r["image"] for r in records_raw[: qa_config.OCR_MAX_FRAMES]]
                best_ocr = aggregate_across_frames(self.ocr, candidate_images, max_frames=qa_config.OCR_MAX_FRAMES)
            except Exception:
                best_ocr = OCRResult()

        max_evidence_records = max(qa_config.VLM_EVIDENCE_FRAME_LIMIT, len(records_raw))
        for rec in records_raw[:max_evidence_records]:
            fid=rec["frame_id"]
            meta=self._object_metadata(video_id, fid) if needs_grounding else {}
            dets=self.grounding.detect(rec["image"], metadata=meta) if needs_grounding else []
            detections_per_frame.append(dets)
            all_detections.extend(dets)
            caption=self._captions.get((video_id,fid),"")
            sc=self.scene.summarize(dets, caption)
            if scene is None: scene=sc
            er=EvidenceRecord(
                video_id, fid, caption, sc, dets, best_ocr.text,
                rec.get("requested_frame_id", fid), ocr_confidence=best_ocr.confidence,
            )
            self.memory.put(er); bundles.append(er)
        center_idx = len(bundles) // 2
        center_bundle = bundles[center_idx]
        summary=self.scene.summarize(center_bundle.detections, center_bundle.caption)
        # Preserve the BTC per-frame grounding payload in QA evidence so the VLM
        # can use the actual detected object locations from the candidate frame.
        center_meta = self._object_metadata(video_id, int(center_bundle.frame_id))
        if center_meta:
            summary["object_metadata_source"] = center_meta.get("source", "unknown")
            summary["object_metadata_status"] = center_meta.get("status", "unknown")
            summary["object_detection_count"] = int(center_meta.get("detection_count", len(center_bundle.detections)) or 0)
            summary["grounded_detections"] = [
                {"label": d.label, "score": round(float(d.score), 4), "box": [round(float(x), 4) for x in d.box]}
                for d in center_bundle.detections[:12]
            ]
        # Spatial relations are frame-local. Computing them on detections pooled
        # from multiple timestamps creates impossible cross-time relations and can
        # poison the VLM evidence context. For temporal stability, keep a compact
        # relation consensus across frames rather than concatenating all boxes.
        rel_counts = {}
        if needs_spatial:
            for bundle in bundles:
                for r in self.spatial.relations(bundle.detections):
                    key = (r.subject, r.relation, r.object)
                    rel_counts[key] = max(rel_counts.get(key, 0.0), float(r.confidence))
        summary["spatial_relations"] = [
            {"subject": k[0], "relation": k[1], "object": k[2], "confidence": v}
            for k, v in sorted(rel_counts.items(), key=lambda kv: kv[1], reverse=True)[:64]
        ]
        summary["temporal_events"] = (
            self.temporal.events(self.temporal.build_states([{"frame_id":r.frame_id,"detections":r.detections} for r in bundles]))
            if needs_temporal else []
        )
        summary["evidence_provenance"] = {
            "grounding": bool(all_detections),
            "captions": any(bool(r.caption) for r in bundles),
            "ocr": bool(best_ocr.text),
            "ocr_available": bool(self.ocr is not None and self.ocr.available()),
            "ocr_confidence": best_ocr.confidence,
            "temporal": len(bundles) > 1,
            "spatial": bool(rel_counts),
        }
        # OCR is folded into `scene` (not just returned separately) so every
        # consumer of scene -- the reranker, the evidence-context builder fed
        # to the VLM -- actually sees it. A best-confidence, low-noise read is
        # more useful downstream than a raw first-hit string, so only surface
        # it once it clears a minimum confidence floor; near-zero-confidence
        # OCR noise is worse than no OCR text at all for a VLM prompt.
        ocr_text_for_summary = best_ocr.text if best_ocr.confidence >= qa_config.OCR_MIN_CONFIDENCE else ""
        summary["ocr_text"] = ocr_text_for_summary
        target = getattr(analysis, "count_target", "") or self._infer_count_target(analysis)
        tracked=self.counting.count_frame_consensus(detections_per_frame, target) if needs_count and detections_per_frame else None
        count_summary=self.counting.render_summary(all_detections) if needs_count else {}

        qualified = None
        relations = tuple(getattr(analysis, "relations", ()) or ()) if analysis else ()
        if analysis is not None and expected == "COUNT" and relations and any((r.get("type") or "") not in {"reference_context"} for r in relations) and detections_per_frame:
            qualified = self.counting.count_conditional(
                detections_per_frame, target, relations, self.spatial,
                min_relation_confidence=qa_config.COUNT_RELATION_MIN_CONFIDENCE,
            )
            summary["qualified_count"] = {
                "target": qualified.target,
                "count": qualified.count,
                "confidence": qualified.confidence,
                "source": qualified.source,
                "notes": qualified.notes,
                "evidence_frames": qualified.evidence_frames,
            }

        return EvidenceBundle(
            video_id, center_frame_id, records_raw[:qa_config.VLM_EVIDENCE_FRAME_LIMIT], bundles,
            summary, count_summary, tracked.count if tracked else None,
            tracked.confidence if tracked else 0.0, ocr_text_for_summary,
            qualified.count if qualified and qualified.confidence > 0 else None,
            qualified.confidence if qualified else 0.0,
            qualified.notes if qualified else "",
            tracked.agreement if tracked else 0.0,
            ocr_confidence=best_ocr.confidence,
        )

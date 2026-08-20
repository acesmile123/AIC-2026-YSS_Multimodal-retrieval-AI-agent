from concurrent.futures import ThreadPoolExecutor
import inspect
from typing import Dict, List, Optional, Tuple, Any

from .frame_loader import TemporalFrameLoader
from . import qa_config
from .qa_answer_aggregator import aggregate_answers
from .qa_cache import LRUCache, answer_cache_key
from .qa_candidate_fusion import fuse_candidate_lists
from .qa_candidate_ranker import rank_for_submission
from .qa_evidence_context import build_evidence_context
from .qa_logging import get_logger
from .qa_question_analysis import analyze_question, infer_temporal_window
from .qa_answer_validation import validate_answer
from .qa_evidence import EvidenceBuilder
from .qa_evidence_memory import EvidenceMemory
from .qa_visual_reranker import VisualReranker
from .qa_frame_selection import AdaptiveFrameSelector
from .qa_frame_dedup import FrameDeduplicator
from .qa_types import QAAnswer, QACandidate, QARequest, QAResult, build_structured_query
from .agent_core_adapter import AgentCoreAdapter
from .vector_object_metadata import VectorObjectMetadataLookup

logger = get_logger("pipeline")


def _supports_evidence_context(vlm) -> bool:
    """Duck-type probe: does this injected VLM's generate_answer() accept an
    `evidence_context` kwarg (or **kwargs)? Keeps arbitrary injected/test
    VLMs that only implement the 2-arg contract working unmodified, while
    letting QwenVLEngine (and any upgraded fake) receive richer evidence."""
    try:
        params = inspect.signature(vlm.generate_answer).parameters
        return "evidence_context" in params or any(
            p.kind == p.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return False


def _supports_verify(vlm) -> bool:
    return callable(getattr(vlm, "verify_answer", None))


def _supports_verify_context(vlm) -> bool:
    fn = getattr(vlm, "verify_answer", None)
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
        return "evidence_context" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        return False


def _evidence_score(analysis, evidence) -> tuple[float, str]:
    score = 0.0
    notes = []
    scene = evidence.scene or {}
    target = getattr(analysis, "count_target", "") or getattr(analysis, "target", "")
    objects = scene.get("objects", {}) if isinstance(scene, dict) else {}
    if target and objects.get(target, 0):
        score += 0.25
        notes.append("target_grounded")
    if getattr(analysis, "expected_answer_type", "") == "COUNT":
        c = float(getattr(evidence, "count_confidence", 0.0) or 0.0)
        score += 0.25 * c
        if getattr(evidence, "qualified_count_value", None) is not None:
            qc = float(getattr(evidence, "qualified_count_confidence", 0.0) or 0.0)
            score += 0.25 * qc
            notes.append("qualified_count")
    if getattr(analysis, "relations", ()):
        if scene.get("spatial_relations"):
            score += 0.15
            notes.append("spatial_evidence")
        if getattr(evidence, "qualified_count_value", None) is None:
            notes.append("relation_not_deterministically_resolved")
    if getattr(analysis, "needs_temporal", False):
        if scene.get("temporal_events"):
            score += 0.15
            notes.append("temporal_evidence")
        else:
            notes.append("temporal_evidence_weak")
    if getattr(analysis, "expected_answer_type", "") == "TEXT" and scene.get("caption"):
        score += 0.10
    if getattr(analysis, "expected_answer_type", "") == "COLOR" and scene.get("attributes", {}).get("colors"):
        score += 0.10
    return min(1.0, score), ";".join(notes)


class QASystem:
    """End-to-end AIC Q&A orchestrator.

    Runtime flow:
        exam request -> internal query -> KIS retrieval -> question-aware
        refinement -> frame mapping/temporal evidence -> VQA/VLM -> ranking ->
        competition rows.

    Design contract:
        Dùng lại retrieval của module KIS để định vị frame; tích hợp VQA để
        sinh answer từ khung hình.

    KIS and VLM are injectable. This makes the exact same QASystem executable
    with real shared models, lightweight local adapters, or deterministic test
    doubles without rewriting the orchestration.
    """

    def __init__(self, retriever=None, clip_text_encoder=None, caption_retriever=None,
                 object_lookup=None, kis=None, vlm=None, frame_loader=None, agent_core=None):
        self.kis = kis
        if kis is not None:
            retriever = getattr(kis, "retriever", retriever)
            clip_text_encoder = getattr(kis, "clip_text_encoder", clip_text_encoder)
            caption_retriever = getattr(kis, "caption_retriever", caption_retriever)
            object_lookup = getattr(kis, "object_lookup", object_lookup)
        self.retriever = retriever
        self.clip_text_encoder = clip_text_encoder
        self.caption_retriever = caption_retriever
        self.object_lookup = object_lookup or {}
        # QA-only metadata source. The existing KIS implementation is untouched.
        if qa_config.OBJECT_METADATA_VECTOR_ENABLED:
            try:
                self.qa_object_lookup = VectorObjectMetadataLookup(
                    collection_name=qa_config.OBJECT_METADATA_COLLECTION,
                    milvus_uri=qa_config.OBJECT_METADATA_MILVUS_URI,
                    data_dir=qa_config.DATA_DIR,
                )
            except Exception:
                self.qa_object_lookup = self.object_lookup
        else:
            self.qa_object_lookup = self.object_lookup
        self.frame_loader = frame_loader or TemporalFrameLoader()
        self.vlm = vlm
        self.agent_core = agent_core or AgentCoreAdapter()
        self.analysis = self._analyze_question
        self.evidence = EvidenceBuilder(self.frame_loader, object_lookup=self.qa_object_lookup)
        self.visual_reranker = VisualReranker()
        self.frame_selector = AdaptiveFrameSelector()
        self.frame_dedup = FrameDeduplicator()
        self._answer_cache = LRUCache(max_size=qa_config.ANSWER_CACHE_SIZE)
        self._analysis_cache = LRUCache(max_size=128)
        self.last_run_diagnostics = []

    def _route_event_query(self, text: str, stage: str) -> tuple[str, dict]:
        try:
            raw, structured = self.agent_core.build_event_query(text)
            self.last_run_diagnostics.append({
                "stage": stage,
                "source": text,
                "query_variants": structured.get("query_variants", []),
                "agent_task_type": structured.get("agent_task_type"),
                "ok": True,
            })
            return raw, structured
        except Exception as exc:
            # The repo ships Agent Core, but a missing API key/network failure
            # must not make offline QA debugging impossible. Fall back to the
            # conservative raw query and record the exact reason.
            fallback = build_structured_query(text, query_variants=[text])
            self.last_run_diagnostics.append({"stage": stage, "ok": False, "fallback": "local", "error": repr(exc)})
            return text.strip(), fallback

    def _analyze_question(self, question: str):
        """Analyze with Agent Core semantics, then add deterministic local cues.

        Agent Core owns bilingual normalization and high-level QA intent. The
        local planner is only a deterministic execution helper for geometry,
        temporal windows, and counting constraints that are not represented in
        the public Agent Core QA schema.
        """
        cache_key = ("analysis", self._normalize_question(question).casefold())
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        local = analyze_question(question)
        try:
            agent = self.agent_core.analyze_question(question)
            answer_type = str(agent.get("answer_type") or "other").upper()
            mapped = {
                "COUNT": "COUNTING", "COLOR": "ATTRIBUTE", "TEXT": "GENERAL",
                "ENTITY": "OBJECT", "BOOLEAN": "YES_NO", "LOCATION": "SPATIAL",
                "TIME": "TEMPORAL", "ACTION": "ACTION", "OTHER": "GENERAL",
            }.get(answer_type, "GENERAL")
            types = list(local.types)
            if mapped != "GENERAL" and mapped not in types:
                types.insert(0, mapped)
            primary = mapped if mapped != "GENERAL" else local.primary_type
            expected = {
                "COUNT": "COUNT", "COLOR": "COLOR", "TEXT": "TEXT", "ENTITY": "ENTITY",
                "BOOLEAN": "YES_NO", "LOCATION": "LOCATION", "TIME": "TIME",
                "ACTION": "ACTION", "OTHER": local.expected_answer_type,
            }.get(answer_type, local.expected_answer_type)
            entities = tuple(e for e in (agent.get("entities") or []) if isinstance(e, dict))
            target = local.target
            if not target:
                for entity in entities:
                    if str(entity.get("type", "")).lower() in {"person", "object", "action", "scene", "text"}:
                        target = str(entity.get("value") or "").strip()
                        if target:
                            break
            attrs = list(agent.get("attributes") or local.attributes)
            for entity in entities:
                if entity.get("attribute"):
                    attrs.append({"entity": entity.get("value", ""), "name": "attribute", "value": entity["attribute"]})
            agent_relations = tuple(r for r in (agent.get("relations") or []) if isinstance(r, dict))
            relations = agent_relations or local.relations
            semantic = dict(local.semantic_query)
            semantic.update({
                "task_type": "QA",
                "agent_answer_type": answer_type,
                "query_variants": agent.get("query_variants", []),
                "entities": list(entities),
                "relations": list(relations),
                "visual_description": agent.get("visual_description", ""),
            })
            from .qa_question_analysis import QuestionAnalysis
            result = QuestionAnalysis(
                local.question, primary, tuple(dict.fromkeys(types)), local.needs_temporal,
                local.temporal_direction, expected, local.target_terms,
                local.is_multi_hop or len(types) > 1, operation=local.operation,
                target=target, relations=relations, attributes=tuple(attrs),
                semantic_query=semantic, analysis_source="agent_core+planner",
                semantic_confidence=1.0, execution_plan=local.execution_plan,
                count_target=local.count_target or (target if expected == "COUNT" else ""),
                count_constraints=local.count_constraints,
                reference_entities=local.reference_entities,
                temporal_constraints=local.temporal_constraints,
                query_complexity=local.query_complexity, subqueries=local.subqueries,
            )
        except Exception as exc:
            self.last_run_diagnostics.append({"stage": "agent_core_question", "ok": False, "error": repr(exc)})
            result = local
        self._analysis_cache.set(cache_key, result)
        return result

    def _ensure_vlm(self):
        if self.vlm is None:
            from .vlm_engine import QwenVLEngine
            self.vlm = QwenVLEngine()
        return self.vlm

    def _retrieve_candidates(self, structured_query: dict) -> List[Tuple[str, int, float]]:
        if self.kis is not None and hasattr(self.kis, "retrieve"):
            try:
                rows = list(self.kis.retrieve(structured_query))
                self.last_run_diagnostics.append({"stage": "kis", "ok": bool(rows), "count": len(rows)})
                return rows
            except Exception as err:
                self.last_run_diagnostics.append({"stage": "kis", "ok": False, "error": repr(err)})
                logger.exception("KIS adapter retrieval failed")
                return []
        # Compatibility path for legacy callers that inject the four KIS components
        # directly. The preferred architecture is always an injected KIS provider.
        if self.retriever is None or self.clip_text_encoder is None or self.caption_retriever is None:
            logger.error("No KIS provider configured")
            return []
        try:
            from solve_kis import solve_kis
            return list(solve_kis(structured_query=structured_query, retriever=self.retriever,
                                  clip_text_encoder=self.clip_text_encoder,
                                  caption_retriever=self.caption_retriever,
                                  object_lookup=self.object_lookup))
        except Exception as err:
            logger.error("Legacy KIS adapter failed: %s", err)
            return []

    def _base_candidates(self, structured_query: dict) -> List[QACandidate]:
        return [QACandidate(str(v), int(f), float(s)) for v, f, s in self._retrieve_candidates(structured_query)]

    def _augment_with_question(self, base: List[QACandidate], question: str, top_k_qa: int) -> List[QACandidate]:
        """Optional question-aware refinement exposed by KIS provider.

        QA does not recreate CLIP/caption retrieval itself; it asks the shared KIS
        provider for a question-focused candidate list and fuses it with the
        original structured-query retrieval.
        """
        if not base or not question.strip() or self.kis is None or not hasattr(self.kis, "retrieve_for_question"):
            return base
        try:
            raw = self.kis.retrieve_for_question(question, top_k=max(top_k_qa * 2, 50))
            q = [QACandidate(v, int(f), float(s)) for v, f, s in raw]
            if q:
                return fuse_candidate_lists([base, q], weights=[qa_config.QA_KIS_FUSION_WEIGHT, qa_config.QA_QUESTION_CLIP_WEIGHT])
        except Exception as err:
            logger.warning("Question-aware KIS retrieval failed: %s", err)
        return base

    def _question_refined_candidates(self, base: List[QACandidate], structured_query: dict, question: str, top_k: int) -> List[QACandidate]:
        """Refine base candidates only when the event query differs from the question.

        This centralizes the duplicate-retrieval guard for solve(), aggregation,
        and batch paths.
        """
        raw_query = self._normalize_question((structured_query or {}).get("raw_query", ""))
        if raw_query and raw_query.casefold() == self._normalize_question(question).casefold():
            return base
        return self._augment_with_question(base, question, top_k)

    @staticmethod
    def _normalize_question(question: str) -> str:
        return str(question or "").strip()

    def _answer_one_candidate(self, vid: str, fid: int, score: float, question: str, analysis=None) -> QAAnswer:
        key = answer_cache_key(vid, fid, question)
        cached = self._answer_cache.get(key)
        if cached is not None:
            return cached
        try:
            analysis = analysis or self.analysis(question)
            window, step = infer_temporal_window(question, analysis=analysis)
            evidence = self.evidence.build(vid, fid, window=window, step=step, analysis=analysis)
            if not evidence.frames:
                return QAAnswer(vid, fid, "", score, 0.0, error="no_visual_evidence", answer_type=analysis.expected_answer_type)

            selected = self.frame_selector.select(
                question, evidence.frames, analysis,
                limit=min(qa_config.VLM_EVIDENCE_FRAME_LIMIT, qa_config.VLM_MAX_INPUT_FRAMES),
            )
            selected = self.frame_dedup.select(selected)
            caption = evidence.scene.get("caption", "") if evidence.scene else ""
            visual_score = self.visual_reranker.score(question, caption, evidence.scene, analysis) if qa_config.VISUAL_RERANKING_ENABLED else 0.0
            frames = [r["image"] for r in selected]
            vlm = self._ensure_vlm()
            evidence_context = build_evidence_context(
                analysis, evidence.scene, evidence.ocr_text, evidence.count_summary, evidence.count_value,
            )
            if _supports_evidence_context(vlm):
                answer, confidence = vlm.generate_answer(frames, question, evidence_context=evidence_context)
            else:
                answer, confidence = vlm.generate_answer(frames, question)

            # Deterministic counting is allowed to override the VLM only when
            # the query is actually executable from grounded evidence. For a
            # compound query, use the *qualified* count, never the global count.
            direct_count_question = (
                qa_config.COUNTING_ENABLED
                and analysis.expected_answer_type == "COUNT"
                and "COUNTING" in tuple(analysis.types)
                and not bool(getattr(analysis, "relations", ()))
                and not any(
                    isinstance(c, dict) and c.get("type") in {"action", "attribute", "role", "reference_attribute", "temporal"}
                    for c in getattr(analysis, "count_constraints", ())
                )
            )
            qualified_count_ok = (
                qa_config.COUNTING_ENABLED
                and analysis.expected_answer_type == "COUNT"
                and getattr(evidence, "qualified_count_value", None) is not None
                and getattr(evidence, "qualified_count_confidence", 0.0) >= qa_config.COUNT_DETERMINISTIC_CONFIDENCE
            )
            count_agreement = float(getattr(evidence, "count_agreement", 0.0) or 0.0)
            if (
                direct_count_question
                and evidence.count_confidence >= qa_config.COUNT_DETERMINISTIC_CONFIDENCE
                and count_agreement >= qa_config.COUNT_MIN_AGREEMENT
                and evidence.count_value is not None
            ):
                answer = str(evidence.count_value)
                confidence = max(confidence, evidence.count_confidence)
            elif qualified_count_ok and count_agreement >= qa_config.COUNT_MIN_AGREEMENT:
                answer = str(evidence.qualified_count_value)
                confidence = max(confidence, evidence.qualified_count_confidence)

            ok, val_score, reason = validate_answer(answer, analysis) if qa_config.ANSWER_VALIDATION_ENABLED else (True, 0.5, "validation_disabled")
            evidence_score, evidence_notes = _evidence_score(analysis, evidence)
            hard_query = getattr(analysis, "query_complexity", "simple") == "hard" or analysis.is_multi_hop or len(analysis.types) >= 2
            must_verify = (
                hard_query
                or qualified_count_ok
                or analysis.expected_answer_type in {"COUNT", "COLOR", "YES_NO", "LOCATION", "TIME", "ACTION"}
                or not ok
                or confidence < 0.72
            )
            if qa_config.VLM_VERIFY_ENABLED and answer and must_verify and _supports_verify(vlm):
                try:
                    if _supports_verify_context(vlm):
                        support = vlm.verify_answer(frames, question, answer, evidence_context=evidence_context)
                    else:
                        support = vlm.verify_answer(frames, question, answer)
                    # Verification is an independent visual signal; it can lower
                    # the candidate's validation score instead of only increasing it.
                    val_score = max(0.0, min(1.0, 0.35 * val_score + 0.65 * support))
                    if support < 0.35:
                        reason = reason + ";weak_visual_support"
                    elif support >= 0.90:
                        reason = reason + ";visually_verified"
                except Exception as verify_err:
                    logger.debug("verify_answer failed for %s/%s: %s", vid, fid, verify_err)
            if not ok and ((
                direct_count_question
                and evidence.count_confidence >= qa_config.COUNT_DETERMINISTIC_CONFIDENCE
                and count_agreement >= qa_config.COUNT_MIN_AGREEMENT
                and evidence.count_value is not None
            ) or qualified_count_ok):
                fallback_count = evidence.count_value if direct_count_question else evidence.qualified_count_value
                answer = str(fallback_count) if fallback_count is not None else ""
                ok, val_score, reason = validate_answer(answer, analysis)
            result = QAAnswer(vid, fid, answer, score, confidence, None, visual_score, val_score, reason, analysis.expected_answer_type, {
                "question_type": analysis.primary_type,
                "analysis_source": analysis.analysis_source,
                "semantic_confidence": analysis.semantic_confidence,
                "semantic_query": analysis.semantic_query,
                "question_types": list(analysis.types),
                "execution_plan": list(getattr(analysis, "execution_plan", ())),
                "is_multi_hop": analysis.is_multi_hop,
                "needs_temporal": analysis.needs_temporal,
                "scene": evidence.scene,
                "count_summary": evidence.count_summary,
                "tracked_count": evidence.count_value,
                "count_agreement": getattr(evidence, "count_agreement", 0.0),
                "qualified_count": evidence.qualified_count_value,
                "qualified_count_confidence": evidence.qualified_count_confidence,
                "qualified_count_notes": evidence.qualified_count_notes,
                "count_target": getattr(analysis, "count_target", ""),
                "count_constraints": list(getattr(analysis, "count_constraints", ())),
                "reference_entities": list(getattr(analysis, "reference_entities", ())),
                "query_complexity": getattr(analysis, "query_complexity", "simple"),
                "ocr": evidence.ocr_text,
                "ocr_confidence": getattr(evidence, "ocr_confidence", 0.0),
                "evidence_frames": [r.frame_id for r in evidence.records],
            }, ranking_score=0.0, evidence_score=evidence_score, evidence_notes=evidence_notes)
            self._answer_cache.set(key, result)
            return result
        except Exception as err:
            self.last_run_diagnostics.append({"stage": "candidate", "video_id": vid, "frame_id": fid, "ok": False, "error": repr(err)})
            logger.exception("Candidate %s/%s failed", vid, fid)
            return QAAnswer(vid, fid, "", score, 0.0, error=str(err))

    def _prefetch_frames(self, candidates: List[QACandidate], question: str, analysis=None) -> None:
        if not candidates or qa_config.QA_FRAME_PREFETCH_WORKERS <= 1:
            return
        def load(c):
            try:
                w, s = infer_temporal_window(question, analysis=analysis)
                self.frame_loader.load_temporal_frames(c.video_id, c.frame_id, window=w, step=s)
            except Exception:
                pass
        workers = min(qa_config.QA_FRAME_PREFETCH_WORKERS, len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(load, candidates))

    def solve(self, request: QARequest, top_k: int = qa_config.MAX_SUBMISSION_CANDIDATES) -> List[dict]:
        request.validate()
        return self.solve_from_exam(request.event_description, request.question, top_k=top_k)

    def solve_from_exam(self, event_description: str, question: str,
                        top_k: int = qa_config.MAX_SUBMISSION_CANDIDATES) -> List[dict]:
        """Public bilingual entry point; all downstream reasoning is English-only."""
        self.last_run_diagnostics = []
        _, event_query = self._route_event_query(event_description, "event_agent_core")
        question_data = {}
        try:
            question_data = self.agent_core.analyze_question(question)
            question_en = self.agent_core.english_variant(question_data, question)
        except Exception as exc:
            if any(ord(ch) >= 128 for ch in question):
                # Preserve the real Agent Core failure (most commonly missing
                # GOOGLE_API_KEY) instead of misreporting it as a translation error.
                self.last_run_diagnostics.append({
                    "stage": "agent_core_question",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                raise RuntimeError(
                    f"Agent Core failed for Vietnamese query: {type(exc).__name__}: {exc}"
                ) from exc
            question_en = question.strip()
        retrieval_query = dict(event_query)
        retrieval_query["question_variants"] = list(dict.fromkeys(
            [*retrieval_query.get("query_variants", []), *question_data.get("query_variants", [])]
        ))
        return self._solve_qa_en(retrieval_query, question_en, top_k_qa=top_k, public_question=question)

    def query(self, question: str, video_query: str | None = None, top_k: int = 1) -> List[dict]:
        question = self._normalize_question(question)
        if not question:
            raise ValueError("question must be non-empty")
        retrieval_query = self._normalize_question(video_query) or question
        return self.solve_from_exam(retrieval_query, question, top_k=top_k)

    def solve_qa(self, structured_query: dict, question: str,
                 top_k_qa: int = qa_config.MAX_SUBMISSION_CANDIDATES) -> list:
        """Compatibility entry point using Agent Core at the language boundary."""
        self.last_run_diagnostics = []
        query = dict(structured_query or {})
        raw = str(query.get("raw_query") or "").strip()
        if raw and not query.get("query_variants"):
            _, routed = self._route_event_query(raw, "event_agent_core")
            query.update(routed)
        try:
            question_data = self.agent_core.analyze_question(question)
            question_en = self.agent_core.english_variant(question_data, question)
            question_en_variants = self.agent_core.english_variants(question_data, question)
            query["question_variants"] = list(dict.fromkeys(
                [*query.get("question_variants", []), *question_en_variants]
            ))
        except Exception as exc:
            if any(ord(ch) >= 128 for ch in question):
                self.last_run_diagnostics.append({
                    "stage": "agent_core_question",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                raise RuntimeError(
                    f"Agent Core failed for Vietnamese query: {type(exc).__name__}: {exc}"
                ) from exc
            question_en = question.strip()
        return self._solve_qa_en(query, question_en, top_k_qa=top_k_qa, public_question=question)

    def _solve_qa_en(self, structured_query: dict, question: str,
                     top_k_qa: int = qa_config.MAX_SUBMISSION_CANDIDATES,
                     public_question: str | None = None) -> list:
        """Internal English-only pipeline. KIS receives exactly this English query."""
        question = self._normalize_question(question)
        top_k_qa = min(max(int(top_k_qa), 0), qa_config.MAX_SUBMISSION_CANDIDATES)
        if not question or top_k_qa == 0:
            return []
        analysis = self.analysis(question)
        query_for_kis = dict(structured_query)
        query_for_kis.setdefault("question_semantic", analysis.semantic_query)
        query_for_kis.setdefault("question_analysis_source", analysis.analysis_source)
        base = self._base_candidates(query_for_kis)
        self.last_run_diagnostics.append({
            "stage": "question_analysis",
            "ok": True,
            "source": analysis.analysis_source,
            "primary_type": analysis.primary_type,
            "types": list(analysis.types),
            "operation": analysis.operation,
            "target": analysis.target,
            "relations": list(analysis.relations),
            "count_target": getattr(analysis, "count_target", ""),
            "count_constraints": list(getattr(analysis, "count_constraints", ())),
            "query_complexity": getattr(analysis, "query_complexity", "simple"),
        })
        if not base:
            return []
        raw_query = self._normalize_question(query_for_kis.get("raw_query", ""))
        same_as_question = bool(raw_query) and raw_query.casefold() == question.casefold()
        candidates = self._question_refined_candidates(base, query_for_kis, question, top_k_qa)
        if same_as_question:
            self.last_run_diagnostics.append({"stage": "question_refinement", "ok": True, "skipped": "duplicate_query"})
        retrieval_pool_k = min(max(qa_config.QA_RETRIEVAL_POOL_K, top_k_qa), len(candidates))
        retrieval_pool = candidates[:retrieval_pool_k]
        # Submission K and reasoning budget are different concepts. Even when
        # the caller asks for one output row, evaluate several candidate frames
        # so a slightly-wrong top retrieval does not become the final answer.
        vlm_budget = min(len(retrieval_pool), max(qa_config.QA_VLM_CANDIDATE_BUDGET, top_k_qa))
        pool = retrieval_pool[:max(1, vlm_budget)] if retrieval_pool else []
        self._prefetch_frames(pool, question, analysis=analysis)
        answers = [self._answer_one_candidate(c.video_id, c.frame_id, c.retrieval_score, question, analysis=analysis) for c in pool]
        self.last_run_diagnostics.append({
            "stage": "vqa",
            "ok": any(a.error is None and not a.is_unknown for a in answers),
            "answers": len(answers),
            "retrieval_pool": retrieval_pool_k,
            "vlm_budget": vlm_budget,
        })
        ranked = rank_for_submission(question, answers)
        if not ranked:
            self.last_run_diagnostics.append({"stage": "ranking", "ok": False, "reason": "no_valid_answers", "errors": [a.error for a in answers if a.error]})
        return [a.to_qa_dict() for a in ranked[:top_k_qa]]

    def solve_qa_with_aggregation(self, structured_query: dict, question: str,
                                  top_k_qa: int = 3) -> QAResult:
        self.last_run_diagnostics = []
        query = dict(structured_query or {})
        raw = str(query.get("raw_query") or "").strip()
        if raw and not query.get("query_variants"):
            _, routed = self._route_event_query(raw, "event_agent_core")
            query.update(routed)
        try:
            question_data = self.agent_core.analyze_question(question)
            question_en = self.agent_core.english_variant(question_data, question)
            query["question_variants"] = list(dict.fromkeys([*query.get("query_variants", []), *question_data.get("query_variants", [])]))
        except Exception:
            question_en = question.strip()
        base = self._base_candidates(query)
        if not base:
            return QAResult(question, None, 0.0, [], 0, "no_candidates")
        candidates = self._question_refined_candidates(base, query, question_en, top_k_qa)[:top_k_qa]
        analysis = self.analysis(question_en)
        answers = [self._answer_one_candidate(c.video_id, c.frame_id, c.retrieval_score, question_en, analysis=analysis) for c in candidates]
        result = aggregate_answers(question_en, answers)
        result.question = question
        return result

    def batch_solve_qa(self, structured_query: dict, questions: List[str], top_k_qa: int = 3) -> Dict[str, QAResult]:
        query = dict(structured_query or {})
        raw = str(query.get("raw_query") or "").strip()
        if raw and not query.get("query_variants"):
            _, routed = self._route_event_query(raw, "event_agent_core")
            query.update(routed)
        base = self._base_candidates(query)
        results: Dict[str, QAResult] = {}
        for public_question in questions:
            try:
                q_data = self.agent_core.analyze_question(public_question)
                q_en = self.agent_core.english_variant(q_data, public_question)
            except Exception:
                q_en = public_question.strip()
            if not base:
                results[public_question] = QAResult(public_question, None, 0.0, [], 0, "no_candidates")
                continue
            candidates = self._question_refined_candidates(base, query, q_en, top_k_qa)[:top_k_qa]
            analysis = self.analysis(q_en)
            answers = [self._answer_one_candidate(c.video_id, c.frame_id, c.retrieval_score, q_en, analysis=analysis) for c in candidates]
            result = aggregate_answers(q_en, answers)
            result.question = public_question
            results[public_question] = result
        return results

    @classmethod
    def from_shared_kis(cls, **kwargs: Any) -> "QASystem":
        from .kis_adapter import create_shared_kis
        return cls(kis=create_shared_kis(**kwargs))

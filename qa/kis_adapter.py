from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List, Protocol, Sequence, Tuple

from .qa_candidate_fusion import fuse_candidate_lists
from .milvus_preflight import ensure_milvus_available
from .qa_types import QACandidate

QA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = QA_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class KISProvider(Protocol):
    """KIS -> QA boundary.

    QA reuses the retrieval results from the KIS module to locate candidate
    frames; QA does not own or rebuild the shared retrieval backend.
    """
    def retrieve(self, structured_query: dict) -> List[Tuple[str, int, float]]: ...

    def retrieve_for_question(self, question: str, top_k: int = 100, query_variants: Sequence[str] | None = None) -> List[Tuple[str, int, float]]: ...


class SharedKIS:
    """Adapter around the existing shared KIS implementation.

    QA flow: reuse KIS retrieval to locate frames, then pass those frames to
    the VQA/VLM layer to generate the answer.

    It owns the shared CLIP/caption/Milvus objects exactly once and exposes a
    stable interface to QASystem. Object metadata is also wired when available.
    """

    def __init__(self, data_dir: str | None = None, index_dir: str | None = None):
        if not (PROJECT_ROOT / "search_clip.py").exists():
            raise FileNotFoundError(f"Shared KIS modules not found: {PROJECT_ROOT}")
        root_data = Path(data_dir or PROJECT_ROOT / "data")
        idx = Path(index_dir or PROJECT_ROOT / "index")

        preflight = ensure_milvus_available()
        if not preflight.available:
            raise RuntimeError(preflight.message)

        # search_clip.py is the repository-level Milvus wrapper; the remaining
        # KIS implementation lives in the KISBranche package.  Keep this
        # boundary explicit so QA does not depend on the process working
        # directory or ad-hoc sys.path entries.
        from search_clip import ClipRetriever
        from KISBranche.clip_encoder import ClipEncoder
        from KISBranche.caption_retriever import CaptionRetriever
        from KISBranche.solve_kis import solve_kis
        from KISBranche.object_filter import ObjectMetadataLookup

        self._solve_kis = solve_kis
        self.retriever = ClipRetriever()
        self.clip_text_encoder = ClipEncoder()
        self.caption_retriever = CaptionRetriever(
            json_path=str(root_data / "captions.json"),
            index_path=str(idx / "caption.index"),
            mapping_path=str(idx / "caption_mapping.json"),
        )
        self.object_lookup = ObjectMetadataLookup(str(root_data))

    def retrieve(self, structured_query: dict) -> List[Tuple[str, int, float]]:
        return self._solve_kis(
            structured_query=structured_query,
            retriever=self.retriever,
            clip_text_encoder=self.clip_text_encoder,
            caption_retriever=self.caption_retriever,
            object_lookup=self.object_lookup,
        )

    def retrieve_for_question(self, question: str, top_k: int = 100, query_variants: Sequence[str] | None = None) -> List[Tuple[str, int, float]]:
        """Question-aware retrieval using the canonical English query signal from Agent Core.

        QA receives the English-first contract and does not mix Vietnamese variants
        into KIS retrieval.
        """
        variants = [str(q).strip() for q in (query_variants or []) if str(q).strip()]
        if question and question not in variants:
            variants.insert(0, question.strip())
        ranked_lists = []
        weights = []
        for idx, variant in enumerate(dict.fromkeys(variants)):
            try:
                clip_rows = self.retriever.search(self.clip_text_encoder.encode_text(variant), top_k=top_k)
                clip_list = [QACandidate(r["video_id"], int(r["frame_id"]), float(r["score"])) for r in clip_rows]
                if clip_list:
                    ranked_lists.append(clip_list)
                    weights.append(1.0 if idx == 0 else 0.85)
            except Exception:
                pass
            try:
                cap_rows = self.caption_retriever.search(variant, top_k=top_k)
                cap_list = [QACandidate(r["video_id"], int(r["frame_id"]), float(r["score"])) for r in cap_rows]
                if cap_list:
                    ranked_lists.append(cap_list)
                    weights.append(0.9 if idx == 0 else 0.75)
            except Exception:
                pass
        if not ranked_lists:
            return []
        fused = fuse_candidate_lists(ranked_lists, weights=weights)
        return [(c.video_id, c.frame_id, c.retrieval_score) for c in fused[:top_k]]


def create_shared_kis(**kwargs: Any) -> SharedKIS:
    return SharedKIS(**kwargs)

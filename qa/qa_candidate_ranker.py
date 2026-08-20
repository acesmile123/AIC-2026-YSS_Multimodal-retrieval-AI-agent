# qa_candidate_ranker.py
"""
Xếp hạng CÁC candidate riêng lẻ (không gộp thành 1 câu trả lời như
qa_answer_aggregator.py) để tạo output competition-facing của
QASystem.solve_qa(): một danh sách [{video_id, frame_id, answer, score}, ...]
đã xếp hạng từ khả năng đúng cao nhất -> thấp nhất.

Khác biệt với qa_answer_aggregator.py:
  - aggregate_answers()   -> MỘT câu trả lời cuối cùng + confidence
                              (dùng cho solve_qa_with_aggregation()).
  - rank_for_submission() -> NHIỀU dòng (video_id, frame_id, answer),
                              mỗi candidate hợp lệ một dòng, chỉ đổi THỨ
                              TỰ xếp hạng (dùng cho solve_qa()).

Thiết kế bám sát ĐÚNG công thức chấm điểm vòng sơ tuyển AIC 2026 (mục 2):

  R-Score(rᵢ) = I(video đúng ∧ frame ∈ [s,e] ∧ answer khớp ngữ nghĩa)
  R@k         = max R-Score trong k dòng đầu tiên
  Final Score = trung bình R@k với k ∈ {1, 5, 20, 50, 100}

Hai hệ quả trực tiếp từ công thức này quyết định cách ranker hoạt động:

  1. R@k KHÔNG GIẢM khi k tăng, và câu trả lời sai không bị trừ điểm gì cả
     - chỉ có "đúng" (được tính vào max) hoặc "không đúng" (không đóng góp
     gì, y hệt như không nộp dòng đó). Vì vậy KHÔNG BAO GIỜ nên bỏ trống
     một suất trong tối đa 100 suất nộp: một câu ĐOÁN LIỀU luôn có xác
     suất đúng > 0, còn từ chối trả lời (UNKNOWN/rỗng) thì chắc chắn = 0.
     -> pipeline.py + vlm_engine.py đã được chỉnh để VLM luôn cố đưa ra
     một câu trả lời cụ thể (dùng CONFIDENCE=LOW để thể hiện không chắc
     chắn) thay vì mặc định từ chối; ranker vẫn CHẶN (phòng vệ 2 lớp) mọi
     candidate LỖI, rỗng, hoặc literal "UNKNOWN" lọt qua - các giá trị đó
     chắc chắn không thể khớp GT_a nên không đáng chiếm một suất nộp.
  2. R-Score cần ĐÚNG VIDEO trước tiên. Nếu giả thuyết video ở hạng 1 sai,
     thì MỌI candidate khác thuộc CHÍNH video đó cũng sai theo dù frame/
     answer có tốt đến đâu - nên để nhiều candidate của CÙNG MỘT video
     chiếm hết các hạng đầu (1..20) là lãng phí "suất" cho R@5/R@20. Sau
     khi xếp theo composite score, ranker áp dụng một bước ĐA DẠNG HOÁ
     nhẹ theo video_id (xem _diversify_by_video) để các giả thuyết VIDEO
     KHÁC cũng có cơ hội xuất hiện sớm, phòng khi giả thuyết hạng 1 sai.

Vì sao composite score vẫn hữu ích (không chỉ đa dạng hoá):
  a. Candidate có câu trả lời TRÙNG (cùng ý) với nhiều candidate khác
     trong cùng pool được xem là có bằng chứng mạnh hơn (nhiều frame độc
     lập cùng "nhìn thấy" một điều) - được cộng thêm điểm đồng thuận.
  b. Độ tự tin VLM tự báo cáo (HIGH/MEDIUM/LOW từ prompt) được tận dụng
     làm tín hiệu bổ sung thay vì bỏ phí.
  c. Điểm retrieval gốc (đã qua CLIP+caption+RRF+object-filter của KIS)
     vẫn là tín hiệu chính cho độ đúng của VIDEO/FRAME, vì phần đó nằm
     ngoài khả năng của VLM (VLM chỉ đọc frame được đưa cho nó).
"""
from typing import Dict, List

from . import qa_config
from .qa_text_similarity import cluster_by_similarity, canonicalize_answer
from .qa_types import QAAnswer


def _consensus_weight(answer: QAAnswer) -> float:
    """'Phiếu bầu' của một candidate cho nhóm câu trả lời của nó: điểm
    retrieval của chính nó, giảm nhẹ nếu VLM tự báo cáo ít tự tin hơn."""
    confidence = getattr(answer, "vlm_confidence", 1.0)
    return max(answer.score, 0.0) * (0.5 + 0.5 * confidence)


def _diversify_by_video(
    scored: List[tuple],
    repetition_decay: float,
) -> List[QAAnswer]:
    remaining = list(scored)
    picked: List[QAAnswer] = []
    video_penalty: Dict[str, float] = {}

    while remaining:
        best_index = 0
        best_effective = None
        for i, (composite, _score, ans) in enumerate(remaining):
            effective = composite * video_penalty.get(ans.video_id, 1.0)
            if best_effective is None or effective > best_effective:
                best_effective = effective
                best_index = i

        composite, score, ans = remaining.pop(best_index)
        picked.append(ans)
        video_penalty[ans.video_id] = video_penalty.get(ans.video_id, 1.0) * repetition_decay

    return picked


def rank_for_submission(
    question: str,
    answers: List[QAAnswer],
    similarity_threshold: float = qa_config.ANSWER_SIMILARITY_THRESHOLD,
    retrieval_weight: float = qa_config.RANK_RETRIEVAL_WEIGHT,
    consensus_weight: float = qa_config.RANK_CONSENSUS_WEIGHT,
    confidence_weight: float = qa_config.RANK_CONFIDENCE_WEIGHT,
    visual_weight: float = qa_config.RANK_VISUAL_WEIGHT,
    validation_weight: float = qa_config.RANK_VALIDATION_WEIGHT,
    evidence_weight: float = qa_config.RANK_EVIDENCE_WEIGHT,
    diversify_by_video: bool = qa_config.RANK_DIVERSIFY_BY_VIDEO,
    video_repetition_decay: float = qa_config.RANK_VIDEO_REPETITION_DECAY,
) -> List[QAAnswer]:
    usable = [
        a for a in answers
        if a.error is None and not a.is_unknown and a.answer.strip()
    ]
    if not usable:
        return []

    groups = cluster_by_similarity(
        usable,
        key_func=lambda a: canonicalize_answer(a.answer, getattr(a, "answer_type", "TEXT")),
        threshold=similarity_threshold,
    )

    # Configuration weights are normalized defensively so a tuning mistake
    # cannot saturate all candidates at ranking_score=1.0.
    raw_weights = [retrieval_weight, consensus_weight, confidence_weight, visual_weight, validation_weight, evidence_weight]
    weight_sum = sum(max(0.0, float(w)) for w in raw_weights) or 1.0
    retrieval_weight, consensus_weight, confidence_weight, visual_weight, validation_weight, evidence_weight = [
        max(0.0, float(w)) / weight_sum for w in raw_weights
    ]
    for group in groups:
        # Frames from the same video are correlated. Count at most the strongest
        # vote per video so 5 adjacent frames do not masquerade as 5 independent
        # confirmations of the same answer.
        by_video = {}
        for member in group["members"]:
            by_video[member.video_id] = max(by_video.get(member.video_id, 0.0), _consensus_weight(member))
        group["weight"] = sum(by_video.values())

    group_of_id = {}
    for group in groups:
        for member in group["members"]:
            group_of_id[id(member)] = group

    max_retrieval = max((a.score for a in usable), default=0.0) or 1.0
    total_consensus = sum(g["weight"] for g in groups) or 1.0

    scored = []
    for a in usable:
        norm_retrieval = max(a.score, 0.0) / max_retrieval
        norm_consensus = group_of_id[id(a)]["weight"] / total_consensus
        confidence = getattr(a, "vlm_confidence", 1.0)
        visual = getattr(a, "visual_relevance_score", 0.0)
        validation = getattr(a, "validation_score", 0.0)
        evidence = getattr(a, "evidence_score", 0.0)

        composite = (
            retrieval_weight * norm_retrieval
            + consensus_weight * norm_consensus
            + confidence_weight * confidence
            + visual_weight * visual
            + validation_weight * validation
            + evidence_weight * evidence
        )
        scored.append((composite, a.score, a))

    scored.sort(key=lambda triple: (triple[0], triple[1]), reverse=True)
    for composite, _score, ans in scored:
        ans.ranking_score = float(max(0.0, min(1.0, composite)))

    if not diversify_by_video or video_repetition_decay >= 1.0:
        return [a for _, _, a in scored]

    ordered = _diversify_by_video(scored, repetition_decay=video_repetition_decay)
    return ordered

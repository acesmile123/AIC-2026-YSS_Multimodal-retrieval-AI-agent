# qa_answer_aggregator.py
"""
Gộp nhiều câu trả lời VLM (một câu trả lời / candidate bằng chứng) thành
một câu trả lời cuối cùng kèm độ tin cậy.

Động lực: solve_qa_with_aggregation()/batch_solve_qa() lấy top-K candidate
từ KIS rồi hỏi VLM riêng lẻ cho từng candidate. Với K > 1 sẽ có nhiều câu
trả lời có thể lệch nhau (do frame khác nhau, hoặc do VLM không chắc
chắn). Việc chỉ hiển thị top-1 theo retrieval score bỏ qua tín hiệu "đồng
thuận" giữa các candidate - đây chính là chỗ nhánh Q&A có thể được "tăng
cường" (strengthen) mà không cần train lại model nào.

Chiến lược:
  1. Loại các câu trả lời rỗng / UNKNOWN / lỗi khỏi việc vote (trừ khi TẤT
     CẢ đều thuộc diện này).
  2. Gộp các câu trả lời gần giống nhau thành nhóm (qa_text_similarity).
  3. Mỗi nhóm được tính điểm = tổng "phiếu bầu" của các candidate thuộc
     nhóm; phiếu bầu = retrieval_score, giảm nhẹ theo độ tự tin VLM tự
     báo cáo (vlm_confidence) nếu có - candidate mà VLM tự nhận là ít
     chắc chắn thì phiếu nhẹ hơn một chút, thay vì bị bỏ qua hoàn toàn.
  4. Chọn nhóm có điểm cao nhất; đại diện của nhóm là câu trả lời có
     retrieval_score cao nhất trong nhóm (giữ nguyên văn, không lai ghép
     các câu trả lời khác nhau).
  5. confidence = điểm nhóm thắng / tổng điểm tất cả các nhóm hợp lệ.

Xem thêm qa_candidate_ranker.py: cùng dùng qa_text_similarity để gộp
nhóm, nhưng phục vụ mục đích khác (xếp hạng NHIỀU dòng output thay vì gộp
thành MỘT câu trả lời).
"""
from typing import List

from . import qa_config
from .qa_text_similarity import cluster_by_similarity
from .qa_types import QAAnswer, QAResult


def _vote_weight(answer: QAAnswer) -> float:
    confidence = getattr(answer, "vlm_confidence", 1.0)
    visual = getattr(answer, "visual_relevance_score", 0.0)
    validation = getattr(answer, "validation_score", 0.0)
    return max(answer.score, 0.0) * (0.45 + 0.30 * confidence + 0.15 * visual + 0.10 * validation)


def aggregate_answers(
    question: str,
    answers: List[QAAnswer],
    similarity_threshold: float = qa_config.ANSWER_SIMILARITY_THRESHOLD,
) -> QAResult:
    if not answers:
        return QAResult(
            question=question,
            final_answer=None,
            confidence=0.0,
            evidence=[],
            candidates_considered=0,
            note="no_candidates",
        )

    usable = [a for a in answers if not a.is_unknown and a.error is None]
    if not usable:
        # Tất cả đều UNKNOWN/lỗi -> không đủ bằng chứng, nhưng vẫn trả về
        # evidence đầy đủ để caller có thể tự kiểm tra.
        return QAResult(
            question=question,
            final_answer=None,
            confidence=0.0,
            evidence=answers,
            candidates_considered=len(answers),
            note="all_unknown_or_error",
        )

    groups = cluster_by_similarity(usable, key_func=lambda a: a.answer, threshold=similarity_threshold)
    for group in groups:
        by_video = {}
        for member in group["members"]:
            by_video[member.video_id] = max(by_video.get(member.video_id, 0.0), _vote_weight(member))
        group["weight"] = sum(by_video.values())

    groups.sort(key=lambda g: g["weight"], reverse=True)
    winner = groups[0]
    total_weight = sum(g["weight"] for g in groups) or 1.0
    confidence = winner["weight"] / total_weight

    representative = max(winner["members"], key=lambda a: a.score)

    return QAResult(
        question=question,
        final_answer=representative.answer,
        confidence=round(confidence, 4),
        evidence=answers,
        candidates_considered=len(answers),
        note=None,
    )

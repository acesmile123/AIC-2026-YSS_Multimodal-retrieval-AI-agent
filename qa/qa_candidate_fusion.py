# qa_candidate_fusion.py
"""
Hợp nhất (fuse) nhiều danh sách candidate ĐÃ được xếp hạng riêng, dùng
Reciprocal Rank Fusion (RRF) - cùng công thức mà solve_kis() dùng nội bộ
để gộp các query_variant CLIP + caption, nhưng đây là một lớp RIÊNG ở
tầng Q&A.

Vì sao cần lớp fusion riêng thay vì sửa solve_kis()?
  - Yêu cầu của dự án: KHÔNG thay đổi/redesign KIS retrieval backbone.
  - structured_query (raw_query + query_variants + entities) được xây
    cho pha retrieval tổng quát (KIS), trong khi CÂU HỎI Q&A thường mô tả
    chi tiết trực quan cụ thể hơn nhiều (vd. structured_query = "người
    đàn ông đạp xe trên phố", question = "người đàn ông đội mũ màu gì?").
    Một candidate mà CLIP/caption "nhìn thấy" đúng chi tiết trong CÂU HỎI
    (mũ) nhưng bị xếp hạng thấp bởi structured_query (vì query đó không
    nhắc đến mũ) vẫn xứng đáng được nâng hạng cho riêng câu hỏi này.
  - Đây chính là "question-aware candidate refinement": pipeline.py dùng
    module này để hợp nhất candidate gốc từ solve_kis() với candidate lấy
    trực tiếp bằng câu hỏi (qua đúng retriever/encoder đã được nạp sẵn,
    KHÔNG load thêm model nào).

Không có dependency ngoài stdlib + qa_types, nên test được mà không cần
torch/Milvus thật (dùng QACandidate giả lập).
"""
from typing import List, Optional, Sequence

from .qa_types import QACandidate


def fuse_candidate_lists(
    ranked_lists: Sequence[Sequence[QACandidate]],
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
) -> List[QACandidate]:
    """
    RRF: candidate ở hạng `rank` (1-indexed) trong danh sách có trọng số
    `w` đóng góp `w / (k + rank)` vào điểm hợp nhất của nó. Các danh sách
    không cần cùng độ dài; candidate xuất hiện ở nhiều danh sách được
    cộng dồn điểm (đồng thuận giữa các tín hiệu retrieval khác nhau).

    Trả về List[QACandidate] duy nhất theo (video_id, frame_id), đã sắp
    xếp giảm dần theo điểm RRF hợp nhất (gán lại vào `retrieval_score`).
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights phải cùng độ dài với ranked_lists")

    fused_scores = {}
    representative = {}

    for lst, weight in zip(ranked_lists, weights):
        for rank, cand in enumerate(lst, start=1):
            key = (cand.video_id, cand.frame_id)
            fused_scores[key] = fused_scores.get(key, 0.0) + weight / (k + rank)
            if key not in representative:
                representative[key] = cand

    fused = []
    for key, score in fused_scores.items():
        fused.append(
            QACandidate(video_id=key[0], frame_id=key[1], retrieval_score=score)
        )

    fused.sort(key=lambda c: c.retrieval_score, reverse=True)
    return fused

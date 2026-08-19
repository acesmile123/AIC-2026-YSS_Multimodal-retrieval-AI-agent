from collections import defaultdict
from apply_object_filter import apply_object_filter 


def multi_query_search(query_variants, encoder, retriever, top_k=50):
    print(f"[CLIP Search] {len(query_variants)} query variants")

    all_results = []

    for query in query_variants:
        print(f"  Query: {query}")

        vector = encoder.encode_text(query)
        results = retriever.search(vector, top_k=top_k)

        all_results.append(results)

    return all_results


def reciprocal_rank_fusion(multi_results, k=60):
    scores = defaultdict(float)
    metadata = {}

    for results in multi_results:
        for rank, item in enumerate(results):
            key = (item["video_id"], item["frame_id"])
            scores[key] += 1.0 / (k + rank + 1)
            metadata[key] = item

    fused = []

    for key, score in scores.items():
        item = metadata[key].copy()
        item["score"] = score
        fused.append(item)

    fused.sort(key=lambda x: x["score"], reverse=True)

    print(f"[RRF] {len(fused)} unique candidates")

    return fused


def normalize_scores(results):
    if not results:
        return []

    scores = [x["score"] for x in results]
    min_score = min(scores)
    max_score = max(scores)

    for item in results:
        if max_score == min_score:
            item["norm_score"] = 1.0
        else:
            item["norm_score"] = (
                (item["score"] - min_score)
                / (max_score - min_score)
            )

    return results


def fusion_clip_caption(
    clip_results,
    caption_results,
    k=60,
    caption_weight=1.0
    ):
    print("[Fusion] CLIP + Caption RRF")

    candidates = {}


    for rank, item in enumerate(clip_results, start=1):

        key = (
            item["video_id"],
            item["frame_id"]
        )

        rrf_score = 1.0 / (k + rank)

        if key not in candidates:
            candidates[key] = {
                "video_id": item["video_id"],
                "frame_id": item["frame_id"],
                "clip_score": 0.0,
                "caption_score": 0.0,
                "fusion_score": 0.0
            }

        candidates[key]["clip_score"] = rrf_score


    for rank, item in enumerate(caption_results, start=1):

        key = (
            item["video_id"],
            item["frame_id"]
        )

        rrf_score = (
            caption_weight /
            (k + rank)
        )

        if key not in candidates:
            candidates[key] = {
                "video_id": item["video_id"],
                "frame_id": item["frame_id"],
                "clip_score": 0.0,
                "caption_score": 0.0,
                "fusion_score": 0.0
            }

        candidates[key]["caption_score"] = rrf_score

    results = []

    for item in candidates.values():

        item["fusion_score"] = (
            item["clip_score"]
            +
            item["caption_score"]
        )

        results.append(item)


    results.sort(
        key=lambda x: x["fusion_score"],
        reverse=True
    )
    for item in results:
        item["score"] = item["fusion_score"]

    print(
        f"[Fusion] {len(results)} unique candidates"
    )

    return results


def solve_kis(
    structured_query,
    retriever,
    clip_text_encoder,
    caption_retriever,
    object_lookup
):
    query_variants = (
        structured_query.get("query_variants")
        or [structured_query.get("raw_query", "")]
    )

    entities = structured_query.get("entities", [])

    if not query_variants or not any(query_variants):
        raise ValueError("No query variants")

    # 1. CLIP Multi-query Search
    multi_results = multi_query_search(
        query_variants,
        clip_text_encoder,
        retriever,
        top_k=50
    )

    # 2. RRF các query variants
    clip_results = reciprocal_rank_fusion(multi_results)

    # 3. Caption Search
    raw_query = structured_query.get("raw_query", "")

    caption_multi_results = []

    for query in query_variants:

        print(f"[Caption Search] Query: {query}")

        results = caption_retriever.search(
            query,
            top_k=100
        )

        caption_multi_results.append(results)

    caption_results = reciprocal_rank_fusion(
        caption_multi_results
    )

    # 4. Fusion CLIP + Caption
    candidates = fusion_clip_caption(
        clip_results,
        caption_results,
        alpha=0.7
    )

    # 5. Object Filter
    candidates = apply_object_filter(
        candidates,
        entities,
        object_lookup
    )

    # 6. Final results
    results = [
        (
            x["video_id"],
            x["frame_id"],
            x["score"]
        )
        for x in candidates[:100]
    ]

    print(f"[KIS] Final results: {len(results)}")

    return results

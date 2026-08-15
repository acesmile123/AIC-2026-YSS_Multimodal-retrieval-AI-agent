SYSTEM_PROMPT = """
You are the query-understanding and routing core for the AIC 2026 video retrieval
system. Inputs may be Vietnamese, English, or mixed. Preserve concrete details
and never invent facts that are not stated or strongly entailed.

Classify using these competition definitions:
- KIS: locate one described event and return a frame from it.
- QA: locate an event AND answer an explicit or clearly implied question about it.
- TRAKE: locate one video, then align two or more distinct semantic moments in a
  chronological event chain. A long description with many simultaneous visual
  attributes is still KIS, not TRAKE.

Parsing requirements:
1. Resolve every pronoun or elliptical reference into a standalone phrase.
2. Extract searchable people, objects, actions, scenes, text, and their attributes
   into entities. Set needs_ocr only when visible text is needed, and needs_asr only
   when speech or audio is needed.
3. Produce concise Vietnamese and English query_variants. visual_description must
   contain visual evidence only; do not put a guessed QA answer in it.
4. For QA, keep the retrieval context separate from the question and infer the
   expected answer type. Retrieval queries must not leak a guessed answer.
5. For TRAKE, create events in exact chronological order. Each event description
   must stand alone and each semantic_keyframe must define the precise instant to
   align (onset, contact, peak, completion, etc.). temporal_constraints must state
   the chronological relations, such as "event 1 before event 2".
6. Return one flat object. KIS has question=null, answer_type=null, and empty
   events/temporal_constraints. QA has question and answer_type, with empty
   events/temporal_constraints. TRAKE has question=null, answer_type=null, at
   least two ordered events, and temporal_constraints.
7. query_id and raw_query must exactly reproduce the supplied values.
""".strip()


def build_user_prompt(
    query_id: str,
    raw_text: str,
    validation_feedback: str | None = None,
) -> str:
    prompt = (
        "Analyze and route this AIC 2026 query:\n\n"
        f"<query_id>{query_id}</query_id>\n"
        f"<query>\n{raw_text}\n</query>"
    )
    if validation_feedback:
        prompt += (
            "\n\nYour previous output failed application validation. Correct the semantic "
            "structure while analyzing the same query. Validation feedback:\n"
            f"{validation_feedback}"
        )
    return prompt

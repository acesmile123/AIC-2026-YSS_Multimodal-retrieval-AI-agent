def apply_object_filter(candidates, entities, object_lookup):
    object_entities = [
        e for e in entities
        if e.get("type") == "object"
    ]

    if not object_entities:
        return candidates

    filtered = []

    for candidate in candidates:
        metadata = object_lookup.get(
            candidate["video_id"],
            candidate["frame_id"]
        )

        if metadata is None:
            continue

        detected_objects = {
            str(x).lower()
            for x in metadata.get(
                "detection_class_entities",
                []
            )
        }

        matched = True

        for entity in object_entities:
            target = entity.get(
                "value",
                ""
            ).lower()

            if target not in detected_objects:
                matched = False
                break

        if matched:
            filtered.append(candidate)

    print(
        f"[Object Filter] "
        f"{len(candidates)} -> {len(filtered)}"
    )

    return filtered
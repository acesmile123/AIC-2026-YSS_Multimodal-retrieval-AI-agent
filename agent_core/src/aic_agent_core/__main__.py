from __future__ import annotations

import json
import sys

from .router import route_query


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("AIC 2026 Query Router (Ctrl+C để thoát)")
    raw_query = input("Nhập truy vấn: ").strip()
    query_id = input("Query ID (Enter để tự sinh): ").strip() or None
    task_type, structured_query = route_query(raw_query, query_id=query_id)
    print(f"\nTask type: {task_type.value}")
    print(json.dumps(structured_query.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

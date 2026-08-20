# Milvus runtime regression fix

V8 initialized `ClipRetriever()` immediately inside `SharedKIS.__init__`, so the QA process could fail before a Milvus availability check/reconnect path had a chance to run.

V9 adds `milvus_preflight.py` and calls it before `ClipRetriever()` is constructed.

Behavior:
1. Reuse an already-running Milvus instance.
2. By default, try to start a known stopped container (`milvus-standalone`, `milvus_standalone`, `milvus`) using Docker.
3. If a nearby compose file is available, try `docker compose up -d`.
4. Wait for `localhost:19530` to accept TCP connections.
5. If still unavailable, raise an actionable error instead of silently running QA without KIS retrieval.

Environment variables:
- `MILVUS_URI` (default `http://localhost:19530`)
- `QA_MILVUS_AUTO_START=true|false` (default `true`)
- `QA_MILVUS_PROMPT=true|false` (default `false`)
- `MILVUS_CONTAINER_NAME=<container>`
- `MILVUS_COMPOSE_FILE=<compose yaml>`

This is deliberately isolated from the Q&A logic. It restores runtime behavior without changing KIS retrieval semantics.

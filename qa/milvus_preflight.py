from __future__ import annotations

import os
import socket
import subprocess
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MilvusPreflightResult:
    available: bool
    started_container: bool = False
    container_name: str | None = None
    message: str = ""


def _host_port() -> tuple[str, int]:
    uri = os.environ.get("MILVUS_URI", "http://localhost:19530").strip()
    if "://" in uri:
        uri = uri.split("://", 1)[1]
    host, _, port_text = uri.partition(":")
    return host or "localhost", int(port_text or 19530)


def is_milvus_available(timeout: float = 0.7) -> bool:
    host, port = _host_port()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run(cmd: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _candidate_container_names() -> list[str]:
    configured = os.environ.get("MILVUS_CONTAINER_NAME", "").strip()
    names = [configured] if configured else []
    # Common names only; do not start arbitrary containers.
    names.extend(["milvus-standalone", "milvus_standalone", "milvus"])
    out: list[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _docker_container_exists(name: str) -> tuple[bool, bool]:
    """Return (exists, running)."""
    if not name:
        return False, False
    try:
        ps = _run(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False, False
    if ps.returncode != 0:
        return False, False
    return True, ps.stdout.strip().lower() == "true"


def _try_compose_start() -> tuple[bool, str | None]:
    compose_candidates = []
    env_file = os.environ.get("MILVUS_COMPOSE_FILE", "").strip()
    if env_file:
        compose_candidates.append(Path(env_file))
    # Search only near the project, not the entire filesystem.
    here = Path(__file__).resolve().parents[1]
    for p in (here, here.parent, Path.cwd()):
        for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            f = p / name
            if f.exists():
                compose_candidates.append(f)
    seen = set()
    for compose_file in compose_candidates:
        try:
            real = compose_file.resolve()
        except OSError:
            real = compose_file
        if str(real) in seen:
            continue
        seen.add(str(real))
        try:
            result = _run(["docker", "compose", "-f", str(real), "up", "-d"], timeout=45)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return True, str(real)
    return False, None


def ensure_milvus_available(
    *,
    timeout: float = 0.7,
    startup_wait: float = 25.0,
    auto_start: Optional[bool] = None,
    prompt: Optional[bool] = None,
) -> MilvusPreflightResult:
    """Restore the old friendly Milvus preflight without coupling QA to Docker.

    Order:
      1) use already-running Milvus;
      2) optionally ask/auto-start a known Milvus container or local compose;
      3) wait for the TCP endpoint;
      4) fail with an actionable message if still unavailable.

    Environment controls:
      QA_MILVUS_AUTO_START=true/false (default true)
      QA_MILVUS_PROMPT=true/false (default false for non-interactive safety)
      MILVUS_CONTAINER_NAME=<known container name>
      MILVUS_COMPOSE_FILE=<compose yaml>
    """
    if is_milvus_available(timeout=timeout):
        return MilvusPreflightResult(True, message="Milvus available.")

    if auto_start is None:
        auto_start = os.environ.get("QA_MILVUS_AUTO_START", "true").strip().lower() in {"1", "true", "yes", "on"}
    if prompt is None:
        prompt = os.environ.get("QA_MILVUS_PROMPT", "false").strip().lower() in {"1", "true", "yes", "on"}

    approved = bool(auto_start)
    if prompt and not auto_start and sys.stdin.isatty():
        try:
            approved = input("Milvus is unavailable. Start the configured Milvus Docker service? [Y/n] ").strip().lower() not in {"n", "no"}
        except EOFError:
            approved = False

    if not approved:
        return MilvusPreflightResult(
            False,
            message="Milvus is unavailable at localhost:19530. Start Milvus/Docker, then retry.",
        )

    for name in _candidate_container_names():
        exists, running = _docker_container_exists(name)
        if not exists:
            continue
        if not running:
            try:
                result = _run(["docker", "start", name], timeout=15)
            except (OSError, subprocess.SubprocessError) as exc:
                result = None
                last_error = repr(exc)
            else:
                last_error = result.stderr.strip() or result.stdout.strip()
            if result is not None and result.returncode == 0:
                started = True
            else:
                started = False
        else:
            started = False
        deadline = time.monotonic() + max(1.0, startup_wait)
        while time.monotonic() < deadline:
            if is_milvus_available(timeout=timeout):
                suffix = "started Docker container" if started else "reconnected to Docker container"
                return MilvusPreflightResult(True, started, name, f"Milvus available; {suffix}: {name}.")
            time.sleep(0.5)
        # Try the next known container only if this one did not expose the port.

    started_compose, compose_file = _try_compose_start()
    if started_compose:
        deadline = time.monotonic() + max(1.0, startup_wait)
        while time.monotonic() < deadline:
            if is_milvus_available(timeout=timeout):
                return MilvusPreflightResult(True, True, None, f"Milvus available; Docker Compose started: {compose_file}.")
            time.sleep(0.5)

    return MilvusPreflightResult(
        False,
        message=(
            "Milvus is unavailable at localhost:19530. QA did not silently continue without KIS. "
            "Start your Milvus Docker service (or set MILVUS_CONTAINER_NAME / MILVUS_COMPOSE_FILE), then retry."
        ),
    )

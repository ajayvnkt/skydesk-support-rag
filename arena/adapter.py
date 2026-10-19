"""Agent Arena adapter: runs the real SkyDesk pipeline via its CLI in the project venv.

Arena's venv has no openai/langchain, so the bot runs as a subprocess. The route decision and
the retrieved docs become synthetic tool calls so scenarios can check them deterministically.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agent_arena.adapters import AdapterResult

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT / ".venv" / "bin" / "python"


def ask(message: str, *, context: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None, **_: Any) -> AdapterResult:
    try:
        proc = subprocess.run([str(PYTHON), "-m", "rag", "ask", message, "--json"], cwd=PROJECT,
                              capture_output=True, text=True, timeout=300)
    except Exception as exc:  # timeout, missing venv
        return AdapterResult(answer="", error=f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        return AdapterResult(answer="", error=f"exit {proc.returncode}: {proc.stderr[-800:]}")
    try:
        r = json.loads(proc.stdout)
    except ValueError:
        return AdapterResult(answer="", error=f"non-JSON stdout: {proc.stdout[-300:]}")
    calls = [{"name": "route", "args": {"query": message},
              "result": {"route": r.get("route"), "confidence": r.get("confidence"),
                         "reason": r.get("reason")}}]
    if r.get("chunks"):  # the guard path escalates before retrieval, so there is no search to show
        calls.append({"name": "search_kb", "args": {"query": message},
                      "result": {"citations": r.get("citations", []),
                                 "docs": [c.get("doc_id") for c in r["chunks"]]}})
    return AdapterResult(answer=r.get("answer", ""), tool_calls=calls,
                         cost_usd=r.get("cost_usd"), raw=r)

"""网页里浏览跑分：runs/ 下每次跑分的汇总、每个 trial 的结果，以及 agentrt.jsonl 整理成的时间线。

目录结构是 run_tb2.sh 写的：<runs>/<run>/jobs/<job>/<trial>/{result.json, agent/agentrt.jsonl, verifier/test-stdout.txt}。
只读，不改任何文件；路径都限制在 runs 目录里面。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..trace import load
from .report import classify, load_trials, summarize

CLIP = 4000   # 单条内容最多给页面多少字符：regex-chess 一次思考就有几万字


def _clip(s, n: int = CLIP) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"\n…（还有 {len(s) - n} 字符）"


def _inside(root: Path, rel: str) -> Optional[Path]:
    root = root.resolve()
    p = (root / rel).resolve()
    return p if p != root and p.is_relative_to(root) and p.exists() else None


def _seconds(a: Optional[str], b: Optional[str]) -> Optional[float]:
    try:
        iso = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731  3.10 不认结尾的 Z
        return round((iso(b) - iso(a)).total_seconds(), 1)
    except (AttributeError, TypeError, ValueError):
        return None


def _model(r: dict) -> str:
    return ((r.get("config") or {}).get("agent") or {}).get("model_name") or ""


def list_runs(root: Path) -> list[dict]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        trials = load_trials([d])
        if not trials:
            continue
        s = summarize(trials)
        arms = [{"name": n, "agent": n.split("[")[0].split("{")[0], "model": n.split("[")[1].split("]")[0] if "[" in n else "",
                 **{k: a[k] for k in ("valid", "success", "success_rate", "agent_timeouts", "infra_failures", "cost_usd")}}
                for n, a in s["arms"].items()]
        out.append({"name": d.name, "mtime": d.stat().st_mtime, "trials": len(trials), "arms": arms,
                    "broken_tasks": s["broken_tasks"]})
    return out


def run_detail(root: Path, name: str) -> Optional[dict]:
    d = _inside(root, name)
    if d is None or d.parent != root.resolve():
        return None
    rows = []
    for f in sorted(d.rglob("result.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "trial_name" not in r or "task_name" not in r:
            continue
        outcome, reward = classify(r)
        ar = r.get("agent_result") or {}
        meta = ar.get("metadata") or {}
        ex = r.get("agent_execution") or {}
        rows.append({
            "id": str(f.parent.relative_to(d)), "task": r["task_name"].split("/")[-1],
            "agent": (r.get("agent_info") or {}).get("name") or "?", "model": _model(r),
            "outcome": outcome, "reward": reward, "started": r.get("started_at"),
            "agent_seconds": _seconds(ex.get("started_at"), ex.get("finished_at")),
            "time_budget": meta.get("time_budget"), "turns": meta.get("turns"), "status": meta.get("status"),
            "cost_usd": ar.get("cost_usd"), "input_tokens": ar.get("n_input_tokens"),
            "cache_tokens": ar.get("n_cache_tokens"), "output_tokens": ar.get("n_output_tokens"),
            "exception": (r.get("exception_info") or {}).get("exception_type"),
            "has_log": (f.parent / "agent" / "agentrt.jsonl").exists(),
        })
    rows.sort(key=lambda x: (x["agent"] == "oracle", x["agent"], x["task"], x["started"] or ""))
    report = d / "report.md"
    return {"name": d.name, "trials": rows, "report": report.read_text() if report.exists() else ""}


def timeline_item(e: dict, t0: float) -> Optional[dict]:
    """一条 agentrt 事件 → 页面上的一条记录（at 是相对会话开始的秒数）；页面不展示的事件返回 None。
    网页对话的实时轨迹和跑分页的回放都用它，两边看到的是同一套东西。"""
    t, at = e.get("type"), round(e.get("ts", t0) - t0, 1)
    if t == "session_start":
        sb = e.get("sandbox") or {}
        return {"kind": "start", "at": at, "task": _clip(e.get("task", "")), "model": e.get("model"),
                "budget": round(e["deadline"] - e.get("ts", t0)) if e.get("deadline") else None,
                "params": e.get("llm_params") or {}, "tools": e.get("tools") or [], "mode": e.get("mode"),
                "sandbox": sb.get("kind"), "workspace": e.get("workspace")}
    if t == "turn_start":
        return {"kind": "user", "at": at, "text": e.get("text", "")}
    if t == "message":
        m = e.get("message") or {}
        if m.get("role") == "assistant":
            return {"kind": "assistant", "at": at, "content": _clip(m.get("content") or ""),
                    "reasoning": _clip(m.get("reasoning") or m.get("reasoning_content") or "")}
        if m.get("role") == "user" and m.get("kind") == "harness":
            return {"kind": "harness", "at": at, "text": _clip(m.get("content") or "")}
        return None
    if t == "span" and e.get("kind") == "llm":
        return {"kind": "llm", "at": at, "serial": e.get("serial"), "seconds": e.get("seconds"),
                "prompt": e.get("prompt_tokens"), "cached": e.get("cached_tokens"),
                "completion": e.get("completion_tokens"), "reasoning": e.get("reasoning_tokens"),
                "finish": e.get("finish_reason"), "think": e.get("think", True), "prefix_break": e.get("prefix_break")}
    if t == "tool_start":
        return {"kind": "tool", "at": at, "id": e.get("call_id"), "name": e.get("name"),
                "args": _clip(e.get("arguments") or {})}
    if t == "tool_result":
        return {"kind": "result", "at": at, "id": e.get("call_id"), "content": _clip(e.get("content") or ""),
                "error": bool(e.get("is_error")), "denied": e.get("behavior") == "deny", "seconds": e.get("seconds")}
    if t == "dropped_reply":
        return {"kind": "dropped", "at": at, "reasoning": _clip(e.get("reasoning") or "")}
    if t == "compact":
        return {"kind": "note", "at": at,
                "text": f"上下文压缩：{e.get('before_tokens')} → {e.get('after_tokens')} tokens（{e.get('reason')}）"}
    if t == "end":
        st = e.get("stats") or {}
        return {"kind": "end", "at": at, "status": e.get("status"), "final": _clip(e.get("final") or ""),
                "stats": {k: st.get(k) for k in ("llm_calls", "tool_calls", "prompt_tokens", "completion_tokens",
                                                 "cached_tokens", "cost_usd", "wall_seconds")}}
    if t in ("error", "kill_running_failed", "sandbox_lost", "resume", "provision"):
        rest = {k: v for k, v in e.items() if k not in ("seq", "ts", "type")}
        return {"kind": "note", "at": at, "text": f"{t}: {_clip(rest, 500)}"}
    return None


def timeline(events: list[dict]) -> list[dict]:
    t0 = events[0].get("ts", 0) if events else 0
    return [x for x in (timeline_item(e, t0) for e in events) if x]


def trial_detail(root: Path, run: str, trial: str) -> Optional[dict]:
    d = _inside(root, run)
    t = _inside(d, trial) if d else None
    if t is None or not (t / "result.json").is_file():
        return None
    r = json.loads((t / "result.json").read_text())
    log, stdout = t / "agent" / "agentrt.jsonl", t / "verifier" / "test-stdout.txt"
    outcome, reward = classify(r)
    return {
        "task": r.get("task_name", "").split("/")[-1], "outcome": outcome, "reward": reward,
        "exception": r.get("exception_info"), "agent_result": r.get("agent_result"),
        "kwargs": ((r.get("config") or {}).get("agent") or {}).get("kwargs") or {},
        "verifier": stdout.read_text(errors="replace")[-CLIP:] if stdout.exists() else "",
        "timeline": timeline(load(log)) if log.exists() else [],
    }

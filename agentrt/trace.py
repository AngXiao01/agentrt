"""从事件日志生成运行摘要和时间线（相当于一个最小的 tracing 视图）。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[k]


def load(path) -> list[dict]:
    out = []
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def summarize(events: list[dict]) -> dict:
    llm = [e for e in events if e.get("type") == "span" and e.get("kind") == "llm"]
    results = [e for e in events if e.get("type") == "tool_result"]
    compacts = [e for e in events if e.get("type") == "compact"]
    end = next((e for e in reversed(events) if e.get("type") == "end"), None)
    start = next((e for e in events if e.get("type") == "session_start"), {})
    lat = [e["seconds"] for e in llm]
    prompt = sum(int(e.get("prompt_tokens") or 0) for e in llm)
    cached = sum(int(e.get("cached_tokens") or 0) for e in llm)
    return {
        "task": start.get("task", ""),
        "model": start.get("model", ""),
        "mode": start.get("mode", ""),
        "status": end["status"] if end else ("error" if any(e.get("type") == "error" for e in events)
                                             else "incomplete"),
        "wall_seconds": round(events[-1]["ts"] - events[0]["ts"], 2) if events else 0,
        "llm_calls": len(llm),
        "llm_seconds_total": round(sum(lat), 2),
        "llm_p50": round(_pct(lat, 0.5), 2),
        "llm_p95": round(_pct(lat, 0.95), 2),
        "prompt_tokens": prompt,
        "completion_tokens": sum(int(e.get("completion_tokens") or 0) for e in llm),
        "cached_tokens": cached,
        "cache_hit_rate": round(cached / prompt, 3) if prompt else None,
        "reasoning_tokens": sum(int(e.get("reasoning_tokens") or 0) for e in llm),
        "cache_breaks": sum(1 for e in llm if e.get("prefix_break") is not None),
        "provisions": [(e.get("kind"), e.get("seconds")) for e in events if e.get("type") == "provision"],
        "sandbox_lost": sum(1 for e in events if e.get("type") == "sandbox_lost"),
        "tool_calls": dict(Counter(e["name"] for e in results)),
        "tool_errors": sum(1 for e in results if e.get("is_error")),
        "denials": sum(1 for e in results if e.get("behavior") == "deny"),
        "compactions": [(c["reason"], c["before_tokens"], c["after_tokens"]) for c in compacts],
        "approvals": sum(1 for e in events if e.get("type") == "permission" and e.get("approved")),
        "rejections": sum(1 for e in events if e.get("type") == "permission" and not e.get("approved")),
        "human_wait_seconds": round(sum(e.get("wait_seconds", 0) for e in events if e.get("type") == "permission"), 2),
        "resumes": sum(1 for e in events if e.get("type") == "resume"),
        "final": (end or {}).get("final", ""),
    }


def _short(obj, n=90) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"


def timeline(events: list[dict]) -> str:
    if not events:
        return "（空）"
    t0 = events[0]["ts"]
    lines = []
    for e in events:
        t = f"+{e['ts'] - t0:7.2f}s"
        typ = e.get("type")
        if typ == "span":
            brk = f"  前缀在第 {e['prefix_break']} 条变了" if e.get("prefix_break") is not None else ""
            think = f" think={e['reasoning_tokens']}" if e.get("reasoning_tokens") else ""
            lines.append(f"{t}  llm#{e.get('serial')}  {e['seconds']:.2f}s  prompt={e.get('prompt_tokens')} "
                         f"cached={e.get('cached_tokens')} out={e.get('completion_tokens')}{think} "
                         f"calls={e.get('n_calls')}{brk}")
        elif typ == "provision":
            lines.append(f"{t}  [provision] {e.get('kind')} {e.get('image') or ''} {e.get('container') or ''} "
                         f"{e.get('seconds')}s（第 {e.get('n')} 个）")
        elif typ == "sandbox_lost":
            lines.append(f"{t}  [sandbox_lost] {_short(e.get('error'), 80)}")
        elif typ == "tool_start":
            lines.append(f"{t}  → {e['name']} {_short(e['arguments'])}  [{e.get('decision')}]")
        elif typ == "tool_result":
            flag = "DENY " if e.get("behavior") == "deny" else ("ERR  " if e.get("is_error") else "")
            lines.append(f"{t}  ← {e['name']} {flag}{_short(e['content'])}")
        elif typ == "permission":
            who = "已批准" if e.get("approved") else "已拒绝"
            lines.append(f"{t}  ? {e['name']} 需要确认（{_short(e.get('reason'), 40)}）→ {who}，等待 {e.get('wait_seconds')}s")
        elif typ == "subagent_start":
            lines.append(f"{t}  ⤷ 子 agent #{e['n']} 开始：{_short(e.get('task'), 60)}  (日志 {e.get('log')})")
        elif typ == "subagent_end":
            lines.append(f"{t}  ⤶ 子 agent #{e['n']} 结束 {e.get('status')}，{e.get('tool_calls')} 次工具，{e.get('tokens')} tokens")
        elif typ == "turn_start":
            lines.append(f"{t}  ── 新一轮：{_short(e.get('text', ''), 60)}")
        elif typ == "compact":
            nread, nmod = len(e.get("read_files") or []), len(e.get("modified_files") or [])
            files = f"  read {nread} / modified {nmod}" if nread or nmod else ""
            kept = f"  keep@{e['first_kept_id']}" if e.get("first_kept_id") else ""
            lines.append(f"{t}  [compact:{e['reason']}] {e['before_tokens']} → {e['after_tokens']} tokens{kept}{files}")
        elif typ == "message" and e["message"]["role"] == "user":
            lines.append(f"{t}  user: {_short(e['message']['content'])}")
        elif typ == "message" and e["message"]["role"] == "assistant" and not e["message"].get("tool_calls"):
            lines.append(f"{t}  assistant: {_short(e['message']['content'])}")
        elif typ in ("session_start", "resume", "error", "end"):
            detail = {k: v for k, v in e.items() if k in ("mode", "model", "status", "error", "prev_turns")}
            lines.append(f"{t}  [{typ}] {_short(detail)}")
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    rows = [
        ("状态", s["status"]), ("模型", s["model"]), ("模式", s["mode"]),
        ("总耗时", f"{s['wall_seconds']}s"),
        ("模型调用", f"{s['llm_calls']} 次，合计 {s['llm_seconds_total']}s，p50 {s['llm_p50']}s / p95 {s['llm_p95']}s"),
        ("tokens", f"prompt {s['prompt_tokens']} / out {s['completion_tokens']} / cached {s['cached_tokens']}"
                   + (f"（命中率 {s['cache_hit_rate']:.0%}）" if s["cache_hit_rate"] is not None else "")
                   + (f" / 思考 {s['reasoning_tokens']}" if s.get("reasoning_tokens") else "")),
        ("前缀改写", f"{s.get('cache_breaks', 0)} 次（压缩、改 system、跨轮丢 reasoning 都会让 KV cache 失效）"),
        ("工具调用", ", ".join(f"{k}×{v}" for k, v in s["tool_calls"].items()) or "无"),
        ("工具报错 / 权限拒绝", f"{s['tool_errors']} / {s['denials']}"),
        ("人工确认", f"批准 {s['approvals']} / 拒绝 {s['rejections']}，共等待 {s['human_wait_seconds']}s"),
        ("压缩", ", ".join(f"{r}:{a}→{b}" for r, a, b in s["compactions"]) or "无"),
    ]
    if s.get("provisions") or s.get("sandbox_lost"):
        rows.append(("sandbox", f"provision {len(s['provisions'])} 次"
                                f"（{', '.join(f'{k} {sec}s' for k, sec in s['provisions'])}），丢失 {s['sandbox_lost']} 次"))
    return "\n".join(f"{k:<14}{v}" for k, v in rows)

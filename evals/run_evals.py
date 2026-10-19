"""跑评测：每个 (配置, 任务) 重复 k 次，输出成功率、pass^k、轮数、延迟、token、缓存命中、拒绝次数。

    python -m evals.run_evals --k 3                         # 默认 baseline，全部任务
    python -m evals.run_evals --variants baseline,volatile_prefix --k 3
    python -m evals.run_evals --tasks injection,escape --variants baseline --k 5

无人值守：权限模式 auto + policy 审批器（模式层面的例行确认放行，hook 标记的危险操作拒绝）。
每次试验都在全新的临时目录里跑。
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from agentrt.cli import api_key
from agentrt.context import ContextManager
from agentrt.llm import OpenAICompatLLM
from agentrt.loop import Agent
from agentrt.permissions import PermissionEngine, policy_approver
from agentrt.session import EventLog

from .tasks import TASKS, TASKS_BY_ID

VARIANTS = {
    "baseline": {},
    "no_compact": {"compact": False},
    "tight_ctx": {"max_context": 6000},          # 逼出压缩，看压缩对成功率的影响
    "volatile_prefix": {"volatile_prefix": True},  # 前缀每轮带时间戳：对照 KV cache 命中率
    "native": {"native_tools": True},              # 需要 vLLM 开 tool parser
    "no_hooks": {"hooks": []},                     # 去掉 hooks，看安全任务的差异
}


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def run_trial(task, variant: dict, args, out_dir: Path) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"agentrt_{task.id}_"))
    ws = root / "ws"
    ws.mkdir()
    task.setup(root, ws)
    llm = OpenAICompatLLM(args.base_url, model=args.model, api_key=api_key(),
                          native_tools=variant.get("native_tools", False),
                          merge_system=args.merge_system, temperature=args.temperature)
    perms = PermissionEngine("auto", hooks=variant["hooks"]) if "hooks" in variant else PermissionEngine("auto")
    log = EventLog.new(out_dir / "sessions")
    agent = Agent(llm, ws, log, permissions=perms, approver=policy_approver,
                  context=ContextManager(llm, max_context_tokens=llm.context_budget(
                                             variant.get("max_context", args.max_context)),
                                         enabled=variant.get("compact", True)),
                  max_turns=args.max_turns, volatile_prefix=variant.get("volatile_prefix", False))
    t0 = time.time()
    try:
        res = agent.run(task.prompt)
        ok, why = task.check(root, ws, res)
        rec = asdict(res)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"运行异常：{type(e).__name__}: {e}"
        rec = {"status": "crash", "error": traceback.format_exc()[-800:]}
    rec.update({"task": task.id, "success": bool(ok), "check": why, "session": str(log.path),
                "wall_seconds": round(time.time() - t0, 2)})
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        rec["root"] = str(root)
    return rec


def aggregate(recs: list[dict], k: int) -> dict:
    by_task: dict[str, list[dict]] = {}
    for r in recs:
        by_task.setdefault(r["task"], []).append(r)
    walls = [r["wall_seconds"] for r in recs]
    prompt = sum(r.get("prompt_tokens", 0) for r in recs)
    cached = sum(r.get("cached_tokens", 0) for r in recs)
    return {
        "trials": len(recs),
        "success_rate": round(sum(r["success"] for r in recs) / max(1, len(recs)), 3),
        "pass^k": round(sum(all(x["success"] for x in v) for v in by_task.values()) / max(1, len(by_task)), 3),
        "avg_turns": round(statistics.mean(r.get("turns", 0) for r in recs), 2) if recs else 0,
        "avg_tool_calls": round(statistics.mean(r.get("tool_calls", 0) for r in recs), 2) if recs else 0,
        "wall_p50": round(_pct(walls, 0.5), 1),
        "wall_p95": round(_pct(walls, 0.95), 1),
        "avg_prompt_tokens": round(prompt / max(1, len(recs))),
        "cache_hit_rate": round(cached / prompt, 3) if prompt else None,
        "denials": sum(r.get("denials", 0) for r in recs),
        "parse_errors": sum(r.get("parse_errors", 0) for r in recs),
        "compactions": sum(r.get("compactions", 0) for r in recs),
        "per_task": {t: f"{sum(x['success'] for x in v)}/{len(v)}" for t, v in by_task.items()},
    }


def to_markdown(summary: dict, k: int) -> str:
    cols = ["success_rate", "pass^k", "avg_turns", "avg_tool_calls", "wall_p50", "wall_p95",
            "avg_prompt_tokens", "cache_hit_rate", "denials", "parse_errors", "compactions"]
    lines = [f"k = {k}\n", "| variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in summary.items():
        lines.append(f"| {name} | " + " | ".join(str(s[c]) for c in cols) + " |")
    tasks = sorted({t for s in summary.values() for t in s["per_task"]})
    lines += ["", "| task | " + " | ".join(summary) + " |", "|---" * (len(summary) + 1) + "|"]
    for t in tasks:
        lines.append(f"| {t} | " + " | ".join(s["per_task"].get(t, "-") for s in summary.values()) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None)
    p.add_argument("--merge-system", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tasks", default="all")
    p.add_argument("--variants", default="baseline")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--max-context", type=int, default=24000)
    p.add_argument("--out", type=Path, default=Path("evals_out") / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--keep", action="store_true", help="保留每次试验的临时目录，便于排查")
    args = p.parse_args(argv)

    tasks = TASKS if args.tasks == "all" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    variants = args.variants.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (args.out / "trials.jsonl").open("a") as f:
        for vname in variants:
            recs = []
            for task in tasks:
                for i in range(args.k):
                    r = run_trial(task, VARIANTS[vname], args, args.out / vname)
                    r["variant"] = vname
                    recs.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{vname}] {task.id} #{i + 1}: {'PASS' if r['success'] else 'FAIL'}  "
                          f"turns={r.get('turns')} {r['wall_seconds']}s  {r['check'][:80]}")
            summary[vname] = aggregate(recs, args.k)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    md = to_markdown(summary, args.k)
    (args.out / "report.md").write_text(md)
    print("\n" + md)
    print(f"\n结果目录：{args.out}")


if __name__ == "__main__":
    main()

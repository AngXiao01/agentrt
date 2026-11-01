"""跑评测：每个 (配置, 任务) 重复 k 次，输出成功率、pass^k、轮数、延迟、token、缓存命中、拒绝次数。

    python -m evals.run_evals --k 3                         # 默认 baseline，全部任务
    python -m evals.run_evals --variants baseline,volatile_prefix --k 3
    python -m evals.run_evals --tasks injection,escape --variants baseline --k 5
    python -m evals.run_evals --sandbox docker:python:3.12-slim --variants baseline,reasoning_none --k 5 --parallel 4
    python -m evals.run_evals --resume evals_out/20261101-120000          # 接着跑没跑完的试验

无人值守：权限模式 auto + policy 审批器（模式层面的例行确认放行，hook 标记的危险操作拒绝）。
每次试验都在全新的临时目录里跑。--sandbox 给定时换成 coding 工具集（coding_tools.py），工具在 sandbox 里执行；
docker 时工作目录挂载到容器的 /workspace，判定仍在宿主机上看文件。
报告：成功率带 95% Wilson 区间；第一个配置之外的每个配置都和第一个做配对比较（McNemar + 按任务 bootstrap）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from typing import Optional

from agentrt.cli import api_key
from agentrt.coding_tools import build_coding_agent
from agentrt.context import ContextManager
from agentrt.llm import OpenAICompatLLM
from agentrt.loop import Agent
from agentrt.permissions import PermissionEngine, policy_approver
from agentrt.sandboxes import make_sandbox
from agentrt.session import EventLog

from .stats import bootstrap_diff, mcnemar, pass_at_k, pass_hat_k, wilson
from .tasks import TASKS, TASKS_BY_ID

VARIANTS = {
    "baseline": {},
    "no_compact": {"compact": False},
    "tight_ctx": {"max_context": 6000},          # 逼出压缩，看压缩对成功率的影响
    "volatile_prefix": {"volatile_prefix": True},  # 前缀每轮带时间戳：对照 KV cache 命中率
    "native": {"native_tools": True},              # 需要 vLLM 开 tool parser
    "no_hooks": {"hooks": []},                     # 去掉 hooks，看安全任务的差异
    "reasoning_none": {"reasoning_replay": "none"},  # 思考模型：工具调用链里不回传 reasoning
    "reasoning_all": {"reasoning_replay": "all"},    # 思考模型：所有历史 reasoning 都回传
    "bwrap": {"sandbox": "bwrap"},                 # coding 工具集 + Linux bubblewrap
    "docker": {"sandbox": "docker:python:3.12-slim"},  # coding 工具集 + 容器
}


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def make_llm(args, variant: dict):
    if getattr(args, "llm_factory", None):   # 测试注入脚本化模型
        return args.llm_factory(variant)
    return OpenAICompatLLM(args.base_url, model=args.model, api_key=api_key(),
                           native_tools=variant.get("native_tools", args.native_tools),
                           merge_system=args.merge_system, temperature=args.temperature, max_tokens=args.max_tokens,
                           reasoning_replay=variant.get("reasoning_replay", args.reasoning_replay),
                           extra_body=json.loads(args.extra_body) if args.extra_body else None)


def run_trial(task, variant: dict, args, out_dir: Path, i: int = 0) -> dict:
    root = Path(tempfile.mkdtemp(prefix=f"agentrt_{task.id}_"))
    ws = root / "ws"
    ws.mkdir()
    task.setup(root, ws)
    log = EventLog.new(out_dir / "sessions")
    sandbox = None
    t0 = time.time()
    try:
        llm = make_llm(args, variant)
        perms = PermissionEngine("auto", hooks=variant["hooks"]) if "hooks" in variant else PermissionEngine("auto")
        cm = ContextManager(llm, max_context_tokens=llm.context_budget(variant.get("max_context", args.max_context))
                            if hasattr(llm, "context_budget") else variant.get("max_context", args.max_context),
                            enabled=variant.get("compact", True))
        spec = variant.get("sandbox", args.sandbox)
        if spec:
            sandbox = make_sandbox(spec, ws)
            agent = build_coding_agent(llm, sandbox, log, max_turns=args.max_turns, context=cm, permissions=perms,
                                       price=args.price, volatile_prefix=variant.get("volatile_prefix", False),
                                       harness_dir=root / "harness")
        else:
            agent = Agent(llm, ws, log, permissions=perms, approver=policy_approver, context=cm,
                          max_turns=args.max_turns, volatile_prefix=variant.get("volatile_prefix", False),
                          price=args.price)
        res = agent.run(task.prompt)
        ok, why = task.check(root, ws, res)
        rec = asdict(res)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"运行异常：{type(e).__name__}: {e}"
        rec = {"status": "crash", "error": traceback.format_exc()[-800:]}
    finally:
        if sandbox is not None:
            sandbox.close()
    rec.update({"task": task.id, "trial": i, "success": bool(ok), "check": why, "session": str(log.path),
                "wall_seconds": round(time.time() - t0, 2), "sandbox": variant.get("sandbox", args.sandbox)})
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
    wins = sum(r["success"] for r in recs)
    lo, hi = wilson(wins, len(recs))
    kk = min([k] + [len(v) for v in by_task.values()]) if by_task else k

    def per_task(fn):
        return round(statistics.mean(fn(len(v), sum(x["success"] for x in v), kk) for v in by_task.values()), 3) \
            if by_task else 0.0

    return {
        "trials": len(recs),
        "success_rate": round(wins / max(1, len(recs)), 3),
        "ci95": f"{lo:.2f}-{hi:.2f}",
        "pass@k": per_task(pass_at_k),
        "pass^k": per_task(pass_hat_k),
        "avg_turns": round(statistics.mean(r.get("turns", 0) for r in recs), 2) if recs else 0,
        "avg_tool_calls": round(statistics.mean(r.get("tool_calls", 0) for r in recs), 2) if recs else 0,
        "wall_p50": round(_pct(walls, 0.5), 1),
        "wall_p95": round(_pct(walls, 0.95), 1),
        "avg_prompt_tokens": round(prompt / max(1, len(recs))),
        "cache_hit_rate": round(cached / prompt, 3) if prompt else None,
        "denials": sum(r.get("denials", 0) for r in recs),
        "parse_errors": sum(r.get("parse_errors", 0) for r in recs),
        "compactions": sum(r.get("compactions", 0) for r in recs),
        "cache_breaks": sum(r.get("cache_breaks", 0) for r in recs),
        "avg_reasoning_tokens": round(sum(r.get("reasoning_tokens", 0) for r in recs) / max(1, len(recs))),
        "cost_usd": round(sum(r.get("cost_usd", 0) for r in recs), 4),
        "crashes": sum(1 for r in recs if r.get("status") == "crash"),
        "per_task": {t: f"{sum(x['success'] for x in v)}/{len(v)}" for t, v in by_task.items()},
    }


def compare(base: list[dict], other: list[dict]) -> dict:
    """配对：同一任务的第 i 次试验对第 i 次试验。"""
    key = {(r["task"], r.get("trial", 0)): r["success"] for r in base}
    pairs = [(key[(r["task"], r.get("trial", 0))], r["success"]) for r in other
             if (r["task"], r.get("trial", 0)) in key]
    by = lambda rs: {t: [x["success"] for x in rs if x["task"] == t] for t in {x["task"] for x in rs}}  # noqa: E731
    boot = bootstrap_diff(by(base), by(other))
    out = {"pairs": len(pairs), **mcnemar(pairs)}
    if boot:
        out["diff"], out["diff_lo"], out["diff_hi"] = (round(x, 3) for x in boot)
    return out


def to_markdown(summary: dict, k: int, comparisons: Optional[dict] = None) -> str:
    cols = ["success_rate", "ci95", "pass@k", "pass^k", "avg_turns", "avg_tool_calls", "wall_p50", "wall_p95",
            "avg_prompt_tokens", "cache_hit_rate", "cache_breaks", "avg_reasoning_tokens", "cost_usd", "denials",
            "parse_errors", "compactions", "crashes"]
    lines = [f"k = {k}\n", "| variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in summary.items():
        lines.append(f"| {name} | " + " | ".join(str(s.get(c, "-")) for c in cols) + " |")
    if comparisons:
        base = next(iter(summary))
        lines += ["", f"配对比较（对照 {base}；b = 对照成功而本配置失败，c = 反过来；p 为 McNemar 精确检验）", "",
                  "| variant | pairs | b | c | p | Δ成功率 | 95% 区间（按任务 bootstrap） |", "|---|---|---|---|---|---|---|"]
        for name, c in comparisons.items():
            lines.append(f"| {name} | {c['pairs']} | {c['b']} | {c['c']} | {c['p']:.3f} | {c.get('diff', '-')} | "
                         f"{c.get('diff_lo', '-')} ~ {c.get('diff_hi', '-')} |")
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
    p.add_argument("--sandbox", default=None,
                   help="local / bwrap / docker:<镜像>：换成 coding 工具集并在 sandbox 里执行（默认用原来的本地工具）")
    p.add_argument("--native-tools", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning-replay", default="turn", choices=["none", "turn", "all"])
    p.add_argument("--extra-body", default=None, help='JSON，合并进请求体，例如 \'{"thinking": {"type": "enabled"}}\'')
    p.add_argument("--price", default=None, help='JSON，每百万 tokens 美元：\'{"input": .., "cached": .., "output": ..}\'')
    p.add_argument("--parallel", type=int, default=1, help="同时跑几个试验")
    p.add_argument("--resume", type=Path, default=None, help="接着某个结果目录跑：已有的 (配置, 任务, 第几次) 跳过")
    args = p.parse_args(argv)
    args.price = json.loads(args.price) if isinstance(args.price, str) else args.price
    return run(args)


def run(args) -> dict:
    tasks = TASKS if args.tasks == "all" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    variants = args.variants.split(",")
    if getattr(args, "resume", None):
        args.out = args.resume
    args.out.mkdir(parents=True, exist_ok=True)
    trials_path = args.out / "trials.jsonl"
    done: list[dict] = []
    if trials_path.exists():
        done = [json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()]
    seen = {(r["variant"], r["task"], r.get("trial", 0)) for r in done}
    todo = [(v, t, i) for v in variants for t in tasks for i in range(args.k) if (v, t.id, i) not in seen]
    if seen:
        print(f"已有 {len(seen)} 次试验，本次跑剩下的 {len(todo)} 次")
    lock = threading.Lock()
    recs = [r for r in done if r["variant"] in variants]

    def one(v, task, i):
        r = run_trial(task, VARIANTS[v], args, args.out / v, i)
        r["variant"] = v
        with lock, trials_path.open("a") as f:
            recs.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{v}] {task.id} #{i + 1}: {'PASS' if r['success'] else 'FAIL'}  "
                  f"turns={r.get('turns')} {r['wall_seconds']}s  {r['check'][:80]}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for fut in as_completed([pool.submit(one, v, t, i) for v, t, i in todo]):
            fut.result()

    by_v = {v: [r for r in recs if r["variant"] == v] for v in variants}
    summary = {v: aggregate(rs, args.k) for v, rs in by_v.items()}
    comparisons = {v: compare(by_v[variants[0]], by_v[v]) for v in variants[1:]}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "comparisons": comparisons},
                                                      ensure_ascii=False, indent=2))
    md = to_markdown(summary, args.k, comparisons)
    (args.out / "report.md").write_text(md)
    print("\n" + md)
    print(f"\n结果目录：{args.out}")
    return {"summary": summary, "comparisons": comparisons}


if __name__ == "__main__":
    main()

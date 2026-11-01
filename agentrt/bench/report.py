"""汇总 Harbor 的 job 目录：每个配置（agent + 模型 + 参数）的成功率、区间、pass@k / pass^k、token / 成本，
以及配置之间的配对比较。把基础设施失败和 agent 失败分开。

    python3 -m agentrt.bench.report <jobs 目录或 job 目录>... [--out report.md]

判定口径：
- 成功 = verifier 的 reward >= 1（TB2 是 0/1）。
- agent 超时算失败（它就是没在规定时间内做完）。
- 其余异常（容器起不来、镜像拉不下来、判分脚本自己挂了……）算基础设施失败：不进分母，单独列出来。
- 有 oracle 配置时，oracle 没拿满分的任务说明“在这台机器上判不了”，所有配置都把这些任务剔掉（报告里写明剔了哪些）。
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .stats import bootstrap_diff, mcnemar, pass_at_k, pass_hat_k, wilson

HIDDEN_KWARGS = {"api_base", "price"}   # 不影响行为、或者只影响记账的参数，不进配置名


def load_trials(paths: list[Path]) -> list[dict]:
    out = []
    for root in paths:
        for f in sorted(Path(root).rglob("result.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "trial_name" in r and "task_name" in r:   # job 级别的 result.json 没有这两个字段
                out.append(r)
    return out


def arm_of(r: dict) -> str:
    cfg = (r.get("config") or {}).get("agent") or {}
    name = (r.get("agent_info") or {}).get("name") or cfg.get("name") or "?"
    model = cfg.get("model_name") or ""
    kw = {k: v for k, v in (cfg.get("kwargs") or {}).items() if k not in HIDDEN_KWARGS}
    extra = ",".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return name + (f"[{model}]" if model else "") + (f"{{{extra}}}" if extra else "")


def classify(r: dict) -> tuple[str, float]:
    """返回 (outcome, reward)：outcome ∈ ok / fail / agent_timeout / infra。"""
    exc = r.get("exception_info") or {}
    et = exc.get("exception_type") or ""
    rewards = ((r.get("verifier_result") or {}).get("rewards")) or {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)) if rewards else 0.0)
    if "AgentTimeout" in et:
        return "agent_timeout", 0.0
    if et:
        return "infra", 0.0
    if not rewards:
        return "infra", 0.0
    return ("ok" if reward >= 1.0 else "fail"), reward


def summarize(trials: list[dict]) -> dict:
    arms: dict[str, list[dict]] = defaultdict(list)
    for r in trials:
        outcome, reward = classify(r)
        meta = ((r.get("agent_result") or {}).get("metadata")) or {}
        arms[arm_of(r)].append({"task": r["task_name"], "start": r.get("started_at") or "", "outcome": outcome,
                                "reward": reward, "res": r.get("agent_result") or {}, "status": meta.get("status"),
                                "exc": (r.get("exception_info") or {}).get("exception_type")})
    # oracle 判不了满分的任务：这台机器上的判分不可信，所有配置都剔掉
    broken: set[str] = set()
    for name, rs in arms.items():
        if name.startswith("oracle"):
            by = defaultdict(list)
            for x in rs:
                by[x["task"]].append(x["outcome"] == "ok")
            broken |= {t for t, v in by.items() if not all(v)}
    out = {"broken_tasks": sorted(broken), "arms": {}}
    for name, rs in arms.items():
        for i, x in enumerate(sorted(rs, key=lambda x: (x["task"], x["start"]))):
            x["attempt"] = sum(1 for y in rs if y["task"] == x["task"] and y["start"] < x["start"])
        valid = [x for x in rs if x["outcome"] != "infra" and x["task"] not in broken]
        wins = sum(x["outcome"] == "ok" for x in valid)
        lo, hi = wilson(wins, len(valid))
        by_task = defaultdict(list)
        for x in valid:
            by_task[x["task"]].append(x["outcome"] == "ok")
        k = min((len(v) for v in by_task.values()), default=1)
        res = [x["res"] for x in valid]
        cost = [x.get("cost_usd") for x in res if x.get("cost_usd") is not None]
        statuses = defaultdict(int)
        for x in valid:
            statuses[x["status"] or x["outcome"]] += 1
        out["arms"][name] = {
            "trials": len(rs), "valid": len(valid), "infra_failures": sum(x["outcome"] == "infra" for x in rs),
            "infra_types": sorted({x["exc"] for x in rs if x["outcome"] == "infra" and x["exc"]}),
            "tasks": len(by_task), "success": wins, "success_rate": round(wins / len(valid), 3) if valid else None,
            "ci95": f"{lo:.2f}-{hi:.2f}", "k": k,
            "pass@k": round(statistics.mean(pass_at_k(len(v), sum(v), k) for v in by_task.values()), 3) if by_task else None,
            "pass^k": round(statistics.mean(pass_hat_k(len(v), sum(v), k) for v in by_task.values()), 3) if by_task else None,
            "agent_timeouts": sum(x["outcome"] == "agent_timeout" for x in valid),
            "end_status": dict(statuses),
            "avg_input_tokens": round(statistics.mean(x.get("n_input_tokens") or 0 for x in res)) if res else 0,
            "cache_hit_rate": round(sum(x.get("n_cache_tokens") or 0 for x in res)
                                    / max(1, sum(x.get("n_input_tokens") or 0 for x in res)), 3),
            "avg_output_tokens": round(statistics.mean(x.get("n_output_tokens") or 0 for x in res)) if res else 0,
            "cost_usd": round(sum(cost), 4) if cost else None,
            "_valid": valid,
        }
    names = [n for n in out["arms"] if not n.startswith("oracle")]
    out["comparisons"] = {}
    if len(names) > 1:
        base = out["arms"][names[0]]["_valid"]
        key = {(x["task"], x["attempt"]): x["outcome"] == "ok" for x in base}
        for n in names[1:]:
            other = out["arms"][n]["_valid"]
            pairs = [(key[(x["task"], x["attempt"])], x["outcome"] == "ok") for x in other
                     if (x["task"], x["attempt"]) in key]
            by = lambda rs: {t: [y["outcome"] == "ok" for y in rs if y["task"] == t] for t in {y["task"] for y in rs}}  # noqa: E731
            boot = bootstrap_diff(by(base), by(other))
            c = {"vs": names[0], "pairs": len(pairs), **mcnemar(pairs)}
            if boot:
                c["diff"], c["diff_lo"], c["diff_hi"] = (round(v, 3) for v in boot)
            out["comparisons"][n] = c
    for a in out["arms"].values():
        a.pop("_valid")
    return out


def to_markdown(s: dict) -> str:
    cols = ["valid", "tasks", "k", "success_rate", "ci95", "pass@k", "pass^k", "agent_timeouts", "infra_failures",
            "avg_input_tokens", "cache_hit_rate", "avg_output_tokens", "cost_usd"]
    lines = ["| 配置 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, a in s["arms"].items():
        lines.append(f"| {name} | " + " | ".join(str(a.get(c)) for c in cols) + " |")
    if s["broken_tasks"]:
        lines += ["", f"oracle 在这台机器上没拿满分、已从所有配置剔除的任务（{len(s['broken_tasks'])} 个）：" +
                  ", ".join(s["broken_tasks"])]
    infra = {n: a["infra_types"] for n, a in s["arms"].items() if a["infra_failures"]}
    if infra:
        lines += ["", "基础设施失败（不计入成功率）：" + "；".join(f"{n}：{', '.join(t) or '无判分结果'}"
                                                        for n, t in infra.items())]
    if s["comparisons"]:
        lines += ["", "| 配置 | 对照 | pairs | b | c | p（McNemar） | Δ成功率 | 95% 区间 |", "|---|---|---|---|---|---|---|---|"]
        for n, c in s["comparisons"].items():
            lines.append(f"| {n} | {c['vs']} | {c['pairs']} | {c['b']} | {c['c']} | {c['p']:.3f} | {c.get('diff')} | "
                         f"{c.get('diff_lo')} ~ {c.get('diff_hi')} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = summarize(load_trials(a.paths))
    md = to_markdown(s)
    print(md)
    if a.out:
        a.out.write_text(md + "\n")
        a.out.with_suffix(".json").write_text(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

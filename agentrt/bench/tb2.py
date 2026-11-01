"""Terminal-Bench 2.0 任务集：固定版本、分层抽样。

    python3 -m agentrt.bench.tb2 --src ~/.cache/agentrt/terminal-bench-2 --n 30 --seed 0 --out tb2_subset

全量 89 个 × k=3 一次要跑很久、花很多钱；先按难度分层抽一个子集（按 easy / medium / hard 的比例分配名额，
每层内按 seed 打乱），子集名单写进 subset.txt，换配置时用同一个子集才能配对比较。
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO = "https://github.com/harbor-framework/terminal-bench-2.git"
COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"   # 2026-04-29，89 个任务


def tasks(src: Path) -> dict[str, str]:
    """任务名 → 难度。"""
    out = {}
    for t in sorted(p.parent for p in Path(src).glob("*/task.toml")):
        m = re.search(r'^difficulty\s*=\s*"(\w+)"', (t / "task.toml").read_text(), re.M)
        out[t.name] = m.group(1) if m else "unknown"
    return out


def select(src: Path, n: int, seed: int = 0) -> list[str]:
    all_ = tasks(src)
    if n >= len(all_):
        return sorted(all_)
    strata: dict[str, list[str]] = defaultdict(list)
    for name, d in all_.items():
        strata[d].append(name)
    rng = random.Random(seed)
    quota = {d: n * len(v) / len(all_) for d, v in strata.items()}
    take = {d: int(q) for d, q in quota.items()}
    for d in sorted(quota, key=lambda d: quota[d] - take[d], reverse=True)[:n - sum(take.values())]:
        take[d] += 1   # 最大余数法分配剩下的名额
    picked = []
    for d, names in sorted(strata.items()):
        names = sorted(names)
        rng.shuffle(names)
        picked += names[:take[d]]
    return sorted(picked)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--n", type=int, default=89)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    names = select(a.src, a.n, a.seed)
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    for name in names:
        shutil.copytree(a.src / name, a.out / name)
    (a.out / "subset.txt").write_text("\n".join(names) + "\n")
    diff = tasks(a.src)
    counts = defaultdict(int)
    for name in names:
        counts[diff[name]] += 1
    print(f"{len(names)} 个任务 → {a.out}（{', '.join(f'{d} {c}' for d, c in sorted(counts.items()))}）")


if __name__ == "__main__":
    main()

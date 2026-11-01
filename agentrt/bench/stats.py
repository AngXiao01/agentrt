"""评测统计：区间估计和配对比较。只用标准库。

为什么需要：8 个任务 × k=3 = 24 次试验，成功率差 10 个百分点完全可能是噪声。报告里只放点估计，
两个配置谁好谁坏基本是在读随机数。
- wilson：二项比例的 Wilson 区间（n 小、p 靠近 0/1 时比正态近似可靠）。
- pass_at_k / pass_hat_k：n 次试验里成功 c 次时的无偏估计。pass@k = 至少一次成功（Codex 论文的估计量），
  pass^k = k 次全成功（τ-bench 的可靠性指标）。原来的实现只在 n == k 时正确。
- mcnemar：同一批 (任务, 第 i 次) 配对，只看结论不一致的对，精确二项检验。比两组独立比例检验功效高得多。
- bootstrap_diff：按任务重采样（任务是抽样单位，试验嵌套在任务里），给成功率差一个区间。
"""
from __future__ import annotations

import math
import random
from typing import Optional


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pass_at_k(n: int, c: int, k: int) -> float:
    """n 次里 c 次成功，随机抽 k 次至少一次成功的概率：1 − C(n−c, k) / C(n, k)。"""
    if k > n:
        raise ValueError("k 不能大于 n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """n 次里 c 次成功，随机抽 k 次全部成功的概率：C(c, k) / C(n, k)。"""
    if k > n:
        raise ValueError("k 不能大于 n")
    return math.comb(c, k) / math.comb(n, k)


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """配对 (A 成功?, B 成功?)。b = A 成 B 败，c = A 败 B 成；双侧精确二项检验 p 值。"""
    b = sum(1 for a, bb in pairs if a and not bb)
    c = sum(1 for a, bb in pairs if bb and not a)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def bootstrap_diff(a: dict[str, list[bool]], b: dict[str, list[bool]], iters: int = 2000,
                   seed: int = 0) -> Optional[tuple[float, float, float]]:
    """两组按任务配对的成功率差 (B − A)：点估计和 95% 区间。按任务重采样。"""
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return None

    def rate(d, ts):
        xs = [x for t in ts for x in d[t]]
        return sum(xs) / len(xs) if xs else 0.0

    rng = random.Random(seed)
    point = rate(b, tasks) - rate(a, tasks)
    diffs = sorted(rate(b, s) - rate(a, s) for s in ([rng.choice(tasks) for _ in tasks] for _ in range(iters)))
    return (point, diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1])

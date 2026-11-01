"""跑 Terminal-Bench 2.0 之前的环境预检：逐项检查，失败的给出修法。

    python3 -m agentrt.bench.preflight --model deepseek/deepseek-chat [--api-base ...]

为什么要有：TB2 的判分脚本在判分时才从外网装依赖（apt 装 curl → 装 uv → uvx 拉 pytest），
容器没网时 oracle 也会判 0 分；Docker Hub 匿名拉取有限流（89 个任务镜像）。这两类失败如果混进成绩，
看起来就是“agent 做错了”。先把基础设施和 agent 的失败分开。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

from .harbor_agent import KEY_ENVS, _resolve_model

# TB2 判分脚本的典型依赖：apt 源、uv 安装脚本、uv / python 二进制（GitHub release）、PyPI
CONTAINER_HOSTS = ("https://deb.debian.org/debian/", "https://astral.sh/uv/install.sh",
                   "https://github.com/astral-sh/uv/releases/latest", "https://pypi.org/simple/pytest/",
                   "http://archive.ubuntu.com/ubuntu/")
PROBE_IMAGE = "python:3.12-slim"


def _row(ok: bool, name: str, detail: str = "", fix: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
    if not ok and fix:
        print(f"       修法：{fix}")
    return ok


def check_model(model: str, api_base: str | None) -> bool:
    try:
        base, mid = _resolve_model(model, api_base, None)
    except ValueError as e:
        return _row(False, "模型端点", str(e))
    key_env = next((k for k in KEY_ENVS if os.environ.get(k)), None)
    if not key_env:
        return _row(False, "模型 key", f"{', '.join(KEY_ENVS)} 都没设",
                    "在环境设置里加环境变量（不要贴进对话），新会话生效")
    req = urllib.request.Request(base.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {os.environ[key_env]}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return _row(True, "模型端点", f"{base} 可达（key 来自 {key_env}，模型 {mid}）")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return _row(False, "模型端点", f"{base} 返回 {e.code}", f"检查 {key_env} 是否有效")
        return _row(True, "模型端点", f"{base} 可达（/models 返回 {e.code}，不影响 chat）")
    except (urllib.error.URLError, OSError) as e:
        host = base.split("/")[2]
        return _row(False, "模型端点", f"{base} 连不上：{e}", f"在环境的网络设置里放行 {host}")


def check_docker() -> bool:
    if not shutil.which("docker"):
        return _row(False, "docker", "没装")
    p = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        return _row(False, "docker daemon", p.stderr.strip()[:120], "启动 dockerd（云会话里：nohup dockerd &）")
    return _row(True, "docker daemon", p.stdout.strip())


def check_pull(image: str) -> bool:
    p = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=600)
    if p.returncode == 0:
        return _row(True, "Docker Hub 拉取", image)
    msg = (p.stderr or p.stdout).strip()
    fix = ("Docker Hub 匿名拉取限流（按出口 IP 计）：设 DOCKERHUB_USERNAME / DOCKERHUB_TOKEN 后 docker login，"
           "或等额度恢复" if "429" in msg else "")
    return _row(False, "Docker Hub 拉取", msg[-200:], fix)


def check_container_network() -> bool:
    code = ("import urllib.request,sys\n"
            "bad=[]\n"
            f"for u in {list(CONTAINER_HOSTS)!r}:\n"
            "    try: urllib.request.urlopen(u, timeout=10)\n"
            "    except Exception as e:\n"
            "        if getattr(e,'code',0) in (200,301,302,403,404): continue\n"
            "        bad.append(u.split('/')[2]+': '+type(e).__name__+' '+str(e)[:60])\n"
            "print('\\n'.join(bad)); sys.exit(1 if bad else 0)")
    p = subprocess.run(["docker", "run", "--rm", PROBE_IMAGE, "python3", "-c", code],
                       capture_output=True, text=True, timeout=180)
    if p.returncode == 0:
        return _row(True, "容器外网", "判分依赖的域名都能访问")
    hosts = sorted({l.split(":")[0] for l in p.stdout.splitlines() if l.strip()})
    return _row(False, "容器外网", "\n         " + "\n         ".join(p.stdout.strip().splitlines()[:8]),
                f"TB2 判分时要从外网装依赖；放行 {', '.join(hosts) or '这些域名'}（或把网络访问调到不限制）。"
                "否则 oracle 也会判 0 分")


def check_resources() -> bool:
    total, _, free = shutil.disk_usage("/")
    cpus = os.cpu_count() or 1
    ok = free > 40e9
    _row(ok, "磁盘", f"剩余 {free / 1e9:.0f} GB（89 个任务镜像全拉下来要几十 GB；跑完一个任务删一个镜像也行）")
    _row(True, "CPU", f"{cpus} 核（TB2 任务多数要 1 核 2–4 GB；并发建议 ≤ {max(1, cpus - 1)}）")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AGENTRT_MODEL", "deepseek/deepseek-chat"))
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--image", default="alexgshaw/fix-git:20251031", help="拿一个 TB2 任务镜像试拉")
    a = ap.parse_args(argv)
    results = [check_model(a.model, a.api_base), check_docker()]
    if results[-1]:
        results += [check_pull(a.image), check_container_network()]
    check_resources()
    ok = all(results)
    print("\n" + ("可以跑。" if ok else "还不能跑：先处理上面的 FAIL。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

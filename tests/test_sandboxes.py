"""执行环境：本机 / bwrap / Docker。bwrap 和 Docker 不可用时跳过（CI 没有 docker daemon 也能跑其余测试）。"""
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from agentrt.sandboxes import (BwrapSandbox, DockerSandbox, LocalSandbox, SandboxError, SandboxLost, bwrap_available,
                               make_sandbox)

IMAGE = "python:3.12-slim"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode != 0:
            return False
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bwrap = pytest.mark.skipif(not bwrap_available(), reason="bwrap 不可用")
needs_docker = pytest.mark.skipif(not _docker_ok(), reason=f"docker daemon 或镜像 {IMAGE} 不可用")


# ---------------------------------------------------------------- local

def test_local_exec_read_write_and_confinement(tmp_path):
    sb = LocalSandbox(tmp_path / "ws")
    sb.write_bytes("d/a.txt", b"hi\n")
    assert sb.read_bytes("d/a.txt") == b"hi\n"
    r = sb.exec("cat d/a.txt; echo err >&2; exit 3", timeout=10)
    assert r.exit_code == 3 and "hi" in r.output and "err" in r.output and not r.timed_out
    with pytest.raises(SandboxError):
        sb.read_bytes("../outside.txt")
    with pytest.raises(SandboxError):
        sb.write_bytes("/etc/x", b"")


def test_local_timeout_kills_process_group(tmp_path):
    sb = LocalSandbox(tmp_path)
    marker = tmp_path / "late.txt"
    t0 = time.time()
    r = sb.exec(f"(sleep 3; echo late > {marker}) & sleep 30", timeout=1)
    assert r.timed_out and time.time() - t0 < 5
    time.sleep(3.5)
    assert not marker.exists()   # 后台子进程也被整组杀掉了


def test_make_sandbox_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        make_sandbox("vm:foo", tmp_path)


# ---------------------------------------------------------------- bwrap

@needs_bwrap
def test_bwrap_write_scope_network_and_home(tmp_path):
    ws = tmp_path / "ws"
    sb = BwrapSandbox(ws)
    assert sb.exec("echo ok > in_ws.txt && cat in_ws.txt", timeout=10).output.strip() == "ok"
    r = sb.exec("echo x > /etc/agentrt_probe", timeout=10)
    assert r.exit_code != 0 and "Read-only file system" in r.output
    r = sb.exec("echo x > ro.txt", timeout=10, mode="read-only")
    assert r.exit_code != 0 and not (ws / "ro.txt").exists()
    r = sb.exec("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)\"", timeout=10)
    assert r.exit_code != 0   # 断网
    secret = Path.home() / f".agentrt_probe_{uuid.uuid4().hex[:6]}"
    secret.write_text("TOPSECRET")
    try:
        r = sb.exec(f"cat {secret}", timeout=10)
        assert "TOPSECRET" not in r.output   # 主目录被 tmpfs 盖住了
    finally:
        secret.unlink()


@needs_bwrap
def test_bwrap_timeout_kills_everything(tmp_path):
    sb = BwrapSandbox(tmp_path / "ws")
    r = sb.exec("(sleep 2; echo late > late.txt) & sleep 30", timeout=1)
    assert r.timed_out
    time.sleep(2.5)
    assert not (tmp_path / "ws" / "late.txt").exists()   # pid 命名空间里的进程随 bwrap 一起没了


# ---------------------------------------------------------------- docker

@needs_docker
def test_docker_lazy_provision_exec_and_files(tmp_path):
    seen = []
    sb = DockerSandbox(image=IMAGE, mounts=((str(tmp_path), "/workspace"),), on_provision=seen.append)
    try:
        assert sb.cid is None and not seen   # 懒启动：没用到就不起容器
        r = sb.exec("pwd; python3 -c 'print(6*7)'", timeout=30)
        assert r.exit_code == 0 and "/workspace" in r.output and "42" in r.output
        assert len(seen) == 1 and seen[0]["kind"] == "docker"
        sb.write_bytes("bin/data.bin", bytes(range(256)))
        assert sb.read_bytes("bin/data.bin") == bytes(range(256))
        assert (tmp_path / "bin/data.bin").read_bytes() == bytes(range(256))   # 挂载：宿主机上能看到
        with pytest.raises(FileNotFoundError):
            sb.read_bytes("nope.txt")
        assert sb.exec("python3 -c \"import urllib.request as u; u.urlopen('http://1.1.1.1', timeout=3)\"",
                       timeout=20).exit_code != 0   # 默认 --network none
    finally:
        sb.close()


@needs_docker
def test_docker_timeout_kills_in_container_tree(tmp_path):
    sb = DockerSandbox(image=IMAGE, mounts=((str(tmp_path), "/workspace"),))
    try:
        r = sb.exec("(sleep 3; echo late > late.txt) & sleep 60", timeout=1)
        assert r.timed_out and r.seconds < 8
        time.sleep(3.5)
        assert not (tmp_path / "late.txt").exists()   # 容器里的后台进程也被 timeout -s KILL 整组杀掉
    finally:
        sb.close()


@needs_docker
def test_docker_lost_container_is_cattle(tmp_path):
    sb = DockerSandbox(image=IMAGE)
    try:
        sb.exec("echo hi > /workspace/f", timeout=20)
        subprocess.run(["docker", "rm", "-f", sb.cid], capture_output=True, timeout=30)
        with pytest.raises(SandboxLost):
            sb.exec("cat /workspace/f", timeout=20)
        r = sb.exec("cat /workspace/f || echo gone", timeout=20)   # 下一次调用重新 provision，旧状态不在了
        assert "gone" in r.output and sb.provisions == 2
    finally:
        sb.close()


@needs_docker
def test_docker_command_output_mentioning_not_running_is_not_lost(tmp_path):
    sb = DockerSandbox(image=IMAGE)
    try:
        r = sb.exec("echo 'service is not running'; exit 3", timeout=20)
        assert r.exit_code == 3 and sb.provisions == 1   # 命令自己的输出不会被误判成容器丢失
    finally:
        sb.close()

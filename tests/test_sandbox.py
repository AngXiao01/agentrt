import sys
from pathlib import Path

import pytest

from agentrt import sandbox
from agentrt.permissions import PermissionEngine
from agentrt.tools import SandboxedBash, ToolContext

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="需要 macOS sandbox-exec")


@pytest.mark.parametrize("cmd,ro", [
    ("ls -la", True), ("du -sh * | sort -h | tail -5", True), ("find . -name '*.log'", True),
    ("grep -r foo . | wc -l", True), ("unzip -l a.zip", True), ("tar -tf a.tar", True),
    ("find . -name x -delete", False), ("find . -exec rm {} +", False), ("sort -o out.txt a.txt", False),
    ("echo hi > a.txt", False), ("ls; rm a", False), ("ls && rm a", False), ("cat $(echo a)", False),
    ("cat `echo a`", False), ("rm a", False), ("mv a b", False), ("unzip a.zip", False), ("tar -xf a.tar", False),
    ("curl http://x", False), ("python3 -c 1", False), ("", False),
])
def test_read_only_classifier(cmd, ro):
    assert sandbox.is_read_only_command(cmd) is ro


def test_permission_uses_per_call_read_only(tmp_path):
    ctx, bash = ToolContext(tmp_path), SandboxedBash()
    assert PermissionEngine("default").decide(bash, {"command": "ls"}, ctx).behavior == "allow"
    assert PermissionEngine("default").decide(bash, {"command": "touch a"}, ctx).behavior == "ask"
    assert PermissionEngine("plan").decide(bash, {"command": "touch a"}, ctx).behavior == "deny"
    assert PermissionEngine("plan").decide(bash, {"command": "ls"}, ctx).behavior == "allow"
    assert PermissionEngine("auto").decide(bash, {"command": "touch a"}, ctx).behavior == "ask"   # 高风险：auto 也问
    assert PermissionEngine("auto").decide(bash, {"command": "ls"}, ctx).behavior == "allow"


@pytest.fixture
def sbx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello")
    probe = Path.home() / ".agentrt_sandbox_probe"
    probe.write_text("TOP-SECRET")
    yield SandboxedBash(), ToolContext(ws), probe
    probe.unlink(missing_ok=True)
    (Path.home() / ".agentrt_sandbox_write_probe").unlink(missing_ok=True)


@needs_sandbox
def test_sandbox_allows_normal_work(sbx):
    bash, ctx, _ = sbx
    assert "hello" in bash.run({"command": "cat a.txt"}, ctx)
    bash.run({"command": "mkdir out && cp a.txt out/b.txt && echo x > new.txt"}, ctx)
    assert (ctx.workspace / "out/b.txt").exists() and (ctx.workspace / "new.txt").exists()
    assert "[exit 0]" in bash.run({"command": "zip -v >/dev/null 2>&1; true"}, ctx)


@needs_sandbox
def test_sandbox_blocks_secrets_writes_outside_delete_rename_network(sbx):
    bash, ctx, probe = sbx
    out = bash.run({"command": f"cat {probe}"}, ctx)
    assert "TOP-SECRET" not in out and "Operation not permitted" in out            # 主目录里的东西读不到
    bash.run({"command": f"echo pwn > {Path.home()}/.agentrt_sandbox_write_probe"}, ctx)
    assert not (Path.home() / ".agentrt_sandbox_write_probe").exists()               # 写不到工作目录之外
    assert "Operation not permitted" in bash.run({"command": "rm a.txt"}, ctx) and (ctx.workspace / "a.txt").exists()
    assert "Operation not permitted" in bash.run({"command": "mv a.txt b.txt"}, ctx) and (ctx.workspace / "a.txt").exists()
    bash.run({"command": "python3 -c \"import os; os.remove('a.txt')\""}, ctx)
    assert (ctx.workspace / "a.txt").exists()                                       # 绕过 rm 命令也没用：拦的是系统调用
    net = bash.run({"command": "curl -s -m 3 -o /dev/null -w '%{http_code}' https://example.com"}, ctx)
    assert "200" not in net                                                          # 联不了网
    assert "Operation not permitted" in bash.run({"command": "cat $(dirname $PWD)/../../../../../etc/hosts >/dev/null; "
                                                             f"cat {probe}"}, ctx)   # 老的“$(dirname $PWD)”绕法


def test_assistant_gets_bash_only_with_sandbox(tmp_path, monkeypatch):
    from agentrt.assistant import build_assistant
    from agentrt.context import ContextManager
    from agentrt.llm import ScriptedLLM
    from agentrt.permissions import allow_all_approver
    from agentrt.session import EventLog
    llm = ScriptedLLM([])
    mk = lambda: build_assistant(llm, tmp_path, EventLog.new(tmp_path / "s"), permissions=PermissionEngine("default"),  # noqa: E731
                                 approver=allow_all_approver, context=ContextManager(llm), on_event=None)
    monkeypatch.setattr(sandbox, "available", lambda: False)
    assert "bash" not in mk().tools                                                   # 没有沙箱：失败即关闭
    monkeypatch.setattr(sandbox, "available", lambda: True)
    assert "bash" in mk().tools

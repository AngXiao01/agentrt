import json
import sys
from pathlib import Path

import pytest

from agentrt.assistant import build_assistant
from agentrt.context import ContextManager, estimate_tokens
from agentrt.llm import ScriptedLLM
from agentrt.mcp import load_mcp_tools
from agentrt.permissions import PermissionEngine, allow_all_approver
from agentrt.session import EventLog
from conftest import call

FAKE = str(Path(__file__).with_name("fake_mcp_server.py"))


@pytest.fixture
def mcp(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "fake": {"command": sys.executable, "args": [FAKE], "env": {"SECRET_TOKEN": "s3cr3t-value"}, "trusted": True},
        "web": {"type": "http", "url": "http://x"},
        "bad": {"command": "/no/such/binary"}}}))
    tools, clients, warns = load_mcp_tools(str(cfg))
    yield {t.name: t for t in tools}, warns
    for c in clients:
        c.close()


def test_mcp_tools_loaded_with_readonly_hint(mcp):
    tools, warns = mcp
    assert set(tools) == {"mcp__fake__echo", "mcp__fake__write_note", "mcp__fake__boom"}
    assert tools["mcp__fake__echo"].read_only and not tools["mcp__fake__write_note"].read_only
    assert any("web" in w for w in warns) and any("bad" in w for w in warns)     # 不支持的 / 启动失败的只告警


def test_mcp_call_and_error(mcp, tmp_path):
    from agentrt.tools import ToolContext, ToolError
    tools, _ = mcp
    ctx = ToolContext(tmp_path)
    assert tools["mcp__fake__echo"].run({"text": "你好"}, ctx) == 'echo:{"text": "你好"}'   # 忽略 server 打的非 JSON 行
    with pytest.raises(ToolError, match="出错了"):
        tools["mcp__fake__boom"].run({}, ctx)
    assert "缺少必填" in tools["mcp__fake__echo"].validate({}, ctx)
    # MCP 的 path 参数不是工作目录内的路径，不做越界检查
    assert tools["mcp__fake__write_note"].validate({"path": "/anywhere"}, ctx) is None


def test_mcp_permissions_and_env_not_leaked(mcp, tmp_path):
    tools, _ = mcp
    llm = ScriptedLLM([call("mcp__fake__echo", text="a"), call("mcp__fake__write_note", path="x"), "完成"])
    log = EventLog.new(tmp_path / "s")
    asked = []
    ag = build_assistant(llm, tmp_path, log, permissions=PermissionEngine("default"),
                         approver=lambda t, a, d: asked.append(t.name) or True, context=ContextManager(llm),
                         on_event=None, extra_tools=list(tools.values()))
    assert ag.send("go").final == "完成"
    assert asked == ["mcp__fake__write_note"]                          # 只读的直接放行，会改动的要确认
    assert "s3cr3t-value" not in log.path.read_text()                  # server 的 env 不进日志
    assert "mcp__fake__echo" in ag.system                              # 工具说明进了 system prompt


def test_delegate_returns_only_summary_and_is_read_only(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    # 脚本顺序：主 → 子（调用 find_files）→ 子（报告）→ 主（最终答复）
    llm = ScriptedLLM([call("delegate", task="查 txt"), '<tool_call>{"name":"find_files","arguments":{"name":"*.txt"}}</tool_call>',
                       "结论：只有 a.txt", "子 agent 说只有 a.txt"])
    log = EventLog.new(tmp_path / "s")
    ag = build_assistant(llm, tmp_path, log, permissions=PermissionEngine("default"), approver=allow_all_approver,
                         context=ContextManager(llm), on_event=None)
    res = ag.send("调查一下")
    assert res.final == "子 agent 说只有 a.txt"
    main_ctx = json.dumps(ag._live, ensure_ascii=False)
    assert "结论：只有 a.txt" in main_ctx and "共 1 个文件" not in main_ctx    # 主上下文里只有摘要，没有子 agent 的原始工具输出
    kinds = [e["type"] for e in log.events()]
    assert "subagent_start" in kinds and "subagent_end" in kinds
    sub = log.path.with_name(f"{log.path.stem}.sub1.jsonl")
    assert sub.exists() and "find_files" in sub.read_text()
    child_tools = next(e for e in log.events() if e["type"] == "subagent_start")["tools"]
    assert "delegate" not in child_tools and "trash" not in child_tools    # 白名单：只读，不能再派生


def test_context_count_uses_server_usage_then_estimates_the_tail():
    cm = ContextManager(ScriptedLLM([]), max_context_tokens=100000)
    msgs = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 400}]
    assert cm.count(msgs) == cm.estimate(msgs)                          # 还没有用量：回退到按发出的消息粗估
    cm.note_usage(2, prompt_tokens=5000, completion_tokens=100)         # 发出 2 条，服务端说上下文 5100
    msgs.append({"role": "assistant", "content": "a" * 40})
    msgs.append({"role": "tool", "tool_call_id": "x", "name": "grep", "content": "r" * 4000})
    assert cm.count(msgs) == 5100 + estimate_tokens(msgs[3:])           # 锚点覆盖到 assistant（第 3 条），只估算之后新增的
    cm._anchor = (len(msgs) + 5, 1)                                     # 消息被改写（比锚点短）→ 不能再信锚点
    assert cm.count(msgs) == cm.estimate(msgs)


def test_read_only_hint_only_counts_for_trusted_servers(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"stranger": {"command": sys.executable, "args": [FAKE]}}}))
    tools, clients, _ = load_mcp_tools(str(cfg))
    try:
        echo = {t.name: t for t in tools}["mcp__stranger__echo"]
        assert echo.declared_read_only and not echo.read_only          # 自称只读，但 server 没被标成可信
        llm = ScriptedLLM([call("mcp__stranger__echo", text="a"), "完成"])
        asked = []
        ag = build_assistant(llm, tmp_path, EventLog.new(tmp_path / "s"), permissions=PermissionEngine("default"),
                             approver=lambda t, a, d: asked.append(t.name) or True, context=ContextManager(llm),
                             on_event=None, extra_tools=tools)
        ag.send("go")
        assert asked == ["mcp__stranger__echo"]                         # 所以要确认
    finally:
        for c in clients:
            c.close()


def test_estimate_ignores_reasoning_that_is_not_replayed():
    cm = ContextManager(ScriptedLLM([], reasoning_replay="none"))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "reasoning": "想" * 30000}]
    assert cm.estimate(msgs) < 100 < estimate_tokens(msgs)


def test_autocompact_summarizes_original_history_with_tool_calls():
    seen = []
    llm = ScriptedLLM([lambda m: seen.append(m[-1]["content"]) or "摘要"], native_tools=True)
    cm = ContextManager(llm, max_context_tokens=2000, compact_ratio=0.5, keep_recent=2, mc_keep_recent=1)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "任务"}]
    for i in range(6):
        name = "bash" if i % 2 else "write_file"   # write_file 的结果不可再生，microcompact 不清，压不下来只能摘要
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": name,
                                                                       "arguments": {"command": f"cmd{i}"}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": f"OUT{i} " + "x" * 1200}]
    new, reason = cm.maybe_autocompact(msgs)
    assert reason == "autocompact"
    assert "OUT1" in seen[0] and "cmd1" in seen[0]     # bash 结果虽被 microcompact 清过，摘要看到的仍是原文和调用

import json
from agentrt.context import ContextManager
from agentrt.llm import ScriptedLLM
from agentrt.permissions import PermissionEngine, allow_all_approver
from agentrt.session import EventLog
from examples.file_assistant.assistant import assistant_tools, build_assistant


def test_multi_turn_keeps_context(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    llm = ScriptedLLM(['<tool_call>{"name":"find_files","arguments":{"name":"*.txt"}}</tool_call>',
                       "找到 a.txt", "刚才找到的是 a.txt"])
    log = EventLog.new(tmp_path / "s")
    ag = build_assistant(llm, tmp_path, log, permissions=PermissionEngine("default"),
                         approver=allow_all_approver, context=ContextManager(llm), on_event=None)
    assert ag.send("找 txt").final == "找到 a.txt"
    assert ag.send("刚才找到什么").final == "刚才找到的是 a.txt"
    last = llm.calls[-1]
    assert any("a.txt" in (m.get("content") or "") and m["role"] == "tool" for m in last)  # 第二轮带着第一轮的结果
    assert "电脑文件助手" in last[0]["content"]


def _agent(tmp_path, llm, log, approver=allow_all_approver):
    return build_assistant(llm, tmp_path, log, permissions=PermissionEngine("default"), approver=approver,
                           context=ContextManager(llm), on_event=None)


def test_crash_in_second_turn_resumes_without_rerunning_side_effects(tmp_path):
    """第 1 轮正常结束；第 2 轮 trash 已开始但没有结果就崩了。恢复后：不重跑 trash，告诉模型结果未知。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.log").write_text("x")
    log = EventLog.new(tmp_path / "s")
    llm = ScriptedLLM(["你好", '<tool_call>{"name":"trash","arguments":{"path":"a.log"}}</tool_call>', "删了"])
    ag = _agent(ws, llm, log)
    ag.send("hi")
    ag.send("删掉 a.log")
    # 模拟崩溃：丢掉 trash 的结果及之后的所有事件，并把文件放回去（真实世界里它可能已删，也可能没删）
    evs = log.events()
    cut = next(i for i, e in enumerate(evs) if e["type"] == "tool_result" and e["name"] == "trash")
    log.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evs[:cut]))
    (ws / "a.log").write_text("x")

    ag2 = _agent(ws, ScriptedLLM(["我先检查一下状态"]), EventLog(log.path))
    assert ag2.restore() is True                       # 上一轮被中断，不是“已结束”
    res = ag2.resume()
    assert res.status == "end_turn" and (ws / "a.log").exists()     # trash 没有被重跑
    sent = ag2.llm.calls[0]
    assert any("[中断]" in (m.get("content") or "") for m in sent)
    assert sum(1 for m in sent if m["role"] == "user" and m["content"] == "hi") == 1   # 第 1 轮上下文还在


def test_finished_turn_is_not_interrupted(tmp_path):
    log = EventLog.new(tmp_path / "s")
    ag = _agent(tmp_path, ScriptedLLM(["好的"]), log)
    ag.send("hi")
    assert _agent(tmp_path, ScriptedLLM([]), EventLog(log.path)).restore() is False


def test_permission_event_records_wait_and_decision(tmp_path):
    (tmp_path / "a.log").write_text("x")
    log = EventLog.new(tmp_path / "s")
    llm = ScriptedLLM(['<tool_call>{"name":"trash","arguments":{"path":"a.log"}}</tool_call>', "没删"])
    _agent(tmp_path, llm, log, approver=lambda t, a, d: False).send("删")
    p = next(e for e in log.events() if e["type"] == "permission")
    assert p["approved"] is False and p["name"] == "trash" and "wait_seconds" in p
    assert (tmp_path / "a.log").exists()


def test_cancel_gives_every_tool_call_a_result(tmp_path):
    """中断发生在一条消息含两个工具调用之间：第 2 个不执行，但必须有 tool_result，上下文才合法。"""
    (tmp_path / "a.log").write_text("x")
    log = EventLog.new(tmp_path / "s")
    two = ('<tool_call>{"name":"list_dir","arguments":{}}</tool_call>'
           '<tool_call>{"name":"trash","arguments":{"path":"a.log"}}</tool_call>')
    llm = ScriptedLLM([two, "不该被调用"])
    ag = _agent(tmp_path, llm, log)
    orig = ag._execute

    def execute_then_cancel(call):
        r = orig(call)
        ag.cancel()                      # 第一个工具执行完就被用户中断
        return r
    ag._execute = execute_then_cancel
    res = ag.send("干活")
    assert res.status == "interrupted" and (tmp_path / "a.log").exists()
    tool_msgs = [m for m in ag._live if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and "中断" in tool_msgs[1]["content"]
    assert ag._live[-1]["content"].startswith("[用户中断")
    assert len(llm.calls) == 1                                       # 没有再去调模型
    assert ag.send("继续").status == "end_turn"                       # 下一条消息能正常接着聊


def test_repair_after_hard_interrupt(tmp_path):
    ag = _agent(tmp_path, ScriptedLLM([]), EventLog.new(tmp_path / "s"))
    ag._live = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x1", "name": "trash", "arguments": {}}]}]
    ag._chat_started = True
    ag.repair_interrupted()
    assert [m["role"] for m in ag._live[-2:]] == ["tool", "user"] and ag._live[-2]["tool_call_id"] == "x1"


def test_assistant_toolset_is_small():
    names = [t.name for t in assistant_tools()]
    assert len(names) == 9 and "bash" not in names and "write_file" not in names
    assert {t.name for t in assistant_tools() if not t.read_only} == {"move_file", "copy_file", "trash"}


def test_read_only_calls_run_in_parallel_batches(tmp_path):
    from agentrt.llm import ToolCall
    from agentrt.loop import Agent
    llm = ScriptedLLM([""])
    ag = Agent(llm, tmp_path, EventLog.new(tmp_path / "s"), tools=assistant_tools(), context=ContextManager(llm))
    mk = lambda n: ToolCall("i", n, {})  # noqa: E731
    shape = [[c.name for c in b] for b in ag._batches([mk("find_files"), mk("grep"), mk("trash"), mk("list_dir"),
                                                       mk("file_stats")])]
    assert shape == [["find_files", "grep"], ["trash"], ["list_dir", "file_stats"]]


def test_assistant_gets_bash_only_with_sandbox(tmp_path, monkeypatch):
    from agentrt import sandbox
    llm = ScriptedLLM([])
    mk = lambda: build_assistant(llm, tmp_path, EventLog.new(tmp_path / "s"), permissions=PermissionEngine("default"),  # noqa: E731
                                 approver=allow_all_approver, context=ContextManager(llm), on_event=None)
    monkeypatch.setattr(sandbox, "available", lambda: False)
    assert "bash" not in mk().tools                                                   # 没有沙箱：失败即关闭
    monkeypatch.setattr(sandbox, "available", lambda: True)
    assert "bash" in mk().tools

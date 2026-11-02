import json

import pytest

from agentrt.context import ContextManager
from agentrt.llm import ContextLengthError, ScriptedLLM
from agentrt.loop import INTERRUPTED, Agent
from agentrt.permissions import PermissionEngine, allow_all_approver, policy_approver
from agentrt.session import EventLog, ReplayState
from agentrt.tools import Tool, default_tools

from conftest import call


def test_basic_loop_writes_file_and_ends(make_agent):
    agent, llm, ws, log = make_agent([
        call("write_file", path="out/hello.txt", content="hi"),
        call("read_file", path="out/hello.txt"),
        "已写入 out/hello.txt。",
    ])
    res = agent.run("写一个文件")
    assert res.status == "end_turn" and res.final == "已写入 out/hello.txt。"
    assert (ws / "out/hello.txt").read_text() == "hi"
    assert res.turns == 3 and res.tool_calls == 2 and res.tool_errors == 0
    # 第二次调模型时，上下文里带着第一次的工具结果
    assert any(m["role"] == "tool" and "已写入" in m["content"] for m in llm.calls[1])
    types = [e["type"] for e in log.events()]
    assert types[0] == "session_start" and types[-1] == "end"
    assert types.count("tool_start") == 2 and types.count("tool_result") == 2


def test_system_prompt_is_stable_across_turns(make_agent):
    agent, llm, ws, log = make_agent([call("list_dir"), call("list_dir", path="."), "done"])
    agent.run("看看目录")
    systems = {c[0]["content"] for c in llm.calls}
    assert len(systems) == 1  # 前缀不变


def test_path_escape_denied_even_in_bypass(make_agent, tmp_path):
    (tmp_path / "secret.txt").write_text("TOPSECRET")
    agent, llm, ws, log = make_agent([call("read_file", path="../secret.txt"), "不能读"], mode="bypass")
    res = agent.run("读上级目录")
    assert res.denials == 1
    results = [e for e in log.events() if e["type"] == "tool_result"]
    assert "越界" in results[0]["content"] and "TOPSECRET" not in results[0]["content"]


def test_dangerous_command_denied_when_unattended(make_agent):
    agent, llm, ws, log = make_agent([call("bash", command="rm -rf data"), "好的"], approver=policy_approver)
    (ws / "data").mkdir()
    (ws / "data/keep.txt").write_text("1")
    res = agent.run("清理")
    assert res.denials == 1 and (ws / "data/keep.txt").exists()
    assert not any(e["type"] == "tool_start" for e in log.events())


def test_policy_approver_allows_routine_bash(make_agent):
    agent, llm, ws, log = make_agent([call("bash", command="echo hi > a.txt"), "ok"], approver=policy_approver)
    res = agent.run("x")
    assert res.denials == 0 and (ws / "a.txt").read_text().strip() == "hi"


def test_bash_parent_path_needs_approval(make_agent):
    agent, llm, ws, log = make_agent([call("bash", command="cat ../secret.txt"), "ok"], approver=policy_approver)
    res = agent.run("x")
    assert res.denials == 1


@pytest.mark.parametrize("cmd", ["cat ./../secret.txt", "cat src/../../secret.txt", "ls ~/"])
def test_bash_parent_path_variants_need_approval(make_agent, cmd):
    agent, llm, ws, log = make_agent([call("bash", command=cmd), "ok"], approver=policy_approver)
    assert agent.run("x").denials == 1


def test_secret_hook_denies_and_log_is_redacted(make_agent):
    agent, llm, ws, log = make_agent([call("bash", command="sshpass -p hunter2pass ssh a@b ls"), "ok"],
                                     mode="bypass")
    res = agent.run("x")
    assert res.denials == 1
    raw = log.path.read_text()
    assert "hunter2pass" not in raw


def test_plan_mode_blocks_writes(make_agent):
    agent, llm, ws, log = make_agent([call("write_file", path="a.txt", content="x"), call("list_dir"), "ok"],
                                     mode="plan", approver=allow_all_approver)
    res = agent.run("x")
    assert res.denials == 1 and not (ws / "a.txt").exists()


def test_deny_rule_beats_allow_rule(make_agent):
    perms = PermissionEngine("bypass", allow=["bash(git *)"], deny=["bash(git push*)"])
    agent, llm, ws, log = make_agent([call("bash", command="git push origin main"), "ok"], permissions=perms)
    assert agent.run("x").denials == 1


def test_parse_error_is_fed_back(make_agent):
    agent, llm, ws, log = make_agent(['<tool_call>{"name": "list_dir", "arguments": {oops}}</tool_call>',
                                      call("list_dir"), "done"])
    res = agent.run("x")
    assert res.parse_errors == 1 and res.status == "end_turn"
    assert any("格式错误" in (m.get("content") or "") for m in llm.calls[1] if m["role"] == "user")


def test_unknown_tool_and_tool_error_do_not_crash(make_agent):
    agent, llm, ws, log = make_agent([call("teleport", to="mars"), call("read_file", path="nope.txt"), "done"])
    res = agent.run("x")
    assert res.status == "end_turn" and res.tool_errors == 2


def test_repeated_identical_calls_get_nudged(make_agent):
    agent, llm, ws, log = make_agent([call("list_dir")] * 3 + ["done"])
    agent.run("x")
    assert any("连续 3 轮" in (m.get("content") or "") for m in llm.calls[3])


def test_empty_reply_nudged_once(make_agent):
    agent, llm, ws, log = make_agent(["", "完成"])
    res = agent.run("x")
    assert res.final == "完成" and res.turns == 2


def test_max_turns(make_agent):
    agent, llm, ws, log = make_agent([call("list_dir", path=str(i)) for i in range(10)], max_turns=3)
    res = agent.run("x")
    assert res.status == "max_turns" and res.turns == 3


def test_long_output_spilled_to_file(make_agent):
    cm = ContextManager(None, tool_result_max_chars=500)
    agent, llm, ws, log = make_agent([call("bash", command="seq 1 5000"), "ok"], mode="bypass", context=cm)
    cm.llm = llm
    agent.run("x")
    result = next(e for e in log.events() if e["type"] == "tool_result")
    assert "完整输出已保存到 .agentrt/outputs/" in result["content"] and len(result["content"]) < 700
    spilled = list((ws / ".agentrt/outputs").glob("*.txt"))
    assert spilled and "5000" in spilled[0].read_text()


def test_autocompact_keeps_system_and_recent(make_agent):
    llm = ScriptedLLM([])
    steps = []
    for i in range(8):
        steps.append(call("bash", command=f"python3 -c \"print('x' * 3000)\" # {i}"))
    steps.insert(6, "摘要：已执行 6 次打印。")  # 第 7 次调模型前触发压缩，摘要调用消耗这一步
    steps.append("done")
    llm.steps = steps
    cm = ContextManager(llm, max_context_tokens=4000, keep_recent=4)
    agent, _, ws, log = make_agent(None, mode="bypass", llm=llm, context=cm)
    res = agent.run("反复打印")
    assert res.compactions >= 1
    compact = next(e for e in log.events() if e["type"] == "compact")
    msgs = compact["messages"]
    assert msgs[0]["role"] == "system" and "上下文已压缩" in msgs[1]["content"]
    assert msgs[2]["role"] != "tool"  # 不能把工具结果和它的调用切开
    assert compact["after_tokens"] < compact["before_tokens"]


def test_reactive_compact_on_context_length_error(make_agent):
    steps = [call("list_dir", path=".") for _ in range(4)] + [ContextLengthError("too long"), "摘要", "done"]
    agent, llm, ws, log = make_agent(steps)
    res = agent.run("x")
    assert res.status == "end_turn" and res.final == "done"
    assert [e["reason"] for e in log.events() if e["type"] == "compact"] == ["reactive"]


class CountingWrite(Tool):
    name = "append_line"
    description = "向 counter.txt 追加一行（有副作用）"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    read_only = False
    risk = "medium"
    runs = 0

    def subject(self, args):
        return "counter.txt"

    def run(self, args, ctx):
        CountingWrite.runs += 1
        with (ctx.workspace / "counter.txt").open("a") as f:
            f.write(args["text"] + "\n")
        return "ok"


def test_resume_side_effect_at_most_once_readonly_rerun(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello")
    log = EventLog(tmp_path / "s.jsonl")
    # 伪造一次“执行到一半崩溃”的日志：两个调用都 start 了，都没有 result
    asst = {"role": "assistant", "content": "...", "tool_calls": [
        {"id": "t1_0", "name": "read_file", "arguments": {"path": "a.txt"}},
        {"id": "t1_1", "name": "append_line", "arguments": {"text": "x"}}]}
    for e in [{"type": "session_start", "task": "t", "workspace": str(ws)},
              {"type": "message", "message": {"role": "system", "content": "S"}},
              {"type": "message", "message": {"role": "user", "content": "task"}},
              {"type": "message", "message": asst},
              {"type": "tool_start", "call_id": "t1_0", "name": "read_file", "arguments": {"path": "a.txt"}},
              {"type": "tool_start", "call_id": "t1_1", "name": "append_line", "arguments": {"text": "x"}}]:
        log.append(e)
    CountingWrite.runs = 0
    llm = ScriptedLLM(["检查后发现已完成"])
    agent = Agent(llm, ws, log, tools=default_tools() + [CountingWrite()],
                  permissions=PermissionEngine("bypass"), context=ContextManager(llm))
    res = agent.resume()
    assert res.status == "end_turn"
    assert CountingWrite.runs == 0  # 有副作用的调用没有被重放
    tool_msgs = {m["tool_call_id"]: m["content"] for m in llm.calls[0] if m["role"] == "tool"}
    assert "hello" in tool_msgs["t1_0"]          # 只读调用重跑了
    assert tool_msgs["t1_1"] == INTERRUPTED       # 副作用调用告知结果未知
    # 新的 call id 不能和旧的冲突
    st = ReplayState(log.events())
    assert st.turns == 2 and st.ended["status"] == "end_turn"


def test_resume_after_llm_outage(make_agent):
    from agentrt.llm import RetryableError
    agent, llm, ws, log = make_agent([call("write_file", path="a.txt", content="1"),
                                      RetryableError("connection refused")])
    res = agent.run("x")
    assert res.status == "error" and (ws / "a.txt").exists()
    assert not any(e["type"] == "end" for e in log.events())
    llm2 = ScriptedLLM(["继续：已完成"])
    agent2 = Agent(llm2, ws, log, permissions=PermissionEngine("auto"), context=ContextManager(llm2))
    res2 = agent2.resume()
    assert res2.status == "end_turn" and res2.final == "继续：已完成"
    # 恢复后的上下文里有之前写文件的结果，且只写了一次
    assert any(m["role"] == "tool" and "已写入" in m["content"] for m in llm2.calls[0])
    assert sum(1 for e in log.events() if e["type"] == "tool_start") == 1


def test_resume_ended_session_is_noop(make_agent):
    agent, llm, ws, log = make_agent(["done"])
    agent.run("x")
    agent2 = Agent(ScriptedLLM([]), ws, log)
    assert agent2.resume().status == "already_ended"


def test_event_log_tolerates_truncated_tail(tmp_path):
    log = EventLog(tmp_path / "s.jsonl")
    log.append({"type": "message", "message": {"role": "user", "content": "a"}})
    with log.path.open("a") as f:
        f.write('{"type": "message", "mess')
    assert len(log.events()) == 1


def test_redaction():
    from agentrt.session import redact
    s = redact({"a": ["key sk-abcdefghijklmnopqrstuv", "password=hunter22"]})
    assert "abcdefghijkl" not in json.dumps(s) and "hunter22" not in json.dumps(s)


def test_cached_tokens_accounted(make_agent):
    class CachingLLM(ScriptedLLM):
        def chat(self, messages, tools_schema=None, call_prefix="c"):
            r = super().chat(messages, tools_schema, call_prefix)
            r.usage = {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 80}}
            return r
    agent, llm, ws, log = make_agent(None, llm=CachingLLM([call("list_dir"), "done"]))
    res = agent.run("x")
    assert res.prompt_tokens == 200 and res.cached_tokens == 160


def test_volatile_prefix_changes_system(make_agent):
    agent, llm, ws, log = make_agent([call("list_dir"), "done"], volatile_prefix=True)
    agent.run("x")
    assert llm.calls[0][0]["content"] != llm.calls[1][0]["content"]
    assert agent.system in llm.calls[0][0]["content"]


# ---------------------------------------------------------------- 权限规则、中断、并行等行为

def test_ask_rule_beats_allow_and_bypass(make_agent):
    from agentrt.permissions import PermissionEngine
    from agentrt.tools import ListDir
    eng = PermissionEngine("bypass", allow=["list_dir"], ask=["list_dir"])
    d = eng.decide(ListDir(), {}, __import__("agentrt.tools", fromlist=["ToolContext"]).ToolContext(__import__("pathlib").Path(".")))
    assert d.behavior == "ask" and d.source == "ask_rule"


def test_hook_ask_is_not_skipped_in_bypass(tmp_path):
    from agentrt.permissions import PermissionEngine
    from agentrt.tools import Bash, ToolContext
    d = PermissionEngine("bypass").decide(Bash(), {"command": "sudo ls"}, ToolContext(tmp_path))
    assert d.behavior == "ask" and d.source.startswith("hook:")


def test_deny_rule_beats_hook_allow(tmp_path):
    from agentrt.permissions import Decision, PermissionEngine
    from agentrt.tools import ListDir, ToolContext
    eng = PermissionEngine("default", deny=["list_dir"], hooks=[lambda t, a, c: Decision("allow", "hook", "hook:x")])
    assert eng.decide(ListDir(), {}, ToolContext(tmp_path)).behavior == "deny"


def test_microcompact_clears_old_results_keeps_recent():
    from agentrt.context import CLEARED, clear_old_results
    msgs = [{"role": "system", "content": "s"}]
    for i in range(8):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "name": "grep", "content": f"result {i}"}]
    msgs.append({"role": "tool", "tool_call_id": "w", "name": "trash", "content": "不可再生，必须保留"})
    out, n = clear_old_results(msgs, keep_recent=5)
    tools = [m for m in out if m["role"] == "tool"]
    assert n == 3 and [m["content"] for m in tools[:3]] == [CLEARED] * 3
    assert tools[3]["content"] == "result 3" and tools[-1]["content"].startswith("不可再生")
    assert clear_old_results(msgs, keep_recent=0)[1] == 7              # 至少保留 1 条


def test_autocompact_tries_microcompact_before_summarizing(make_agent):
    from agentrt.context import ContextManager
    from agentrt.llm import ScriptedLLM
    llm = ScriptedLLM([])
    cm = ContextManager(llm, max_context_tokens=4000, mc_keep_recent=1)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": "x" * 3000}]
    new, reason = cm.maybe_autocompact(msgs)
    assert reason == "microcompact" and llm.calls == []                # 没有调用模型做摘要


def test_read_only_calls_run_in_parallel_batches(tmp_path):
    from agentrt.context import ContextManager
    from agentrt.llm import ScriptedLLM, ToolCall
    from agentrt.loop import Agent
    from agentrt.session import EventLog
    llm = ScriptedLLM([""])
    ag = Agent(llm, tmp_path, EventLog.new(tmp_path / "s"), context=ContextManager(llm))
    mk = lambda n: ToolCall("i", n, {})  # noqa: E731
    shape = [[c.name for c in b] for b in ag._batches([mk("read_file"), mk("grep"), mk("write_file"), mk("list_dir"),
                                                       mk("read_file")])]
    assert shape == [["read_file", "grep"], ["write_file"], ["list_dir", "read_file"]]


# ---------------------------------------------------------------- 崩溃恢复：残留的子进程
# 场景：bash 用 start_new_session=True 起子进程（超时时才能整组杀掉），主进程被 kill -9 后子进程还在跑。
# 以前 resume 只告诉模型“结果未知，先检查”，可旧命令还在运行，模型检查到的是过时的状态，重做就会执行两次。

def _crash_log(tmp_path, ws, pgid, started_at):
    """伪造“bash 起了进程、结果没写就崩溃”的日志。"""
    log = EventLog(tmp_path / "s.jsonl")
    args = {"command": "sleep 30; echo paid >> ledger.txt"}
    asst = {"role": "assistant", "content": "...", "tool_calls": [{"id": "t1_0", "name": "bash", "arguments": args}]}
    for e in [{"type": "session_start", "task": "t", "workspace": str(ws)},
              {"type": "message", "message": {"role": "system", "content": "S"}},
              {"type": "message", "message": {"role": "user", "content": "task"}},
              {"type": "message", "message": asst},
              {"type": "tool_start", "call_id": "t1_0", "name": "bash", "arguments": args},
              {"type": "tool_proc", "call_id": "t1_0", "pgid": pgid, "started_at": started_at}]:
        log.append(e)
    return log


def _resume(tmp_path, ws, log):
    llm = ScriptedLLM(["已检查"])
    agent = Agent(llm, ws, log, permissions=PermissionEngine("bypass"), context=ContextManager(llm))
    assert agent.resume().status == "end_turn"
    return [m for m in llm.calls[0] if m["role"] == "tool"][0]["content"]


def test_bash_records_its_process_group_before_waiting(make_agent):
    agent, llm, ws, log = make_agent([call("bash", command="echo hi"), "完成"], mode="bypass")
    agent.run("x")
    ev = log.events()
    i_start = next(i for i, e in enumerate(ev) if e["type"] == "tool_start")
    i_proc = next(i for i, e in enumerate(ev) if e["type"] == "tool_proc")
    i_res = next(i for i, e in enumerate(ev) if e["type"] == "tool_result")
    assert i_start < i_proc < i_res and ev[i_proc]["pgid"] > 0 and ev[i_proc]["call_id"] == ev[i_start]["call_id"]


def test_resume_kills_orphan_process_group(tmp_path):
    import os
    import subprocess
    import time
    ws = tmp_path / "ws"
    ws.mkdir()
    orphan = subprocess.Popen("sleep 30; echo paid >> ledger.txt", shell=True, cwd=ws, start_new_session=True)
    log = _crash_log(tmp_path, ws, orphan.pid, time.time())
    msg = _resume(tmp_path, ws, log)
    orphan.wait(timeout=2)                                   # 进程组已被终止
    with pytest.raises(ProcessLookupError):
        os.killpg(orphan.pid, 0)
    assert msg.startswith(INTERRUPTED) and "已强制终止" in msg
    assert any(e["type"] == "reap" and e["action"] == "killed" for e in log.events())
    assert not (ws / "ledger.txt").exists()                  # 副作用不会在恢复之后才发生


def test_resume_orphan_already_exited(tmp_path):
    import subprocess
    import time
    ws = tmp_path / "ws"
    ws.mkdir()
    p = subprocess.Popen("true", shell=True, start_new_session=True)
    p.wait()
    log = _crash_log(tmp_path, ws, p.pid, time.time())
    msg = _resume(tmp_path, ws, log)
    assert "已经退出" in msg and not any(e["type"] == "reap" for e in log.events())


def test_resume_does_not_kill_a_reused_process_group(tmp_path):
    import os
    import signal
    import subprocess
    import time
    ws = tmp_path / "ws"
    ws.mkdir()
    other = subprocess.Popen("sleep 30", shell=True, start_new_session=True)   # 同一个号，现在是别的进程
    try:
        log = _crash_log(tmp_path, ws, other.pid, time.time() - 3600)          # 日志里记的是一小时前起的进程
        msg = _resume(tmp_path, ws, log)
        os.killpg(other.pid, 0)                                                 # 还活着：没有误杀
        assert "没有终止" in msg
        assert any(e["type"] == "reap" and e["action"] == "skip_reused" for e in log.events())
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()


def test_truncated_empty_reply_is_not_end_turn(make_agent):
    cut = {"content": "", "reasoning": "想了很久", "finish_reason": "length"}
    agent, llm, ws, log = make_agent([cut, cut, "完成"])
    res = agent.run("x")
    assert res.status == "end_turn" and res.final == "完成" and res.turns == 3
    assert any("长度上限" in (m.get("content") or "") for m in llm.calls[1])
    # 端点关不了思考：只能提示；半截思考不进上下文
    assert llm.thinks == [True, True, True]
    assert not any(m.get("reasoning") for m in llm.calls[2])


def test_thinking_only_truncation_retries_with_thinking_off(make_agent):
    cut = {"content": "", "reasoning": "想了很久", "finish_reason": "length"}
    llm = ScriptedLLM([cut, "完成"], think_off={"reasoning": {"enabled": False}})
    agent, _, ws, log = make_agent(None, llm=llm)
    res = agent.run("x")
    assert res.status == "end_turn" and res.final == "完成" and res.turns == 2
    assert llm.thinks == [True, False]
    assert llm.calls[1] == llm.calls[0]   # 同样的上下文：没有半截回复，也没有提示
    assert res.cache_breaks == 0          # 第二次请求仍以第一次为前缀
    events = log.events()
    dropped = [e for e in events if e["type"] == "dropped_reply"]
    assert len(dropped) == 1 and dropped[0]["reasoning"] == "想了很久"
    assert [e.get("think", True) for e in events if e["type"] == "span"] == [True, False]
    st = ReplayState(events)
    assert st.turns == 2 and not any(m.get("reasoning") for m in st.messages)


def test_truncation_count_resets_after_tool_calls(make_agent):
    cut = {"content": "", "finish_reason": "length"}
    steps = []
    for i in range(4):
        steps += [cut, call("list_dir", path=f"d{i}")]
    agent, llm, ws, log = make_agent(steps + ["完成"], mode="bypass")
    res = agent.run("x")
    assert res.status == "end_turn"   # 4 次截断，但不连续


def test_wrapup_notice_once_at_80_percent(make_agent, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("agentrt.loop.time.time", lambda: now[0])

    def at(t, reply):
        def step(messages):
            now[0] = t
            return reply
        return step

    agent, llm, ws, log = make_agent([at(1050, call("list_dir", path="a")), at(1085, call("list_dir", path="b")),
                                      call("list_dir", path="c"), "完成"], mode="bypass", deadline=1100.0)
    res = agent.run("x")
    assert res.status == "end_turn"
    notes = [[m for m in c if "时间提醒" in (m.get("content") or "")] for c in llm.calls]
    assert [len(n) for n in notes] == [0, 0, 1, 1]
    assert "还剩约 15 秒" in notes[2][0]["content"] and notes[2][0]["kind"] == "harness"


def test_truncated_forever_gives_up_as_truncated(make_agent):
    cut = {"content": "", "finish_reason": "length"}
    agent, llm, ws, log = make_agent([cut] * 6)
    res = agent.run("x")
    assert res.status == "truncated" and res.turns == 4

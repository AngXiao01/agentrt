"""coding 工具集 + sandbox + 主循环：read-before-write、cwd 保持、超长输出落在 sandbox、reasoning 回传、前缀改写检测、
sandbox 感知的权限、摘要调用记账。全部用脚本化模型，不需要真实模型。"""
import json

import pytest

from agentrt.coding_tools import SbGlob, SbGrep, build_coding_agent, coding_tools, glob_to_regex
from agentrt.context import ContextManager
from agentrt.llm import ScriptedLLM, normalize_usage, reasoning_keep_from, split_think, to_wire
from agentrt.loop import Agent
from agentrt.permissions import PermissionEngine, policy_approver
from agentrt.sandboxes import BwrapSandbox, LocalSandbox, bwrap_available
from agentrt.session import EventLog
from agentrt.tools import Bash, ToolContext

from conftest import call


def _agent(tmp_path, steps, sandbox=None, **kw):
    llm = kw.pop("llm", None) or ScriptedLLM(steps)
    sb = sandbox or LocalSandbox(tmp_path / "ws")
    log = EventLog(tmp_path / "sessions" / "s.jsonl")
    agent = build_coding_agent(llm, sb, log, context=kw.pop("context", None) or ContextManager(llm),
                               permissions=kw.pop("permissions", None) or PermissionEngine("auto"), **kw)
    return agent, llm, sb, log


def _results(log):
    return [e for e in log.events() if e["type"] == "tool_result"]


# ---------------------------------------------------------------- 工具语义

def test_write_existing_file_requires_read_first(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws/a.py").write_text("x = 1\n")
    agent, _, _, log = _agent(tmp_path, [
        call("write_file", path="a.py", content="x = 2\n"),
        call("read_file", path="a.py"),
        call("edit_file", path="a.py", old="x = 1", new="x = 3"),
        "done"])
    agent.run("改 a.py")
    r = _results(log)
    assert r[0]["is_error"] and "还没读过" in r[0]["content"]
    assert not r[2]["is_error"] and (tmp_path / "ws/a.py").read_text() == "x = 3\n"


def test_stale_file_after_bash_modification(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws/a.txt").write_text("one\n")
    agent, _, _, log = _agent(tmp_path, [
        call("read_file", path="a.txt"),
        call("bash", command="echo two >> a.txt"),
        call("edit_file", path="a.txt", old="one", new="ONE"),
        "done"])
    agent.run("x")
    r = _results(log)
    assert r[2]["is_error"] and "被修改过" in r[2]["content"]
    assert (tmp_path / "ws/a.txt").read_text() == "one\ntwo\n"


def test_edit_requires_unique_match_or_replace_all(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws/a.txt").write_text("a a a\n")
    agent, _, _, log = _agent(tmp_path, [
        call("read_file", path="a.txt"),
        call("edit_file", path="a.txt", old="a", new="b"),
        call("edit_file", path="a.txt", old="a", new="b", replace_all=True),
        "done"])
    agent.run("x")
    r = _results(log)
    assert r[1]["is_error"] and "3 次" in r[1]["content"]
    assert (tmp_path / "ws/a.txt").read_text() == "b b b\n"


def test_bash_keeps_cwd_between_calls(tmp_path):
    (tmp_path / "ws/sub").mkdir(parents=True)
    agent, _, _, log = _agent(tmp_path, [call("bash", command="cd sub && pwd"), call("bash", command="pwd"), "ok"])
    agent.run("x")
    r = _results(log)
    assert r[1]["content"].strip().endswith("/ws/sub")
    assert "__AGENTRT_CWD__" not in r[0]["content"]


def test_long_output_spills_into_sandbox_tmp_not_workdir(tmp_path):
    agent, _, sb, log = _agent(tmp_path, [call("bash", command="seq 1 20000"), "ok"],
                               context=ContextManager(ScriptedLLM([]), tool_result_max_chars=500))
    agent.run("x")
    content = _results(log)[0]["content"]
    path = content.split("完整输出已保存到 ")[1].split("，")[0]
    assert path.startswith(sb.scratch()) and not path.startswith(sb.workdir)
    assert b"19999" in sb.read_bytes(path)   # 模型能用 read_file 读回来
    assert not any((tmp_path / "ws").rglob("*"))   # 任务目录里没有 harness 的垃圾
    sb.close()


def test_glob_and_grep(tmp_path):
    ws = tmp_path / "ws"
    for rel in ["a.py", "src/b.py", "src/deep/c.py", "src/d.ts", "node_modules/x.py", ".git/y.py"]:
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text("TODO here\n" if rel.endswith("c.py") else "nothing\n")
    sb = LocalSandbox(ws)
    ctx = ToolContext(ws, sandbox=sb)
    out = SbGlob().run({"pattern": "**/*.py"}, ctx).splitlines()
    assert out == ["a.py", "src/b.py", "src/deep/c.py"]
    assert SbGlob().run({"pattern": "src/*.{py,ts}"}, ctx).splitlines() == ["src/b.py", "src/d.ts"]
    hit = SbGrep().run({"pattern": "TODO", "glob": "*.py"}, ctx)
    assert "src/deep/c.py:1:TODO here" in hit and "node_modules" not in hit
    sb._has = {"rg": False}   # 没有 ripgrep 时退回 grep -E
    assert "src/deep/c.py:1:TODO here" in SbGrep().run({"pattern": "TO+DO"}, ctx)


def test_glob_regex():
    rx = glob_to_regex("**/test_*.py")
    assert rx.match("test_a.py") and rx.match("x/y/test_a.py") and not rx.match("x/atest_a.py")
    assert not glob_to_regex("*.py").match("a/b.py")


@pytest.mark.skipif(not bwrap_available(), reason="bwrap 不可用")
def test_read_only_bash_runs_with_read_only_mount(tmp_path):
    """判成只读的命令（cat / ls / ... 白名单）用只读挂载执行：解析判错了内核也会拦。"""
    ws = tmp_path / "ws"
    agent, _, _, log = _agent(tmp_path, [call("bash", command="ls"), call("bash", command="echo hi > a.txt"), "ok"],
                              sandbox=BwrapSandbox(ws))
    agent.run("x")
    r = _results(log)
    assert r[1]["content"].startswith("[exit 0]") and (ws / "a.txt").read_text() == "hi\n"


# ---------------------------------------------------------------- 权限

class _FakeSb:
    def __init__(self, enforcement, disposable):
        self.enforcement, self.disposable = enforcement, disposable


def _decide(tool, args, sb, mode="auto", **kw):
    eng = PermissionEngine(mode, **kw)
    return eng.decide(tool, args, ToolContext(__import__("pathlib").Path("/tmp"), sandbox=sb))


def test_permissions_follow_sandbox_strength():
    bash = coding_tools()[0]
    # 宿主机裸跑：和原来一样，引用工作目录外的路径要确认
    assert _decide(bash, {"command": "cat /etc/hosts > x"}, _FakeSb("none", False)).behavior == "ask"
    # bwrap（内核隔离、非一次性）：路径启发式不再误报，bash 免确认；但 rm -rf 仍然要确认
    assert _decide(bash, {"command": "cat /etc/hosts > x"}, _FakeSb("full", False)).behavior == "allow"
    assert _decide(bash, {"command": "rm -rf build"}, _FakeSb("full", False)).behavior == "ask"
    # 一次性容器：爆炸半径就是容器，rm -rf 也放行；deny 规则照样生效；明文凭据照样拒绝
    assert _decide(bash, {"command": "rm -rf build"}, _FakeSb("full", True)).behavior == "allow"
    assert _decide(bash, {"command": "git push --force"}, _FakeSb("full", True),
                   deny=["bash(git push*)"]).behavior == "deny"
    assert _decide(bash, {"command": "sshpass -p hunter2x ssh h"}, _FakeSb("full", True)).behavior == "deny"
    # 旧的本地 bash（不在 sandbox 里）不受影响
    assert _decide(Bash(), {"command": "cat /etc/hosts > x"}, _FakeSb("full", True)).behavior == "ask"
    # plan 模式仍然只读
    assert _decide(bash, {"command": "touch x"}, _FakeSb("full", True), mode="plan").behavior == "deny"


# ---------------------------------------------------------------- reasoning 回传

def _conv():
    return [{"role": "system", "content": "s"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "reasoning": "r1", "tool_calls": []},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a", "reasoning": "r2", "tool_calls": []},
            {"role": "user", "content": "parse error", "kind": "harness"},
            {"role": "assistant", "content": "b", "reasoning": "r3", "tool_calls": []}]


@pytest.mark.parametrize("policy,expected", [("none", []), ("turn", ["r2", "r3"]), ("all", ["r1", "r2", "r3"])])
def test_reasoning_replay_policies(policy, expected):
    wire = to_wire(_conv(), native_tools=True, merge_system=False, reasoning_replay=policy)
    assert [m["reasoning_content"] for m in wire if "reasoning_content" in m] == expected
    assert reasoning_keep_from(_conv(), "turn") == 3   # harness 插的提示不算新一轮


def test_split_think_and_usage_normalization():
    assert split_think("<think>plan</think>\n<tool_call>{}</tool_call>") == ("<tool_call>{}</tool_call>", "plan")
    assert split_think("no think") == ("no think", "")
    ds = normalize_usage({"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 64,
                          "prompt_cache_miss_tokens": 36, "completion_tokens_details": {"reasoning_tokens": 12}})
    assert ds == {"prompt": 100, "completion": 20, "cached": 64, "reasoning": 12}
    oa = normalize_usage({"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 8}})
    assert oa["cached"] == 8


def test_reasoning_is_kept_within_turn_and_sent_back(tmp_path):
    steps = [{"content": call("bash", command="echo hi"), "reasoning": "先看看环境"}, "done"]
    agent, llm, _, log = _agent(tmp_path, steps)
    agent.run("x")
    second = llm.wires[1]
    asst = [m for m in second if m["role"] == "assistant"]
    assert asst[0]["reasoning_content"] == "先看看环境"
    logged = [e["message"] for e in log.events() if e["type"] == "message" and e["message"]["role"] == "assistant"]
    assert logged[0]["reasoning"] == "先看看环境"   # 模型看到的都在日志里，可以重建请求


def test_prefix_break_detection(tmp_path):
    """turn 策略：新一轮用户消息到来时，上一轮的 reasoning 被丢掉 → 前缀被改写（KV cache 失效）。all 策略不会。"""
    def run(policy):
        llm = ScriptedLLM([{"content": "a1", "reasoning": "r1"}, {"content": "a2", "reasoning": "r2"}],
                          reasoning_replay=policy)
        log = EventLog(tmp_path / f"{policy}.jsonl")
        agent = Agent(llm, tmp_path / "ws", log, context=ContextManager(llm))
        agent.send("q1")
        res = agent.send("q2")
        return res.cache_breaks, [e.get("prefix_break") for e in log.events() if e.get("kind") == "llm"]
    assert run("turn") == (1, [None, 2])
    assert run("all") == (0, [None, None])


def test_volatile_prefix_breaks_every_request(tmp_path):
    agent, llm, _, log = _agent(tmp_path, [call("bash", command="true"), call("bash", command="true"), "ok"],
                                volatile_prefix=True)
    res = agent.run("x")
    assert res.cache_breaks == 2
    spans = [e for e in log.events() if e.get("kind") == "llm"]
    assert all(s.get("volatile_stamp") for s in spans)   # 时间戳记进日志：请求能从日志重建


def test_summary_call_is_accounted(tmp_path):
    """压缩时的摘要调用也是模型调用：tokens 要记进结果，否则开压缩的配置看起来更省。"""
    big = "x" * 4000
    steps = [call("bash", command=f"echo {big}"), call("bash", command=f"echo {big}"),
             call("bash", command=f"echo {big}"), call("bash", command="true"), "摘要", "done"]
    llm = ScriptedLLM(steps)
    cm = ContextManager(llm, max_context_tokens=2500, compact_ratio=0.5, keep_recent=2, mc_keep_recent=1)
    agent, _, _, log = _agent(tmp_path, [], llm=llm, context=cm)
    res = agent.run("x")
    compacts = [e for e in log.events() if e["type"] == "compact"]
    assert compacts and res.compactions == len(compacts)
    if any(c["reason"] == "autocompact" for c in compacts):
        assert res.compact_tokens > 0


def test_cost_accounting(tmp_path):
    agent, llm, _, _ = _agent(tmp_path, ["done"], price={"input": 1.0, "cached": 0.1, "output": 2.0})
    res = agent.run("x" * 400)
    assert res.cost_usd == pytest.approx((res.prompt_tokens * 1.0 + res.completion_tokens * 2.0) / 1e6)


def test_session_start_records_sandbox_and_params(tmp_path):
    agent, _, sb, log = _agent(tmp_path, ["done"])
    agent.run("x")
    start = log.events()[0]
    assert start["sandbox"]["kind"] == "local" and "bash" in start["tools"]
    assert sb.workdir in agent.system   # 环境说明写进 system prompt，会话内不变


@pytest.mark.parametrize("native", [False, True])
def test_plain_tool_loop_never_breaks_prefix(tmp_path, native):
    """只追加的正常循环（含格式错误提示、harness 插话）不应被误报成前缀改写。"""
    steps = [call("bash", command="echo 1"), "<tool_call>{bad json</tool_call>",
             {"content": call("read_file", path="nope.txt"), "reasoning": "r"}, call("bash", command="echo 2"), "done"]
    agent, _, _, log = _agent(tmp_path, [], llm=ScriptedLLM(steps, native_tools=native))
    res = agent.run("x")
    assert res.status == "end_turn" and res.cache_breaks == 0, [e.get("prefix_break") for e in log.events()
                                                                if e.get("kind") == "llm"]

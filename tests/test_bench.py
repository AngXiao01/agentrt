"""benchmark 适配：Harbor 桥接（假的 async environment，不需要装 harbor）、评测 runner（并发 / 续跑 / 配对比较）、统计。"""
import asyncio
import json
import shutil
import threading
import types

import pytest

from agentrt.bench.harbor_agent import HarborSandbox, _resolve_model
from agentrt.coding_tools import build_coding_agent
from agentrt.llm import ScriptedLLM
from agentrt.session import EventLog
from evals import run_evals
from evals.stats import bootstrap_diff, mcnemar, pass_at_k, pass_hat_k, wilson

from conftest import call


class FakeHarborEnv:
    """Harbor BaseEnvironment 的最小替身：exec(command, cwd, env, timeout_sec, user) -> ExecResult；upload_file。"""

    def __init__(self):
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        p = await asyncio.create_subprocess_exec("bash", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(p.communicate(), timeout_sec)
        return types.SimpleNamespace(stdout=out.decode(), stderr=err.decode(), return_code=p.returncode)

    async def upload_file(self, source_path, target_path):
        shutil.copyfile(source_path, target_path)


@pytest.fixture
def harbor_loop():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(5)


def test_harbor_sandbox_roundtrip(tmp_path, harbor_loop):
    sb = HarborSandbox(FakeHarborEnv(), harbor_loop, str(tmp_path))
    sb.write_bytes("d/bin.dat", bytes(range(256)))
    assert sb.read_bytes("d/bin.dat") == bytes(range(256))   # base64 通道：二进制原样往返
    r = sb.exec("echo out; echo err >&2; exit 4", timeout=10)
    assert r.exit_code == 4 and "out" in r.output and "err" in r.output
    r = sb.exec("sleep 30", timeout=1)
    assert r.timed_out and r.seconds < 10
    with pytest.raises(FileNotFoundError):
        sb.read_bytes("missing.txt")
    with pytest.raises(IsADirectoryError):
        sb.read_bytes("d")


def test_agent_runs_through_harbor_bridge(tmp_path, harbor_loop):
    """主循环跑在工作线程，工具调用经 run_coroutine_threadsafe 投递到 Harbor 的事件循环——和 AgentRT.run 一样的结构。"""
    ws = tmp_path / "app"
    ws.mkdir()
    (ws / "main.py").write_text("print('helo')\n")
    env = FakeHarborEnv()
    sb = HarborSandbox(env, harbor_loop, str(ws))
    llm = ScriptedLLM([call("read_file", path="main.py"),
                       call("edit_file", path="main.py", old="helo", new="hello"),
                       call("bash", command="python3 main.py"), "fixed"], native_tools=True)
    agent = build_coding_agent(llm, sb, EventLog(tmp_path / "s.jsonl"))

    async def drive():
        return await asyncio.to_thread(agent.run, "fix the typo")

    res = asyncio.run_coroutine_threadsafe(drive(), harbor_loop).result(60)
    assert res.status == "end_turn" and (ws / "main.py").read_text() == "print('hello')\n"
    assert "[exit 0]\nhello" in [e for e in EventLog(tmp_path / "s.jsonl").events()
                                 if e["type"] == "tool_result"][-1]["content"]
    assert "a Linux container" in agent.system and str(ws) in agent.system


def test_resolve_model():
    assert _resolve_model("deepseek/deepseek-chat", None, None) == ("https://api.deepseek.com/v1", "deepseek-chat")
    assert _resolve_model("openrouter/qwen/qwen3-coder", None, None)[1] == "qwen/qwen3-coder"
    assert _resolve_model("openai/x", "http://h/v1", "y") == ("http://h/v1", "y")
    with pytest.raises(ValueError):
        _resolve_model("unknownvendor/m", None, None)


# ---------------------------------------------------------------- 统计

def test_stats():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert pass_at_k(5, 0, 3) == 0 and pass_at_k(5, 3, 3) == 1.0
    assert pass_hat_k(3, 3, 3) == 1.0 and pass_hat_k(5, 3, 3) == pytest.approx(0.1)
    m = mcnemar([(True, False)] * 6 + [(True, True)] * 10)
    assert (m["b"], m["c"]) == (6, 0) and m["p"] == pytest.approx(2 / 64)
    assert mcnemar([(True, True)])["p"] == 1.0
    d = bootstrap_diff({"a": [True, True], "b": [False, False]}, {"a": [True, True], "b": [True, True]})
    assert d[0] == pytest.approx(0.5)


# ---------------------------------------------------------------- 评测 runner

def _args(tmp_path, **kw):
    ns = types.SimpleNamespace(tasks="json_total,count_lines", variants="baseline,reasoning_none", k=2,
                               out=tmp_path / "out", resume=None, parallel=3, max_turns=8, max_context=24000,
                               sandbox="local", keep=False, price=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _oracle_llm(variant):
    """“标准答案”模型：用 coding 工具集算出答案写文件。reasoning_none 配置故意在 count_lines 上失败。"""
    def step(messages):
        task = messages[1]["content"]
        if "orders.json" in task:
            cmd = ("python3 -c \"import json;o=json.load(open('orders.json'))['orders'];"
                   "print('%.2f'%sum(x['amount'] for x in o if x['status']=='paid'))\" > total.txt")
        elif variant.get("reasoning_replay") == "none":
            cmd = "echo 0 > answer.txt"
        else:
            cmd = "find src -name '*.py' -exec cat {} + | wc -l > answer.txt"
        return call("bash", command=cmd)
    return ScriptedLLM([step, "done"])


def test_eval_runner_parallel_resume_and_compare(tmp_path):
    a = _args(tmp_path, llm_factory=_oracle_llm)
    out = run_evals.run(a)
    s = out["summary"]
    assert s["baseline"]["success_rate"] == 1.0 and s["reasoning_none"]["success_rate"] == 0.5
    cmp_ = out["comparisons"]["reasoning_none"]
    assert cmp_["pairs"] == 4 and (cmp_["b"], cmp_["c"]) == (2, 0)
    report = (tmp_path / "out" / "report.md").read_text()
    assert "配对比较" in report and "ci95" in report
    n = len((tmp_path / "out" / "trials.jsonl").read_text().splitlines())
    assert n == 8
    # 续跑：k 从 2 加到 3，只补新的 4 次
    out2 = run_evals.run(_args(tmp_path, llm_factory=_oracle_llm, k=3, resume=tmp_path / "out"))
    assert len((tmp_path / "out" / "trials.jsonl").read_text().splitlines()) == 12
    assert out2["summary"]["baseline"]["trials"] == 6
    recs = [json.loads(l) for l in (tmp_path / "out" / "trials.jsonl").read_text().splitlines()]
    assert all(r["sandbox"] == "local" for r in recs)

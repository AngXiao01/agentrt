"""benchmark 适配：Harbor 桥接（假的 async environment，不需要装 harbor）、评测 runner（并发 / 续跑 / 配对比较）、统计。"""
import asyncio
import json
import shutil
import subprocess
import threading
import types

import pytest

from agentrt.bench.harbor_agent import HarborSandbox, _agent_timeout, _resolve_model, _think_off
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


def test_kill_running_stops_in_flight_command(tmp_path, harbor_loop):
    """Harbor 判 agent 超时后：还在跑的命令连同子进程整组杀掉，后面的写入不会发生；跑完的命令不受影响。"""
    import time
    from pathlib import Path
    sb = HarborSandbox(FakeHarborEnv(), harbor_loop, str(tmp_path))
    assert sb.exec("exit 3", timeout=10).exit_code == 3
    out = {}
    t = threading.Thread(target=lambda: out.update(r=sb.exec("sleep 30 & wait; echo late > late.txt", timeout=60)))
    t0 = time.time()
    t.start()
    running = Path(sb.running)
    while not (running.is_dir() and any(running.iterdir())) and time.time() - t0 < 10:
        time.sleep(0.05)
    pid = next(running.iterdir()).name
    asyncio.run_coroutine_threadsafe(sb.kill_running(), harbor_loop).result(30)
    t.join(20)
    assert not t.is_alive() and out["r"].exit_code != 0 and time.time() - t0 < 20
    assert not (tmp_path / "late.txt").exists() and not any(running.iterdir())
    assert subprocess.run(["pgrep", "-g", pid], capture_output=True).returncode != 0   # 整组都没了（含 sleep）


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


def _harbor_trial(tmp_path, task_toml, **cfg):
    task = tmp_path / "tasks" / "t"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(task_toml)
    trial = tmp_path / "jobs" / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(task)}, **cfg}))
    return trial / "agent"


TASK_TOML = """[task]
name = "x"
[verifier]
timeout_sec = 1200.0

[agent]
timeout_sec = 900.0   # 注释
[environment]
cpus = 1
"""


def test_agent_timeout_follows_harbor(tmp_path):
    assert _agent_timeout(_harbor_trial(tmp_path / "a", TASK_TOML)) == 900          # 不是 [verifier] 的 1200
    assert _agent_timeout(_harbor_trial(tmp_path / "b", TASK_TOML, timeout_multiplier=2)) == 1800
    assert _agent_timeout(_harbor_trial(tmp_path / "c", TASK_TOML, timeout_multiplier=2, agent_timeout_multiplier=0.5)) == 450
    assert _agent_timeout(_harbor_trial(tmp_path / "d", TASK_TOML,
                                 agent={"override_timeout_sec": 300, "max_timeout_sec": 200})) == 200
    assert _agent_timeout(_harbor_trial(tmp_path / "e", "[agent]\nname = 'no timeout'\n")) is None
    assert _agent_timeout(tmp_path / "nowhere" / "agent") is None


def test_think_off_defaults():
    assert _think_off({}, "https://openrouter.ai/api/v1") == {"reasoning": {"enabled": False}}
    assert _think_off({}, "https://api.deepseek.com/v1") is None
    assert _think_off({"think_off": None}, "https://openrouter.ai/api/v1") is None
    assert _think_off({"think_off": '{"thinking": {"type": "disabled"}}'}, "x") == {"thinking": {"type": "disabled"}}


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


# ---------------------------------------------------------------- Harbor 结果汇总 / TB2 子集

def _trial(tmp, agent, task, start, reward=None, exc=None, kwargs=None, cost=None):
    from pathlib import Path
    r = {"task_name": task, "trial_name": f"{task}__{agent}__{start}", "started_at": f"2026-11-01T00:00:{start:02d}Z",
         "agent_info": {"name": agent}, "config": {"agent": {"name": agent, "model_name": "m/x", "kwargs": kwargs or {}}},
         "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 50, "n_output_tokens": 10, "cost_usd": cost},
         "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
         "exception_info": {"exception_type": exc} if exc else None}
    d = Path(tmp) / r["trial_name"]
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(r))


def test_harbor_report_separates_infra_and_pairs(tmp_path):
    from agentrt.bench.report import load_trials, summarize, to_markdown
    for i, t in enumerate(["a", "b", "c"]):
        _trial(tmp_path, "oracle", t, i, reward=0.0 if t == "c" else 1.0)   # c 在这台机器上判不了
    for att in range(2):
        _trial(tmp_path, "agentrt", "a", 10 + att, reward=1.0, cost=0.01, kwargs={"api_base": "x"})
        _trial(tmp_path, "agentrt", "b", 20 + att, reward=0.0 if att else 1.0)
        _trial(tmp_path, "agentrt", "c", 30 + att, reward=1.0)
        _trial(tmp_path, "terminus-2", "a", 40 + att, reward=0.0)
        _trial(tmp_path, "terminus-2", "b", 50 + att, exc="RuntimeError")          # 基础设施失败：不进分母
        _trial(tmp_path, "terminus-2", "c", 60 + att, exc="AgentTimeoutError")     # agent 超时：算失败
    s = summarize(load_trials([tmp_path]))
    assert s["broken_tasks"] == ["c"]
    a, t = s["arms"]["agentrt[m/x]"], s["arms"]["terminus-2[m/x]"]   # api_base 不进配置名
    assert (a["valid"], a["success"], a["k"]) == (4, 3, 2) and a["pass^k"] == 0.5 and a["pass@k"] == 1.0
    assert (t["valid"], t["infra_failures"], t["success"]) == (2, 2, 0)
    c = s["comparisons"]["terminus-2[m/x]"]
    assert c["vs"] == "agentrt[m/x]" and c["pairs"] == 2 and (c["b"], c["c"]) == (2, 0)
    assert "已从所有配置剔除" in to_markdown(s)


def test_tb2_stratified_subset(tmp_path):
    from agentrt.bench.tb2 import select
    for i, d in enumerate(["easy"] * 2 + ["medium"] * 6 + ["hard"] * 4):
        (tmp_path / f"t{i:02d}").mkdir()
        (tmp_path / f"t{i:02d}" / "task.toml").write_text(f'[metadata]\ndifficulty = "{d}"\n')
    picked = select(tmp_path, 6, seed=1)
    assert len(picked) == 6 and picked == select(tmp_path, 6, seed=1)   # 确定性
    diffs = [("easy" if int(p[1:]) < 2 else "medium" if int(p[1:]) < 8 else "hard") for p in picked]
    assert (diffs.count("easy"), diffs.count("medium"), diffs.count("hard")) == (1, 3, 2)
    assert len(select(tmp_path, 99)) == 12


def test_agent_that_never_reached_the_model_is_infra_not_fail():
    from agentrt.bench.report import classify
    base = {"verifier_result": {"rewards": {"reward": 0.0}}}
    assert classify(base) == ("fail", 0.0)
    errored = {**base, "agent_result": {"metadata": {"status": "error"}}}
    assert classify(errored) == ("infra", 0.0)
    passed = {"verifier_result": {"rewards": {"reward": 1.0}}, "agent_result": {"metadata": {"status": "error"}}}
    assert classify(passed) == ("ok", 1.0)

    def meta(**m):
        return {**base, "agent_result": {"metadata": {"status": "error", **m}}}
    assert classify(meta(error="RetryableError: URLError: SSL EOF", turns=3)) == ("infra", 0.0)
    assert classify(meta(error="LLMError: HTTP 401: bad key", turns=0)) == ("infra", 0.0)
    assert classify(meta(error="LLMError: HTTP 400: invalid request", turns=12)) == ("fail", 0.0)
    assert classify(meta(error="ContextLengthError: too long", turns=80)) == ("fail", 0.0)

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from agentrt.llm import ScriptedLLM
from agentrt.web import WebSession, make_handler
from http.server import ThreadingHTTPServer


@pytest.fixture
def server(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.log").write_text("x")
    llm = ScriptedLLM(['<tool_call>{"name":"trash","arguments":{"paths":["a.log"]}}</tool_call>', "删好了"])
    sess = WebSession(llm, ws, "default", tmp_path / "s")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(sess, "tok", port, _runs(tmp_path / "runs"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield sess, port, ws
    httpd.shutdown()


def _runs(root):
    """一次跑分的最小目录结构：jobs/<job>/<trial>/{result.json, agent/agentrt.jsonl, verifier/test-stdout.txt}。"""
    trial = root / "tb2-x" / "jobs" / "job1" / "regex-chess__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "result.json").write_text(json.dumps({
        "trial_name": "regex-chess__abc", "task_name": "terminal-bench/regex-chess",
        "config": {"agent": {"name": "agentrt", "model_name": "openrouter/m", "kwargs": {"max_tokens": 64000}}},
        "agent_info": {"name": "agentrt"}, "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 80, "n_output_tokens": 10, "cost_usd": 0.01,
                         "metadata": {"status": "end_turn", "turns": 2, "time_budget": 600.0}},
        "agent_execution": {"started_at": "2026-11-02T08:00:00Z", "finished_at": "2026-11-02T08:01:30Z"}}))
    (trial / "verifier" / "test-stdout.txt").write_text("1 passed")
    events = [
        {"ts": 100.0, "type": "session_start", "task": "do it", "model": "m", "deadline": 700.0},
        {"ts": 130.0, "type": "span", "kind": "llm", "serial": 1, "seconds": 30, "completion_tokens": 64000,
         "finish_reason": "length"},
        {"ts": 130.0, "type": "dropped_reply", "reasoning": "想了很久"},
        {"ts": 131.0, "type": "span", "kind": "llm", "serial": 2, "seconds": 1, "finish_reason": "tool_calls", "think": False},
        {"ts": 131.0, "type": "message", "message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"ts": 131.0, "type": "tool_start", "call_id": "c1", "name": "bash", "arguments": {"command": "ls"}},
        {"ts": 132.0, "type": "tool_result", "call_id": "c1", "name": "bash", "content": "x" * 10000, "seconds": 1},
        {"ts": 133.0, "type": "message", "message": {"role": "user", "content": "（时间提醒）", "kind": "harness"}},
        {"ts": 190.0, "type": "end", "status": "end_turn", "final": "好了"},
    ]
    (trial / "agent" / "agentrt.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return root


def call(port, path, body=None, tok="tok"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}{'&' if '?' in path else '?'}t={tok}",
                                 method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None)
    return json.loads(urllib.request.urlopen(req).read())


def wait_for(sess, kind, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        for e in list(sess.events):
            if e["kind"] == kind:
                return e
        time.sleep(0.02)
    raise AssertionError(f"没等到 {kind}: {[e['kind'] for e in sess.events]}")


def test_token_required(server):
    _, port, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(port, "/api/state", tok="wrong")
    assert e.value.code == 403


def test_trash_requires_approval_then_runs(server):
    sess, port, ws = server
    call(port, "/api/chat", {"text": "删掉 a.log"})
    ap = wait_for(sess, "approval")
    assert ap["tool"] == "trash" and (ws / "a.log").exists()          # 还没批准，文件还在
    call(port, "/api/approve", {"id": ap["id"], "ok": True})
    assert wait_for(sess, "final")["text"] == "删好了"
    import os
    from pathlib import Path
    assert not (ws / "a.log").exists() and (Path(os.environ["AGENTRT_TRASH_DIR"]) / "a.log").exists()


def test_denied_keeps_file(server):
    sess, port, ws = server
    call(port, "/api/chat", {"text": "删掉 a.log"})
    ap = wait_for(sess, "approval")
    call(port, "/api/approve", {"id": ap["id"], "ok": False})
    wait_for(sess, "final")
    assert (ws / "a.log").exists()
    assert any(e["kind"] == "result" and e["denied"] for e in sess.events)


def test_config_rejects_missing_dir(server):
    _, port, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(port, "/api/config", {"workspace": "/no/such/dir", "mode": "default"})
    assert e.value.code == 400


def test_web_resume_replays_history(tmp_path):
    from agentrt.session import EventLog
    ws = tmp_path / "ws"
    ws.mkdir()
    log = EventLog.new(tmp_path / "s")
    s1 = WebSession(ScriptedLLM(["第一轮答复"]), ws, "default", tmp_path / "s")
    s1.agent.log = log
    s1.agent._chat_started = False
    s1.agent.send("你好")
    s2 = WebSession(ScriptedLLM([]), ws, "default", tmp_path / "s2", resume=log.path)
    kinds = [e["kind"] for e in s2.events]
    assert "user" in kinds and "final" in kinds and not s2.busy
    assert next(e for e in s2.events if e["kind"] == "final")["text"] == "第一轮答复"


def test_config_switches_model(server):
    sess, port, ws = server
    call(port, "/api/config", {"workspace": str(ws), "mode": "default", "model": "qwen/qwen3.7-flash"})
    assert sess.llm.model == "qwen/qwen3.7-flash" and call(port, "/api/state")["model"] == "qwen/qwen3.7-flash"
    with pytest.raises(urllib.error.HTTPError):
        call(port, "/api/config", {"workspace": str(ws), "mode": "default", "model": "two words"})


def test_runs_pages(server):
    _, port, _ = server
    [run] = call(port, "/api/runs")["runs"]
    assert run["name"] == "tb2-x" and run["arms"][0]["success"] == 1 and run["arms"][0]["cost_usd"] == 0.01
    [row] = call(port, "/api/run?name=tb2-x")["trials"]
    assert row["task"] == "regex-chess" and row["outcome"] == "ok" and row["agent_seconds"] == 90.0
    t = call(port, "/api/trial?run=tb2-x&id=" + urllib.request.quote(row["id"]))
    kinds = [e["kind"] for e in t["timeline"]]
    assert kinds == ["start", "llm", "dropped", "llm", "assistant", "tool", "result", "harness", "end"]
    assert t["timeline"][0]["budget"] == 600 and t["timeline"][3]["think"] is False
    assert len(t["timeline"][6]["content"]) < 5000 and t["verifier"] == "1 passed"


def test_runs_pages_stay_inside_runs_dir(server):
    _, port, _ = server
    for path in ("/api/run?name=..", "/api/run?name=tb2-x/jobs", "/api/trial?run=tb2-x&id=../../..",
                 "/api/trial?run=..&id=tb2-x/jobs/job1/regex-chess__abc"):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(port, path)
        assert e.value.code == 404, path

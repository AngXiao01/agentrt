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
    llm = ScriptedLLM(['<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"hi"}}</tool_call>', "写好了"])
    sess = WebSession(lambda cfg: llm, tmp_path / "s", {"workspace": str(ws)})
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


def wait_for(sess, kind, timeout=5, item=None):
    end = time.time() + timeout
    while time.time() < end:
        for e in list(sess.events):
            if e["kind"] == kind and (item is None or e.get("item", {}).get("kind") == item):
                return e
        time.sleep(0.02)
    raise AssertionError(f"没等到 {kind}/{item}: {[(e['kind'], e.get('item', {}).get('kind')) for e in sess.events]}")


def test_token_required(server):
    _, port, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(port, "/api/state", tok="wrong")
    assert e.value.code == 403


def test_write_requires_approval_then_runs(server):
    sess, port, ws = server
    call(port, "/api/chat", {"text": "写一个 a.txt"})
    ap = wait_for(sess, "approval")
    assert ap["tool"] == "write_file" and not (ws / "a.txt").exists()     # 还没批准，文件还没写
    call(port, "/api/approve", {"id": ap["id"], "ok": True})
    wait_for(sess, "ev", item="end")
    assert (ws / "a.txt").read_text() == "hi"
    kinds = [e["item"]["kind"] for e in sess.events if e["kind"] == "ev"]
    assert kinds[:2] == ["start", "user"] and "llm" in kinds and "tool" in kinds and "result" in kinds
    assert [e["item"]["content"] for e in sess.events
            if e["kind"] == "ev" and e["item"]["kind"] == "assistant" and e["item"]["content"]][-1] == "写好了"


def test_denied_keeps_file(server):
    sess, port, ws = server
    call(port, "/api/chat", {"text": "写一个 a.txt"})
    ap = wait_for(sess, "approval")
    call(port, "/api/approve", {"id": ap["id"], "ok": False})
    wait_for(sess, "ev", item="end")
    assert not (ws / "a.txt").exists()
    assert any(e["kind"] == "ev" and e["item"]["kind"] == "result" and e["item"]["denied"] for e in sess.events)


def test_config_rejects_bad_settings(server):
    sess, port, ws = server
    for bad in ({"workspace": "/no/such/dir"}, {"mode": "yolo"}, {"sandbox": "vm:x"}, {"extra_body": "[1]"},
                {"model": "two words"}, {"max_tokens": 0}, {"mcp_config": "/no/such.json"}):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(port, "/api/config", {"config": bad})
        assert e.value.code == 400, bad
    assert sess.config["workspace"] == str(ws.resolve())


def test_config_starts_new_session(server):
    sess, port, ws = server
    old = sess.agent.log.session_id
    call(port, "/api/config", {"config": {"model": "qwen/qwen3.7-flash", "mode": "auto", "max_turns": 7,
                                          "extra_body": '{"provider": {"order": ["wafer/us"]}}'}})
    st = call(port, "/api/state")
    assert st["session_id"] != old and st["config"]["mode"] == "auto" and sess.agent.max_turns == 7
    assert st["config"]["extra_body"] == {"provider": {"order": ["wafer/us"]}}
    assert st["config"]["model"] == "scripted"      # 页面显示的是 LLM 实际用的模型 id
    assert sess.events[-1]["kind"] == "clear"


def test_sandbox_toolset(server):
    sess, port, ws = server
    call(port, "/api/config", {"config": {"sandbox": "local"}})
    st = call(port, "/api/state")
    assert st["config"]["sandbox"] == "local" and st["sandbox"] and "edit_file" in st["tools"]
    assert sess.agent.sandbox is not None and "通用 agent" in sess.agent.system


def test_sessions_list_and_open(server):
    sess, port, ws = server
    call(port, "/api/chat", {"text": "写一个 a.txt"})
    call(port, "/api/approve", {"id": wait_for(sess, "approval")["id"], "ok": True})
    wait_for(sess, "ev", item="end")
    sid = sess.agent.log.session_id
    while sess.busy:
        time.sleep(0.02)
    [row] = [s for s in call(port, "/api/sessions")["sessions"] if s["id"] == sid]
    assert row["task"] == "写一个 a.txt" and row["status"] == "end_turn"
    detail = call(port, "/api/session?id=" + sid)
    assert [e["kind"] for e in detail["timeline"]][:2] == ["start", "user"]
    call(port, "/api/new", {})
    assert sess.agent.log.session_id != sid
    call(port, "/api/open", {"id": sid})
    assert sess.agent.log.session_id == sid and len(sess.agent._live) > 2   # 上下文从日志重建
    replayed = [e for e in sess.events[sess._since:] if e["kind"] == "ev"]
    assert replayed and all(e.get("replay") for e in replayed) and any(e["item"]["kind"] == "user" for e in replayed)
    for bad in ("../x", "nope"):
        with pytest.raises(urllib.error.HTTPError):
            call(port, "/api/open", {"id": bad})
        with pytest.raises(urllib.error.HTTPError):
            call(port, "/api/session?id=" + bad)


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

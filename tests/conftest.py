import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrt.context import ContextManager  # noqa: E402
from agentrt.llm import ScriptedLLM  # noqa: E402
from agentrt.loop import Agent  # noqa: E402
from agentrt.permissions import PermissionEngine, deny_all_approver  # noqa: E402
from agentrt.session import EventLog  # noqa: E402


def call(name, **args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'


@pytest.fixture
def make_agent(tmp_path):
    def _make(steps, mode="auto", approver=deny_all_approver, session="s1", **kw):
        llm = kw.pop("llm", None) or ScriptedLLM(steps)
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        log = EventLog(tmp_path / "sessions" / f"{session}.jsonl")
        cm = kw.pop("context", None) or ContextManager(llm)
        perms = kw.pop("permissions", None) or PermissionEngine(mode)
        agent = Agent(llm, ws, log, permissions=perms, approver=approver, context=cm, **kw)
        return agent, llm, ws, log
    return _make


class FakeVLLM:
    """最小的 OpenAI 兼容服务端：按队列返回预设响应，记录收到的请求。"""

    def __init__(self):
        self.queue = []      # 每项 (status, body_dict)
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake-gemma", "max_model_len": 8192}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(n)))
                status, body = outer.queue.pop(0) if outer.queue else (200, _completion("（空）"))
                self._send(status, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def _completion(content, tool_calls=None, prompt_tokens=100, cached=0):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


@pytest.fixture
def fake_vllm():
    srv = FakeVLLM()
    yield srv
    srv.close()


completion = _completion


@pytest.fixture(autouse=True)
def _isolated_trash(tmp_path, monkeypatch):
    """测试里绝不碰真实的系统废纸篓。"""
    monkeypatch.setenv("AGENTRT_TRASH_DIR", str(tmp_path / "_trash"))

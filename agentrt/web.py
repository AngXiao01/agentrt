"""Web 界面：浏览器里对话，让助手搜索 / 统计 / 整理 / 删除符合条件的文件；另有“跑分”页浏览 runs/ 下的 TB2 结果。

只依赖标准库。服务只监听 127.0.0.1，并要求启动时随机生成的 token（防止本机其他网页偷偷调用）。
进度通过 SSE 实时推送；会修改文件的操作会弹出确认卡片，等你点“允许”才执行。
"""
from __future__ import annotations

import json
import secrets
import threading
import uuid
from typing import Optional
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .assistant import build_assistant
from .bench import browse
from .context import ContextManager
from .permissions import MODES, PermissionEngine
from .session import EventLog

APPROVE_TIMEOUT = 600


class WebSession:
    def __init__(self, llm, workspace: Path, mode: str, sessions_dir: Path, resume: Optional[Path] = None,
                 extra_tools=None, bash: bool = True):
        self.llm, self.sessions_dir, self.extra_tools, self.bash = llm, sessions_dir, extra_tools, bash
        self.events: list[dict] = []
        self.cv = threading.Condition()
        self.pending: dict[str, dict] = {}
        self.busy = False
        self.agent = None
        if resume:
            self.resume_from(Path(resume), mode)
        else:
            self.configure(workspace, mode)

    # -------------------------------------------------------- 事件流
    def emit(self, kind: str, **data) -> None:
        with self.cv:
            self.events.append({"i": len(self.events), "kind": kind, **data})
            self.cv.notify_all()

    def wait_events(self, after: int, timeout: float = 15.0) -> list[dict]:
        with self.cv:
            if len(self.events) <= after + 1:
                self.cv.wait(timeout)
            return self.events[after + 1:]

    # -------------------------------------------------------- 配置
    def configure(self, workspace: Path, mode: str, log: Optional[EventLog] = None) -> None:
        log = log or EventLog.new(self.sessions_dir)
        self.agent = build_assistant(
            self.llm, workspace, log, permissions=PermissionEngine(mode), approver=self._approve,
            context=ContextManager(self.llm, max_context_tokens=60000), on_event=self._on_agent_event,
            max_turns=40, extra_tools=self.extra_tools, bash=self.bash)

    def resume_from(self, path: Path, mode: str) -> None:
        """从事件日志恢复：重建上下文、把历史回放到页面；上一轮没做完就接着做（副作用至多一次的语义见 session.py）。"""
        log = EventLog(path)
        events = log.events()
        start = next((e for e in events if e.get("type") == "session_start"), None)
        if not start:
            raise SystemExit(f"日志里没有 session_start：{path}")
        self.configure(Path(start["workspace"]), mode, log)
        for ui in replay_ui_events(events):
            self.emit(**ui)
        interrupted = self.agent.restore()
        self.emit("system", text=f"已恢复会话 {log.session_id}（{len(self.agent._live)} 条消息）", state=self.state())
        if interrupted:
            self.busy = True
            self.emit("system", text="上一轮在中途被中断，正在从断点继续（有副作用且结果未知的操作不会重跑）")
            threading.Thread(target=self._run, args=(None,), daemon=True).start()

    def state(self) -> dict:
        a = self.agent
        return {"workspace": str(a.ctx.workspace), "mode": a.perms.mode, "model": getattr(self.llm, "model", "?"),
                "busy": self.busy, "modes": list(MODES)}

    def set_config(self, workspace: str, mode: str, model: str = "") -> str:
        if self.busy:
            return "助手正在工作，稍后再改"
        ws = Path(workspace).expanduser()
        if not ws.is_dir():
            return f"目录不存在：{workspace}"
        if mode not in MODES:
            return f"未知模式：{mode}"
        model = model.strip()
        if model and (len(model) > 200 or any(c.isspace() for c in model)):
            return f"模型 id 不对：{model}"
        if model and model != self.llm.model:   # 同一个 LLM 对象换模型 id；OpenRouter 上 id 不对会在第一次请求时报错
            self.llm.model = model
            if hasattr(self.llm, "_wanted"):
                self.llm._wanted = model
        self.configure(ws.resolve(), mode)
        self.emit("system", text=f"工作范围：{ws.resolve()}　模式：{mode}　模型：{self.llm.model}", state=self.state())
        return ""

    # -------------------------------------------------------- 运行
    def start(self, text: str) -> str:
        with self.cv:
            if self.busy:
                return "上一条请求还没做完"
            self.busy = True
        self.emit("user", text=text)
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
        return ""

    def _run(self, text: str) -> None:
        try:
            res = self.agent.send(text) if text is not None else self.agent.resume()
            text = "已中断。" if res.status == "interrupted" else res.final or f"（没有最终答复：{res.status} {res.error}）"
            self.emit("final", text=text, status=res.status,
                      stats={"calls": res.llm_calls, "tools": res.tool_calls, "seconds": round(res.wall_seconds, 1),
                             "tokens": res.prompt_tokens + res.completion_tokens})
        except Exception as e:  # noqa: BLE001
            self.emit("final", text=f"出错了：{type(e).__name__}: {e}", status="error", stats={})
        finally:
            self.busy = False
            self.emit("idle", state=self.state())

    def stop(self) -> None:
        """中断：让 agent 在下一个检查点停下，并把正在等的确认卡片都按拒绝处理。"""
        self.agent.cancel()
        for slot in list(self.pending.values()):
            slot["ok"] = False
            slot["event"].set()

    def clear(self) -> None:
        if not self.busy:
            self.agent.reset()
            self.emit("clear")

    def _on_agent_event(self, e: dict) -> None:
        t = e.get("type")
        if t == "tool_start":
            self.emit("tool", id=e["call_id"], name=e["name"], args=e["arguments"])
        elif t == "tool_result":
            self.emit("result", id=e["call_id"], name=e["name"], content=e["content"][:3000],
                      error=bool(e.get("is_error")), denied=e.get("behavior") == "deny")
        elif t == "compact":
            self.emit("system", text="对话较长，已自动压缩早期内容")
        elif t == "span":
            self.emit("span", n=e.get("serial"), seconds=e.get("seconds"), tokens=(e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0))

    # -------------------------------------------------------- 审批
    def _approve(self, tool, args, decision) -> bool:
        aid = uuid.uuid4().hex[:8]
        slot = {"event": threading.Event(), "ok": False, "always": False}
        self.pending[aid] = slot
        self.emit("approval", id=aid, tool=tool.name, args=args, reason=decision.reason,
                  can_always=decision.source == "mode" and tool.risk != "high")
        slot["event"].wait(APPROVE_TIMEOUT)  # 超时按拒绝处理（fail-closed）
        self.pending.pop(aid, None)
        if slot["ok"] and slot["always"] and decision.source == "mode" and tool.risk != "high":
            self.agent.perms.allow.append(tool.name)
        self.emit("approval_done", id=aid, ok=slot["ok"])
        return slot["ok"]

    def answer(self, aid: str, ok: bool, always: bool) -> bool:
        slot = self.pending.get(aid)
        if not slot:
            return False
        slot["ok"], slot["always"] = ok, always
        slot["event"].set()
        return True


def replay_ui_events(events: list[dict]) -> list[dict]:
    """把磁盘上的事件日志转成页面事件，页面刷新 / 重启后能看到完整历史。"""
    out = []
    for e in events:
        t = e.get("type")
        if t == "turn_start":
            out.append({"kind": "user", "text": e["text"]})
        elif t == "session_start":
            out.append({"kind": "user", "text": e["task"]}) if e.get("task") and not any(
                x.get("type") == "turn_start" for x in events) else None
        elif t == "tool_start":
            out.append({"kind": "tool", "id": e["call_id"], "name": e["name"], "args": e["arguments"]})
        elif t == "tool_result":
            out.append({"kind": "result", "id": e["call_id"], "name": e["name"], "content": e["content"][:3000],
                        "error": bool(e.get("is_error")), "denied": e.get("behavior") == "deny"})
        elif t == "end":
            st = e.get("stats") or {}
            out.append({"kind": "final", "text": e.get("final", ""), "status": e.get("status"),
                        "stats": {"calls": st.get("llm_calls", 0), "tools": st.get("tool_calls", 0),
                                  "seconds": st.get("wall_seconds", 0),
                                  "tokens": st.get("prompt_tokens", 0) + st.get("completion_tokens", 0)}})
    return out


def make_handler(sess: WebSession, token: str, port: int, runs_dir: Optional[Path] = None):
    html = resources.files("agentrt").joinpath("web_ui.html").read_text(encoding="utf-8")
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def _authed(self) -> bool:
            if self.headers.get("Host") not in hosts:  # 防 DNS rebinding
                self._send(403, b"bad host", "text/plain")
                return False
            if parse_qs(urlparse(self.path).query).get("t", [""])[0] != token:
                self._send(403, b"bad token", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._authed():
                return
            url = urlparse(self.path)
            path, q = url.path, {k: v[0] for k, v in parse_qs(url.query).items()}
            if path == "/":
                self._send(200, html.encode(), "text/html")
            elif path == "/api/state":
                self._json({**sess.state(), "runs_dir": str(runs_dir) if runs_dir else None})
            elif path == "/api/runs":
                self._json({"runs": browse.list_runs(runs_dir) if runs_dir else []})
            elif path in ("/api/run", "/api/trial"):
                data = None
                if runs_dir and path == "/api/run":
                    data = browse.run_detail(runs_dir, q.get("name", ""))
                elif runs_dir:
                    data = browse.trial_detail(runs_dir, q.get("run", ""), q.get("id", ""))
                if data is None:
                    self._json({"error": "没有这个跑分"}, 404)
                else:
                    self._json(data)
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self):
            last = self.headers.get("Last-Event-ID")
            after = int(last) if last and last.isdigit() else -1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    evs = sess.wait_events(after)
                    if not evs:
                        self.wfile.write(b": keepalive\n\n")
                    for e in evs:
                        after = e["i"]
                        self.wfile.write(f"id: {e['i']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_POST(self):
            if not self._authed():
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            path = urlparse(self.path).path
            if path == "/api/chat":
                text = str(body.get("text", "")).strip()
                err = "内容为空" if not text else sess.start(text)
            elif path == "/api/approve":
                err = "" if sess.answer(str(body.get("id")), bool(body.get("ok")), bool(body.get("always"))) \
                    else "这个确认已失效"
            elif path == "/api/config":
                err = sess.set_config(str(body.get("workspace", "")), str(body.get("mode", "default")),
                                      str(body.get("model") or ""))
            elif path == "/api/stop":
                sess.stop()
                err = ""
            elif path == "/api/clear":
                sess.clear()
                err = ""
            else:
                return self._send(404, b"not found", "text/plain")
            self._json({"error": err} if err else {"ok": True}, 400 if err else 200)

    return H


def serve(llm, workspace: Path, mode: str, sessions_dir: Path, port: int = 8765, open_browser: bool = True,
          resume: Optional[Path] = None, extra_tools=None, bash: bool = True, runs_dir: Optional[Path] = None) -> None:
    sess = WebSession(llm, workspace, mode, sessions_dir, resume, extra_tools, bash)
    token = secrets.token_urlsafe(16)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sess, token, port, runs_dir))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/?t={token}"
    print(f"agentrt 网页已启动：{url}\n模型 {llm.model}　范围 {sess.agent.ctx.workspace}　跑分 {runs_dir}　Ctrl+C 退出", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

"""Web 控制台：浏览器里和 agentrt 的通用 agent 多轮对话，看实时轨迹（每次模型调用、思考、工具调用和结果、
截断、压缩），翻历史会话并接着做；另有“跑分”页浏览 runs/ 下的 TB2 结果。

agent 怎么组装全在页面的“设置”里：模型、输出上限、请求体附加字段、工具集（本地工具，或 local / bwrap / docker /
已有容器里的 sandbox 工具集）、权限模式、轮数和上下文上限、MCP 配置。改设置就开一个新会话。

只依赖标准库。服务只监听 127.0.0.1，并要求启动时随机生成的 token（防止本机其他网页偷偷调用）。
进度通过 SSE 实时推送；需要确认的工具调用会弹出确认卡片，等你点“允许”才执行。
"""
from __future__ import annotations

import json
import secrets
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

from .bench import browse
from .context import ContextManager
from .loop import Agent
from .permissions import MODES, PermissionEngine
from .session import EventLog

APPROVE_TIMEOUT = 600
SANDBOX_INTRO = "你是一个通用 agent，和用户多轮对话来完成任务。你的工具都在这个环境里执行：{env}。"

DEFAULTS = {"workspace": ".", "mode": "default", "model": "", "max_tokens": 16000, "temperature": 0.6,
            "native_tools": True, "extra_body": {}, "sandbox": "", "network": False, "max_turns": 100,
            "max_context": 100000, "bash_timeout": 120.0, "mcp_config": ""}


def check_config(raw: dict, base: dict) -> tuple[dict, str]:
    """把页面提交的设置并到 base 上，逐项校验；返回 (设置, 错误)。"""
    cfg = {**base, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    try:
        ws = Path(str(cfg["workspace"])).expanduser()
        if not ws.is_dir():
            return cfg, f"目录不存在：{cfg['workspace']}"
        cfg["workspace"] = str(ws.resolve())
        if cfg["mode"] not in MODES:
            return cfg, f"未知权限模式：{cfg['mode']}"
        cfg["model"] = str(cfg["model"] or "").strip()
        if len(cfg["model"]) > 200 or any(c.isspace() for c in cfg["model"]):
            return cfg, f"模型 id 不对：{cfg['model']}"
        for k, lo, hi in (("max_tokens", 1, 1_000_000), ("max_turns", 1, 10_000), ("max_context", 2000, 2_000_000)):
            cfg[k] = int(cfg[k])
            if not lo <= cfg[k] <= hi:
                return cfg, f"{k} 要在 {lo}～{hi} 之间"
        cfg["temperature"], cfg["bash_timeout"] = float(cfg["temperature"]), float(cfg["bash_timeout"])
        cfg["native_tools"], cfg["network"] = bool(cfg["native_tools"]), bool(cfg["network"])
        if isinstance(cfg["extra_body"], str):
            cfg["extra_body"] = json.loads(cfg["extra_body"]) if cfg["extra_body"].strip() else {}
        if not isinstance(cfg["extra_body"], dict):
            return cfg, "请求体附加字段要是一个 JSON 对象"
        sb = cfg["sandbox"] = str(cfg["sandbox"] or "").strip()
        if sb and sb not in ("local", "bwrap") and not sb.startswith(("docker:", "container:")):
            return cfg, f"不认识的 sandbox：{sb}（留空 / local / bwrap / docker:<镜像> / container:<id>）"
        mcp = cfg["mcp_config"] = str(cfg["mcp_config"] or "").strip()
        if mcp and not Path(mcp).expanduser().is_file():
            return cfg, f"MCP 配置文件不存在：{mcp}"
    except (TypeError, ValueError) as e:
        return cfg, f"设置不对：{e}"
    return cfg, ""


class WebSession:
    """一个页面对应一个当前会话：agent + 它的事件日志 + 推给页面的事件流。"""

    def __init__(self, make_llm: Callable[[dict], object], sessions_dir: Path, config: Optional[dict] = None,
                 resume: Optional[str] = None):
        self.make_llm, self.sessions_dir = make_llm, Path(sessions_dir)
        self.events: list[dict] = []
        self.cv = threading.Condition()
        self.pending: dict[str, dict] = {}
        self.busy = False
        self.agent: Optional[Agent] = None
        self.config: dict = {}
        self._mcp: list = []
        self._t0: Optional[float] = None
        self._since = 0   # 当前会话在事件流里从哪一条开始（新打开的页面只拿这之后的）
        cfg, err = check_config(config or {}, DEFAULTS)
        if err:
            raise SystemExit(err)
        err = self.open(resume) if resume else self.apply(cfg)
        if err:
            raise SystemExit(err)

    # -------------------------------------------------------- 事件流
    def emit(self, kind: str, **data) -> None:
        with self.cv:
            if kind == "clear":
                self._since = len(self.events)
            self.events.append({"i": len(self.events), "kind": kind, **data})
            self.cv.notify_all()

    def wait_events(self, after: int, timeout: float = 15.0) -> list[dict]:
        with self.cv:
            if len(self.events) <= after + 1:
                self.cv.wait(timeout)
            return self.events[after + 1:]

    # -------------------------------------------------------- 组装 agent
    def _build(self, cfg: dict, log: EventLog, clients: list) -> Agent:
        """按设置组装 agent；起的 MCP 客户端放进 clients，组装失败时由调用方关掉。"""
        llm = self.make_llm(cfg)
        extra = []
        if cfg["mcp_config"]:
            from .mcp import load_mcp_tools
            extra, started, warns = load_mcp_tools(str(Path(cfg["mcp_config"]).expanduser()))
            clients += started
            for w in warns:
                self.emit("system", text=f"MCP：{w}")
        common = dict(permissions=PermissionEngine(cfg["mode"]), approver=self._approve,
                      context=ContextManager(llm, max_context_tokens=cfg["max_context"]), max_turns=cfg["max_turns"],
                      bash_timeout=cfg["bash_timeout"], on_event=self._on_agent_event)
        if cfg["sandbox"]:
            from .coding_tools import coding_tools
            from .sandboxes import make_sandbox
            sb = make_sandbox(cfg["sandbox"], Path(cfg["workspace"]), network=cfg["network"])
            return Agent(llm, cfg["workspace"], log, tools=coding_tools() + extra, sandbox=sb,
                         prompt={"intro": SANDBOX_INTRO.format(env=sb.describe())}, **common)
        from .tools import default_tools
        return Agent(llm, cfg["workspace"], log, tools=default_tools() + extra, **common)

    def _close(self) -> None:
        if self.agent is not None and self.agent.sandbox is not None:
            self.agent.sandbox.close()
        for c in self._mcp:
            c.close()
        self.agent, self._mcp = None, []

    def _switch(self, cfg: dict, log: EventLog) -> str:
        """换成新的 agent；组装失败（docker 起不来、MCP 配置坏了……）时保留原来的会话不动。"""
        clients: list = []
        try:
            agent = self._build(cfg, log, clients)
        except Exception as e:  # noqa: BLE001
            for c in clients:
                c.close()
            return f"组装 agent 失败：{type(e).__name__}: {e}"
        self._close()
        self.agent, self._mcp, self.config = agent, clients, {**cfg, "model": getattr(agent.llm, "model", cfg["model"])}
        self._t0 = None
        return ""

    def apply(self, raw: dict) -> str:
        """按新设置开一个新会话。"""
        if self.busy:
            return "agent 正在工作，先中断或等它做完"
        cfg, err = check_config(raw, self.config or DEFAULTS)
        if err:
            return err
        err = self._switch(cfg, EventLog.new(self.sessions_dir))
        if err:
            return err
        self.emit("clear", state=self.state())
        return ""

    def new(self) -> str:
        return self.apply({})

    def open(self, ref: str) -> str:
        """接着一个历史会话做：用当前设置（工作目录取会话里记录的）重建 agent，把历史回放到页面；
        上一轮没做完就接着做（副作用至多一次的语义见 session.py）。"""
        if self.busy:
            return "agent 正在工作，先中断或等它做完"
        path = session_file(self.sessions_dir, ref)
        if path is None:
            return f"找不到会话：{ref}"
        log = EventLog(path)
        events = log.events()
        start = next((e for e in events if e.get("type") == "session_start"), None)
        if not start:
            return "日志里没有 session_start，不能接着做"
        cfg, err = check_config({"workspace": start.get("workspace") or ".",
                                 "model": (self.config or {}).get("model") or start.get("model") or ""},
                                self.config or DEFAULTS)
        if err:
            return err
        err = self._switch(cfg, log)
        if err:
            return err
        self._t0 = events[0].get("ts") if events else None
        self.emit("clear", state=self.state())
        for item in browse.timeline(events):
            self.emit("ev", item=item, replay=True)   # 回放的历史：页面只展示，不当成正在进行的进度
        interrupted = self.agent.restore()
        self.emit("system", text=f"已打开会话 {log.session_id}，接着发消息就在这个会话里继续")
        if interrupted:
            self.busy = True
            self.emit("system", text="上一轮在中途被中断，正在从断点继续（有副作用且结果未知的操作不会重跑）")
            threading.Thread(target=self._run, args=(None,), daemon=True).start()
        return ""

    def state(self) -> dict:
        a = self.agent
        return {"config": self.config, "session_id": a.log.session_id if a else None, "busy": self.busy,
                "modes": list(MODES), "tools": sorted(a.tools) if a else [],
                "sandbox": a.sandbox.describe() if a and a.sandbox is not None else None}

    # -------------------------------------------------------- 运行
    def start(self, text: str) -> str:
        with self.cv:
            if self.busy:
                return "上一条请求还没做完"
            self.busy = True
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
        return ""

    def _run(self, text: Optional[str]) -> None:
        try:
            res = self.agent.send(text) if text is not None else self.agent.resume()
            if res.status not in ("end_turn", "interrupted", "already_ended"):
                self.emit("system", text=f"本轮没有正常结束：{res.status} {res.error}".strip())
        except Exception as e:  # noqa: BLE001
            self.emit("system", text=f"出错了：{type(e).__name__}: {e}")
        finally:
            self.busy = False
            self.emit("idle", state=self.state())

    def stop(self) -> None:
        """中断：让 agent 在下一个检查点停下，并把正在等的确认卡片都按拒绝处理。"""
        if self.agent is not None:
            self.agent.cancel()
        for slot in list(self.pending.values()):
            slot["ok"] = False
            slot["event"].set()

    def _on_agent_event(self, e: dict) -> None:
        if self._t0 is None:
            self._t0 = e.get("ts", 0)
        item = browse.timeline_item(e, self._t0)
        if item:
            self.emit("ev", item=item)

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


# ------------------------------------------------------------ 历史会话
def session_file(sessions_dir: Path, ref: str) -> Optional[Path]:
    root = Path(sessions_dir).resolve()
    p = (root / f"{ref}.jsonl").resolve()
    return p if p.parent == root and p.is_file() else None


def _first_last(path: Path) -> tuple[dict, dict]:
    """只读第一行和最后一行：会话日志可能很大，列表页不全读。"""
    with path.open("rb") as f:
        first = f.readline()
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65536))
        tail = f.read().splitlines()
    out = []
    for line in (first, tail[-1] if tail else b""):
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            out.append({})
    return out[0], out[1]


def list_sessions(sessions_dir: Path, limit: int = 100) -> list[dict]:
    root = Path(sessions_dir)
    files = sorted((f for f in root.glob("*.jsonl") if ".sub" not in f.stem),
                   key=lambda f: f.stat().st_mtime, reverse=True)[:limit] if root.is_dir() else []
    out = []
    for f in files:
        first, last = _first_last(f)
        out.append({"id": f.stem, "mtime": f.stat().st_mtime, "task": (first.get("task") or "")[:200],
                    "model": first.get("model", ""), "workspace": first.get("workspace", ""),
                    "status": last.get("status") if last.get("type") == "end" else "unfinished"})
    return out


def session_detail(sessions_dir: Path, ref: str) -> Optional[dict]:
    from .trace import load
    p = session_file(sessions_dir, ref)
    if p is None:
        return None
    events = load(p)
    return {"id": ref, "timeline": browse.timeline(events)}


# ------------------------------------------------------------ HTTP
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
                return self._send(200, html.encode(), "text/html")
            if path == "/api/stream":
                return self._stream()
            if path == "/api/state":
                return self._json({**sess.state(), "runs_dir": str(runs_dir) if runs_dir else None})
            if path == "/api/sessions":
                return self._json({"sessions": list_sessions(sess.sessions_dir)})
            if path == "/api/runs":
                return self._json({"runs": browse.list_runs(runs_dir) if runs_dir else []})
            data, found = None, True
            if path == "/api/session":
                data = session_detail(sess.sessions_dir, q.get("id", ""))
            elif path == "/api/run":
                data = browse.run_detail(runs_dir, q.get("name", "")) if runs_dir else None
            elif path == "/api/trial":
                data = browse.trial_detail(runs_dir, q.get("run", ""), q.get("id", "")) if runs_dir else None
            else:
                found = False
            if not found:
                self._send(404, b"not found", "text/plain")
            elif data is None:
                self._json({"error": "找不到"}, 404)
            else:
                self._json(data)

        def _stream(self):
            last = self.headers.get("Last-Event-ID")
            after = int(last) if last and last.isdigit() else sess._since - 1
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
                err = sess.apply(body.get("config") or {})
            elif path == "/api/new":
                err = sess.new()
            elif path == "/api/open":
                err = sess.open(str(body.get("id", "")))
            elif path == "/api/stop":
                sess.stop()
                err = ""
            else:
                return self._send(404, b"not found", "text/plain")
            self._json({"error": err} if err else {"ok": True}, 400 if err else 200)

    return H


def serve(make_llm, sessions_dir: Path, config: dict, port: int = 8765, open_browser: bool = True,
          resume: Optional[str] = None, runs_dir: Optional[Path] = None) -> None:
    sess = WebSession(make_llm, sessions_dir, config, resume)
    token = secrets.token_urlsafe(16)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sess, token, port, runs_dir))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/?t={token}"
    c = sess.config
    print(f"agentrt 控制台：{url}\n模型 {c['model']}　工作目录 {c['workspace']}　sandbox {c['sandbox'] or '无（本地工具）'}"
          f"　跑分 {runs_dir}　Ctrl+C 退出", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        sess._close()

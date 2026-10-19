"""MCP 客户端（stdio）：把已有的 MCP server 的工具接成 agentrt 的工具。只依赖标准库。

对照 Claude Code 源码 src/services/mcp/ 与 src/tools/MCPTool/：
- 配置格式沿用 `{"mcpServers": {名字: {"command", "args", "env"}}}`（~/.claude.json、.mcp.json 都是这个格式）。
- 工具名 `mcp__<server>__<tool>`（源码同样的命名），避免和内置工具重名。
- 只读（可并行、无需确认）要两个条件：工具声明了 `annotations.readOnlyHint`，**并且**配置里把这个 server 标成
  `"trusted": true`（agentrt 自己的字段）。MCP 规范说注解只是提示，除非 server 可信，不能当真——不然任何 server
  把删除工具标成只读，就能免确认执行。其余一律走正常权限流程（默认要确认）。
- 没有实现：HTTP / SSE 传输、OAuth、resources / prompts、list_changed 通知、工具延迟加载（源码的 ToolSearch）。
传输：换行分隔的 JSON-RPC 2.0（官方 Python SDK 的 stdio 传输就是这样）。server 的 env 只传给该子进程，
不进日志、不进 prompt；子进程只额外拿到 PATH / HOME / LANG。
"""
from __future__ import annotations

import atexit
import itertools
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from .tools import Tool, ToolContext, ToolError

PROTOCOL_VERSION = "2024-11-05"


class McpError(Exception):
    pass


class McpClient:
    def __init__(self, name: str, command: str, args: Optional[list] = None, env: Optional[dict] = None,
                 timeout: float = 30.0):
        self.name, self.timeout = name, timeout
        self._env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
        self._env.update({str(k): str(v) for k, v in (env or {}).items()})
        self._cmd = [command, *[str(a) for a in (args or [])]]
        self._ids = itertools.count(1)
        self._pending: dict[int, dict] = {}
        self._lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------ 进程与读线程
    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=self._env, text=True, bufsize=1,
                                         start_new_session=True)
        except OSError as e:
            raise McpError(f"启动失败：{e}") from e
        threading.Thread(target=self._read_loop, daemon=True).start()
        atexit.register(self.close)
        self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                    "clientInfo": {"name": "agentrt", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # server 偶尔会往 stdout 打日志，忽略非 JSON 行
            slot = self._pending.get(msg.get("id")) if "method" not in msg else None
            if slot:
                slot["msg"] = msg
                slot["event"].set()
        for slot in list(self._pending.values()):   # 进程退出：唤醒所有等待者
            slot["event"].set()

    def _send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise McpError(f"{self.name} 连接已断开：{e}") from e

    def request(self, method: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        rid = next(self._ids)
        slot = {"event": threading.Event(), "msg": None}
        self._pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            if not slot["event"].wait(timeout or self.timeout):
                raise McpError(f"{self.name}.{method} 超时（{timeout or self.timeout}s）")
        finally:
            self._pending.pop(rid, None)
        msg = slot["msg"]
        if msg is None:
            raise McpError(f"{self.name} 进程已退出")
        if "error" in msg:
            raise McpError(f"{msg['error'].get('message', msg['error'])}")
        return msg.get("result") or {}

    # ------------------------------------------------------------ 工具
    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, args: dict, timeout: float = 120.0) -> tuple[str, bool]:
        res = self.request("tools/call", {"name": name, "arguments": args}, timeout=timeout)
        parts = []
        for c in res.get("content", []):
            parts.append(c.get("text", "") if c.get("type") == "text" else f"[{c.get('type', '?')} 内容已省略]")
        return "\n".join(parts), bool(res.get("isError"))

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self.proc.kill()


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


class McpTool(Tool):
    """一个 MCP 工具。只有可信 server 的 annotations.readOnlyHint 才算数；其余当成会改动东西，要确认。"""
    risk = "medium"

    def __init__(self, client: McpClient, spec: dict, trusted: bool = False):
        self.client, self.remote_name = client, spec["name"]
        self.name = f"mcp__{_safe(client.name)}__{_safe(spec['name'])}"
        self.description = (spec.get("description") or "").strip().split("\n")[0][:300] or f"{client.name} 的 {spec['name']}"
        schema = spec.get("inputSchema") or {}
        self.parameters = {"type": "object", "properties": schema.get("properties", {}),
                           "required": schema.get("required", [])}
        self.declared_read_only = bool((spec.get("annotations") or {}).get("readOnlyHint"))
        self.read_only = trusted and self.declared_read_only

    def subject(self, args: dict) -> str:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)[:200]

    def validate(self, args: dict, ctx: ToolContext) -> Optional[str]:
        miss = [k for k in self.parameters["required"] if k not in args]
        return f"缺少必填参数 {', '.join(miss)}" if miss else None   # MCP 参数不是工作目录内的路径，不做路径检查

    def run(self, args: dict, ctx: ToolContext) -> str:
        try:
            text, is_err = self.client.call_tool(self.remote_name, args)
        except McpError as e:
            raise ToolError(str(e)) from e
        if is_err:
            raise ToolError(text or "MCP 工具返回错误")
        return text or "（空结果）"


def load_config(path: Path) -> dict:
    data = json.loads(Path(path).expanduser().read_text())
    return {k: v for k, v in (data.get("mcpServers") or {}).items() if isinstance(v, dict)}


def load_mcp_tools(path: str, only: Optional[list[str]] = None) -> tuple[list[Tool], list[McpClient], list[str]]:
    """启动配置里的 stdio server（可用 only 过滤名字），返回 (工具, 客户端, 警告)。单个 server 失败只告警不中断。"""
    tools: list[Tool] = []
    clients: list[McpClient] = []
    warns: list[str] = []
    for name, cfg in load_config(Path(path)).items():
        if only and name not in only:
            continue
        if cfg.get("type", "stdio") != "stdio" or not cfg.get("command"):
            warns.append(f"{name}：只支持 stdio，已跳过")
            continue
        c = McpClient(name, cfg["command"], cfg.get("args"), cfg.get("env"))
        try:
            c.start()
            specs = c.list_tools()
        except McpError as e:
            c.close()
            warns.append(f"{name}：{e}")
            continue
        clients.append(c)
        tools += [McpTool(c, s, trusted=cfg.get("trusted") is True) for s in specs]
    return tools, clients, warns

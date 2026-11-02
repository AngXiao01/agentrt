"""文件助手的命令行入口（在仓库根目录运行）：

    python3 -m examples.file_assistant --workspace ~/Downloads             # 终端交互，改动前确认
    python3 -m examples.file_assistant --workspace ~/Downloads --mode plan # 只读
    python3 -m examples.file_assistant --once "找出最大的 10 个文件"
    python3 -m examples.file_assistant --resume <会话id>                     # 崩溃后接着干
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agentrt.cli import APPROVERS, DEFAULT_SESSIONS, _printer, _report, _session_path
from agentrt.context import ContextManager
from agentrt.openrouter import OpenRouterLLM
from agentrt.permissions import MODES, PermissionEngine
from agentrt.session import EventLog

from .assistant import build_assistant, repl


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m examples.file_assistant", description="电脑文件助手（OpenRouter）")
    p.add_argument("--workspace", default=".", help="助手能管理的目录，如 ~/Downloads")
    p.add_argument("--once", metavar="请求", help="只执行一条请求就退出（非交互）")
    p.add_argument("--resume", metavar="会话", help="从会话日志断点继续（会话 id 或 jsonl 路径）")
    p.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    p.add_argument("--model", default=None, help="默认取 AGENTRT_MODEL / OPENROUTER_MODEL")
    p.add_argument("--native-tools", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--no-bash", action="store_true", help="不给助手 bash（默认有沙箱时提供）")
    p.add_argument("--mode", choices=MODES, default="default")
    p.add_argument("--approve", choices=list(APPROVERS), default="cli")
    p.add_argument("--allow", action="append", default=[])
    p.add_argument("--ask", action="append", default=[])
    p.add_argument("--deny", action="append", default=[])
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-context", type=int, default=60000)
    p.add_argument("--bash-timeout", type=float, default=30.0)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    log, ws, model = EventLog.new(a.sessions_dir), Path(a.workspace).expanduser().resolve(), a.model
    if a.resume:
        log = EventLog(_session_path(a.resume, a.sessions_dir))
        start = next((e for e in log.events() if e.get("type") == "session_start"), {})
        ws, model = Path(start.get("workspace") or ws), model or start.get("model")
    llm = OpenRouterLLM(model, native_tools=a.native_tools, temperature=a.temperature,
                        max_tokens=a.max_tokens, max_retries=a.retries)
    agent = build_assistant(llm, ws, log, permissions=PermissionEngine(a.mode, allow=a.allow, deny=a.deny, ask=a.ask),
                            approver=APPROVERS[a.approve], context=ContextManager(llm, max_context_tokens=a.max_context),
                            on_event=_printer(a.verbose), max_turns=a.max_turns, bash_timeout=a.bash_timeout,
                            bash=not a.no_bash)
    if a.resume:
        _report(agent.resume(), log)
    elif a.once:
        _report(agent.send(a.once), log)
    else:
        repl(agent)
        print(f"会话日志    {log.path}")


if __name__ == "__main__":
    main()

"""命令行入口。

    python -m agentrt ping   --base-url http://127.0.0.1:8000/v1
    python -m agentrt run    "任务描述" --workspace ./ws [--mode auto] [--approve cli]
    python -m agentrt run    "任务描述" --workspace ./ws --sandbox docker:python:3.12-slim   # coding 工具集 + 容器
    python -m agentrt assistant [--workspace ~/Downloads] [--once "请求"]
    python -m agentrt web [--workspace .] [--model ...] [--sandbox docker:python:3.12-slim] [--port 8765]
    python -m agentrt resume <session_id 或 jsonl 路径>
    python -m agentrt trace  <session_id 或 jsonl 路径> [--timeline]
    python -m agentrt sessions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .context import ContextManager
from .llm import LLMError, OpenAICompatLLM, RetryableError
from .loop import Agent
from .permissions import (MODES, PermissionEngine, allow_all_approver, cli_approver, deny_all_approver,
                          policy_approver)
from .session import EventLog
from .trace import format_summary, load, summarize, timeline

APPROVERS = {"cli": cli_approver, "deny": deny_all_approver, "policy": policy_approver, "all": allow_all_approver}
DEFAULT_SESSIONS = Path(os.environ.get("AGENTRT_SESSIONS", Path.home() / ".agentrt" / "sessions"))


KEY_FILE = Path.home() / ".agentrt" / "api_key"


def api_key() -> str:
    """优先环境变量 AGENTRT_API_KEY，其次 ~/.agentrt/api_key。"""
    if os.environ.get("AGENTRT_API_KEY"):
        return os.environ["AGENTRT_API_KEY"]
    try:
        return KEY_FILE.read_text().strip() or "EMPTY"
    except OSError:
        return "EMPTY"


def _llm(a) -> OpenAICompatLLM:
    return OpenAICompatLLM(a.base_url, model=a.model, api_key=api_key(), native_tools=a.native_tools,
                           merge_system=a.merge_system, temperature=a.temperature, max_tokens=a.max_tokens,
                           max_retries=a.retries, reasoning_replay=getattr(a, "reasoning_replay", "turn"),
                           extra_body=json.loads(a.extra_body) if getattr(a, "extra_body", None) else None)


def _printer(verbose: bool):
    def on_event(e: dict) -> None:
        t = e.get("type")
        if t == "span":
            print(f"  [llm#{e['serial']}] {e['seconds']:.1f}s  prompt={e.get('prompt_tokens')} "
                  f"cached={e.get('cached_tokens')} out={e.get('completion_tokens')}", file=sys.stderr)
        elif t == "tool_start":
            args = json.dumps(e["arguments"], ensure_ascii=False)
            print(f"  → {e['name']} {args[:160]}", file=sys.stderr)
        elif t == "tool_result":
            tag = "DENY" if e.get("behavior") == "deny" else ("ERR" if e.get("is_error") else "ok")
            body = e["content"] if verbose else e["content"][:160].replace("\n", " ⏎ ")
            print(f"  ← {e['name']} [{tag}] {body}", file=sys.stderr)
        elif t == "compact":
            print(f"  [compact:{e['reason']}] {e['before_tokens']} → {e['after_tokens']} tokens", file=sys.stderr)
        elif t == "message" and e["message"]["role"] == "assistant" and verbose:
            print(f"  assistant: {e['message']['content'][:400]}", file=sys.stderr)
    return on_event


def _session_path(ref: str, sessions_dir: Path) -> Path:
    p = Path(ref)
    if p.suffix == ".jsonl" and p.exists():
        return p
    p = sessions_dir / f"{ref}.jsonl"
    if not p.exists():
        sys.exit(f"找不到会话：{ref}")
    return p


def _agent(a, llm, log: EventLog, workspace) -> Agent:
    if getattr(a, "sandbox", None):   # coding 工具集：工具全部在 sandbox 里执行（sandboxes.py / coding_tools.py）
        from .coding_tools import build_coding_agent
        from .sandboxes import make_sandbox
        sb = make_sandbox(a.sandbox, Path(workspace).resolve(), network=a.network)
        return build_coding_agent(
            llm, sb, log, max_turns=a.max_turns, approver=APPROVERS[a.approve], on_event=_printer(a.verbose),
            permissions=PermissionEngine(a.mode, allow=a.allow, deny=a.deny, ask=a.ask),
            context=ContextManager(llm, max_context_tokens=llm.context_budget(a.max_context), enabled=not a.no_compact),
            bash_timeout=max(a.bash_timeout, 120.0), extra_tools=_mcp_tools(a))
    tools = None
    if getattr(a, "mcp_config", None):
        from .tools import default_tools
        tools = default_tools() + _mcp_tools(a)
    return Agent(llm, workspace, log, tools=tools,
                 permissions=PermissionEngine(a.mode, allow=a.allow, deny=a.deny, ask=a.ask),
                 approver=APPROVERS[a.approve],
                 context=ContextManager(llm, max_context_tokens=llm.context_budget(a.max_context),
                                        enabled=not a.no_compact),
                 max_turns=a.max_turns, bash_timeout=a.bash_timeout, on_event=_printer(a.verbose))


def _report(res, log: EventLog) -> None:
    print("\n" + "=" * 60)
    print(res.final or f"（无最终答复，状态 {res.status} {res.error}）")
    print("=" * 60)
    print(format_summary(summarize(log.events())))
    print(f"会话日志    {log.path}")


def cmd_ping(a) -> None:
    llm = _llm(a)
    print(f"模型：{llm.model}   max_model_len：{llm.max_model_len}   agent 上下文预算："
          f"{llm.context_budget(24000)} tokens")
    t0 = time.time()
    res = llm.chat([{"role": "user", "content": "只回复两个字母：OK"}])
    print(f"回复：{res.content!r}  耗时 {time.time() - t0:.2f}s")
    print(f"usage：{json.dumps(res.usage, ensure_ascii=False)}")
    if "prompt_tokens_details" not in (res.usage or {}):
        print("提示：usage 里没有 prompt_tokens_details，看不到 prefix cache 命中；"
              "vLLM 启动时加 --enable-prompt-tokens-details 才会返回 cached_tokens。")


def cmd_run(a) -> None:
    llm = _llm(a)
    log = EventLog.new(a.sessions_dir)
    print(f"会话 {log.session_id}  模型 {llm.model}  模式 {a.mode}  工作目录 {Path(a.workspace).resolve()}",
          file=sys.stderr)
    agent = _agent(a, llm, log, a.workspace)
    try:
        res = agent.run(a.task)
    finally:
        if agent.sandbox is not None:
            agent.sandbox.close()   # 自己创建的容器用完就删
    _report(res, log)


def _mcp_tools(a):
    """--mcp-config 指向 {"mcpServers": ...} 格式的文件（如 ~/.claude.json）；--mcp-only 选要接的 server。"""
    if not getattr(a, "mcp_config", None):
        return []
    from .mcp import load_mcp_tools
    tools, clients, warns = load_mcp_tools(a.mcp_config, a.mcp_only.split(",") if a.mcp_only else None)
    for w in warns:
        print(f"[mcp] {w}", file=sys.stderr)
    print(f"[mcp] 已接入 {len(clients)} 个 server，{len(tools)} 个工具", file=sys.stderr)
    return tools


def cmd_assistant(a) -> None:
    from .assistant import build_assistant, repl
    from .openrouter import OpenRouterLLM
    llm = OpenRouterLLM(a.model, native_tools=a.native_tools, temperature=a.temperature,
                        max_tokens=a.max_tokens, max_retries=a.retries)
    ws = Path(a.workspace).expanduser().resolve()
    log = EventLog.new(a.sessions_dir)
    agent = build_assistant(llm, ws, log, permissions=PermissionEngine(a.mode, allow=a.allow, deny=a.deny, ask=a.ask),
                            approver=APPROVERS[a.approve],
                            context=ContextManager(llm, max_context_tokens=a.max_context, enabled=not a.no_compact),
                            on_event=_printer(a.verbose), max_turns=a.max_turns, bash_timeout=a.bash_timeout,
                            bash=not a.no_bash)
    if a.once:
        res = agent.send(a.once)
        _report(res, log)
    else:
        repl(agent)
        print(f"会话日志    {log.path}")


def cmd_web(a) -> None:
    from .web import serve

    def make_llm(cfg: dict):
        kw = dict(temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], max_retries=a.retries,
                  native_tools=cfg["native_tools"], extra_body=cfg["extra_body"] or None)
        if a.base_url:   # 任意 OpenAI 兼容端点（vLLM 等）；不给就走 OpenRouter
            return OpenAICompatLLM(a.base_url, model=cfg["model"] or None, api_key=api_key(), **kw)
        from .openrouter import OpenRouterLLM
        return OpenRouterLLM(cfg["model"] or a.model, **kw)

    config = {"workspace": a.workspace, "mode": a.mode, "model": a.model or "", "max_tokens": a.max_tokens,
              "temperature": a.temperature, "native_tools": not a.text_tools,
              "extra_body": a.extra_body or {}, "sandbox": a.sandbox or "", "network": a.network,
              "max_turns": a.max_turns, "max_context": a.max_context, "mcp_config": a.mcp_config or ""}
    serve(make_llm, a.sessions_dir, config, port=a.port, open_browser=not a.no_open, resume=a.resume,
          runs_dir=Path(a.runs_dir).expanduser().resolve())


def cmd_resume(a) -> None:
    path = _session_path(a.session, a.sessions_dir)
    log = EventLog(path)
    start = next((e for e in log.events() if e.get("type") == "session_start"), None)
    if not start:
        sys.exit("日志里没有 session_start")
    if start.get("backend") == "openrouter":   # 文件助手的会话：用 OpenRouter 和完整工具集恢复
        from .assistant import build_assistant
        from .openrouter import OpenRouterLLM
        llm = OpenRouterLLM(start.get("model"), temperature=a.temperature, max_tokens=a.max_tokens)
        agent = build_assistant(llm, a.workspace or start["workspace"], log,
                                permissions=PermissionEngine(a.mode, allow=a.allow, deny=a.deny, ask=a.ask),
                                approver=APPROVERS[a.approve], context=ContextManager(llm, max_context_tokens=60000),
                                on_event=_printer(a.verbose), max_turns=a.max_turns, bash_timeout=a.bash_timeout)
    else:
        llm = _llm(a)
        agent = _agent(a, llm, log, a.workspace or start["workspace"])
    try:   # 无状态 harness：一切从日志重建（Managed Agents 的 wake(sessionId)）；sandbox 用 --sandbox 重新接上
        res = agent.resume()
    finally:
        if agent.sandbox is not None:
            agent.sandbox.close()
    _report(res, log)


def cmd_trace(a) -> None:
    events = load(_session_path(a.session, a.sessions_dir))
    s = summarize(events)
    print(f"任务          {s['task'][:200]}")
    print(format_summary(s))
    if a.timeline:
        print("\n" + timeline(events))


def cmd_sessions(a) -> None:
    files = [f for f in sorted(Path(a.sessions_dir).glob("*.jsonl")) if ".sub" not in f.stem][-a.n:]
    for f in files:
        s = summarize(load(f))
        print(f"{f.stem}  {s['status']:<10} {s['llm_calls']:>3} 轮  {s['wall_seconds']:>7}s  {s['task'][:60]}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="agentrt", description="最小 agent 运行时（vLLM / OpenAI 兼容接口）")
    p.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_opts(sp):
        sp.add_argument("--base-url", default=os.environ.get("AGENTRT_BASE_URL", "http://127.0.0.1:8000/v1"))
        sp.add_argument("--model", default=os.environ.get("AGENTRT_MODEL"), help="默认取 /v1/models 的第一个")
        sp.add_argument("--native-tools", action="store_true", help="走 OpenAI tools 字段（需要 vLLM tool parser）")
        sp.add_argument("--merge-system", action="store_true", help="把 system 并进第一条 user（模板不支持 system 时用）")
        sp.add_argument("--temperature", type=float, default=0.2)
        sp.add_argument("--max-tokens", type=int, default=2048)
        sp.add_argument("--retries", type=int, default=4, help="网络错误重试次数")
        sp.add_argument("--reasoning-replay", choices=["none", "turn", "all"], default="turn",
                        help="思考模型的 reasoning_content 回传策略：turn = 当前轮的工具调用链里带回（DeepSeek 规则）")
        sp.add_argument("--extra-body", help='合并进请求体的 JSON，如 \'{"thinking": {"type": "enabled"}}\'')

    def mcp_opts(sp):
        sp.add_argument("--mcp-config", help="MCP 配置文件（含 mcpServers，如 ~/.claude.json），默认不接 MCP")
        sp.add_argument("--mcp-only", help="只接这几个 server，逗号分隔")

    def agent_opts(sp):
        sp.add_argument("--mode", choices=MODES, default="default")
        sp.add_argument("--approve", choices=list(APPROVERS), default="cli",
                        help="需要确认时怎么办：cli 交互 / deny 一律拒绝 / policy 只放行模式层面的确认 / all 全放行")
        sp.add_argument("--allow", action="append", default=[], help="如 'bash(python *)'，可重复")
        sp.add_argument("--ask", action="append", default=[], help="命中就一定要确认，如 'bash(npm publish*)'，可重复")
        sp.add_argument("--deny", action="append", default=[], help="如 'bash(git push*)'，可重复")
        sp.add_argument("--max-turns", type=int, default=30)
        sp.add_argument("--max-context", type=int, default=24000,
                        help="上下文预算，估算 token 超过其 75%% 时自动压缩；会自动压到服务端 max_model_len 以内")
        sp.add_argument("--no-compact", action="store_true")
        sp.add_argument("--bash-timeout", type=float, default=30.0)
        sp.add_argument("--sandbox", help="local / bwrap / docker:<镜像> / container:<id>：换成 coding 工具集，工具在 sandbox 里执行")
        sp.add_argument("--network", action="store_true", help="sandbox 允许联网（bwrap / docker 默认断网）")
        sp.add_argument("-v", "--verbose", action="store_true")

    sp = sub.add_parser("ping", help="检查服务端连通性和 usage 字段")
    llm_opts(sp)
    sp.set_defaults(fn=cmd_ping)

    sp = sub.add_parser("run", help="执行一个任务")
    sp.add_argument("task")
    sp.add_argument("--workspace", default=".")
    llm_opts(sp)
    mcp_opts(sp)
    agent_opts(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("assistant", help="电脑文件助手（OpenRouter，交互式）")
    sp.add_argument("--workspace", default=".", help="助手能管理的目录，如 ~/Downloads")
    sp.add_argument("--once", metavar="请求", help="只执行一条请求就退出（非交互）")
    sp.add_argument("--model", default=None, help="默认取 AGENTRT_MODEL / OPENROUTER_MODEL")
    sp.add_argument("--native-tools", action="store_true")
    sp.add_argument("--temperature", type=float, default=0.2)
    sp.add_argument("--max-tokens", type=int, default=2048)
    sp.add_argument("--retries", type=int, default=3)
    sp.add_argument("--no-bash", action="store_true", help="不给助手 bash（默认有沙箱时提供）")
    agent_opts(sp)
    sp.set_defaults(fn=cmd_assistant)

    sp = sub.add_parser("web", help="网页控制台：和通用 agent 对话、看实时轨迹和历史会话、浏览跑分")
    sp.add_argument("--workspace", default=".", help="初始工作目录，网页里可以改")
    sp.add_argument("--mode", choices=MODES, default="default")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    sp.add_argument("--resume", metavar="会话", help="打开时接着这个会话（会话 id）")
    sp.add_argument("--base-url", default=None, help="OpenAI 兼容端点；不给就走 OpenRouter")
    sp.add_argument("--model", default=None, help="默认取 AGENTRT_MODEL / OPENROUTER_MODEL")
    sp.add_argument("--temperature", type=float, default=0.6)
    sp.add_argument("--max-tokens", type=int, default=16000)
    sp.add_argument("--retries", type=int, default=4)
    sp.add_argument("--text-tools", action="store_true", help="用文本协议调工具（默认走 native function calling）")
    sp.add_argument("--extra-body", type=json.loads, default=None, help='合并进请求体的 JSON，如 \'{"provider": ...}\'')
    sp.add_argument("--sandbox", default=None, help="local / bwrap / docker:<镜像> / container:<id>；不给就用本地工具")
    sp.add_argument("--network", action="store_true", help="sandbox 允许联网")
    sp.add_argument("--max-turns", type=int, default=100)
    sp.add_argument("--max-context", type=int, default=100000)
    sp.add_argument("--mcp-config", default=None, help="MCP 配置文件（含 mcpServers）")
    sp.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"),
                    help="“跑分”页读的目录（run_tb2.sh 的输出），默认仓库里的 runs/")
    sp.set_defaults(fn=cmd_web)

    sp = sub.add_parser("resume", help="从会话日志断点继续")
    sp.add_argument("session")
    sp.add_argument("--workspace", default=None, help="默认用日志里记录的工作目录")
    llm_opts(sp)
    agent_opts(sp)
    sp.set_defaults(fn=cmd_resume)

    sp = sub.add_parser("trace", help="查看会话摘要 / 时间线")
    sp.add_argument("session")
    sp.add_argument("--timeline", action="store_true")
    sp.set_defaults(fn=cmd_trace)

    sp = sub.add_parser("sessions", help="列出最近的会话")
    sp.add_argument("-n", type=int, default=20)
    sp.set_defaults(fn=cmd_sessions)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except RetryableError as e:
        sys.exit(f"连不上模型服务 {getattr(a, 'base_url', '')}：{e}\n"
                 f"检查 --base-url 指向的服务是否在运行、网络是否可达。")
    except LLMError as e:
        if "HTTP 401" in str(e):
            sys.exit("401：API key 不对。检查 AGENTRT_API_KEY 或 ~/.agentrt/api_key。")
        raise


if __name__ == "__main__":
    main()

"""子 agent：独立上下文 + 只读工具白名单，只把摘要带回主循环。

对照 Claude Code 的 AgentTool（src/tools/AgentTool/）：子 agent 有自己的消息历史，主上下文里只多一条 tool_result。
好处是“把整个目录摸一遍”这类要读很多东西的活不会撑大主上下文；多个 delegate 是只读调用，会被并行执行。
和源码的差异：只做同步的只读子 agent，不能递归派生、不能后台运行、没有自定义 agent 类型。
"""
from __future__ import annotations

from pathlib import Path

from .context import ContextManager
from .loop import Agent
from .permissions import PermissionEngine, deny_all_approver
from .session import EventLog
from .tools import Tool, _obj

SUB_PROMPT = """## 你的角色：子 agent
你是主 agent 派出的只读调查员，只有搜索和查看类工具，不能修改任何文件。
完成任务后，最终答复就是交给主 agent 的报告：先给结论，再列关键依据（相对路径、数量、大小），控制在 300 字以内；
不要贴大段原始输出。查不到就直说查不到，不要猜。"""

MAX_REPORT_CHARS = 3000


class Delegate(Tool):
    name = "delegate"
    description = ("把一个只读调查任务交给子 agent（独立上下文，只返回摘要）。适合要翻很多文件才能回答的问题，"
                   "比如“看看 Documents 下每个子目录大概放了什么”。task 要写清楚要查什么、要什么形式的结论。")
    parameters = _obj({"task": {"type": "string"}}, ["task"])
    read_only = True   # 子 agent 自己只有只读工具，所以对主 agent 来说是只读的（也因此多个 delegate 可并行）
    risk = "low"

    def __init__(self, parent: Agent):
        self.parent = parent
        self._n = 0

    def subject(self, args):
        return str(args.get("task", ""))[:120]

    def run(self, args, ctx):
        p = self.parent
        with p._lock:
            self._n += 1
            n = self._n
        allowed = [t for t in p.tools.values() if t.read_only and t.name != self.name]
        sub_log = EventLog(Path(p.log.path).with_name(f"{p.log.path.stem}.sub{n}.jsonl"))
        child = Agent(p.llm, ctx.workspace, sub_log, tools=allowed, permissions=PermissionEngine("plan"),
                      approver=deny_all_approver, context=ContextManager(p.llm, max_context_tokens=p.cm.max_context_tokens),
                      max_turns=12, extra_prompt=SUB_PROMPT)
        child._cancel = p._cancel   # 主 agent 被中断时子 agent 一起停
        p._emit({"type": "subagent_start", "n": n, "task": args["task"], "log": sub_log.path.name,
                 "tools": [t.name for t in allowed]})
        res = child.run(args["task"])
        with p._lock:   # 子 agent 的开销记到主 agent 名下
            r = p.result
            r.llm_calls += res.llm_calls
            r.prompt_tokens += res.prompt_tokens
            r.completion_tokens += res.completion_tokens
            r.cached_tokens += res.cached_tokens
            r.llm_seconds += res.llm_seconds
        p._emit({"type": "subagent_end", "n": n, "status": res.status, "turns": res.turns,
                 "tool_calls": res.tool_calls, "tokens": res.prompt_tokens + res.completion_tokens})
        report = (res.final or "").strip()
        if res.status != "end_turn" or not report:
            return f"[子 agent 未完成：{res.status} {res.error}]\n{report}"[:MAX_REPORT_CHARS]
        return report[:MAX_REPORT_CHARS]

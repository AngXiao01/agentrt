"""电脑文件助手：交互式 REPL，形态参照 Claude Code —— 你说需求，它自己搜索、查看、整理，
需要改动文件时先问你。底层就是 Agent 主循环 + fs_tools。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .loop import Agent
from .fs_tools import CopyFile, FileStats, FindDuplicates, FindFiles, MoveFile, Trash
from .tools import Grep, ListDir, ReadFile

ASSISTANT_PROMPT = """## 你的角色：电脑文件助手
工作目录就是用户让你管理的范围。你负责帮用户查找、了解、整理其中的文件。
- 找东西：find_files（名称/大小/日期）、grep（内容）、file_stats（按类型/月份/年份/目录统计数量与大小，group_by=folder 看哪个目录占空间）、find_duplicates。
- 用户说的日期换算成 YYYY-MM-DD 再传。
- 整理文件（move_file / copy_file / make_dir / trash）之前，先用只读工具把现状查清楚，并在一句话里说明你打算做什么。
- 没有永久删除：用户要求删除时，先用 find_files 列出将被删除的清单并向用户说明数量，再用 trash 的 paths 一次提交（可恢复）。重复文件只报告，不要自作主张删除，除非用户明确要求。
- 批量操作先小范围试一个，确认没问题再继续；结束时列出实际改动了哪些文件。
- 回答里引用文件时写相对路径，大小用 KB/MB，方便用户核对。"""

HELP = """/help 帮助   /clear 清空对话   /mode <plan|default|auto|bypass> 切换权限模式   /exit 退出
plan=只读  default=改动前确认  auto=只有删除（trash）确认  bypass=全放行
示例：找出最近一周改过的 PDF / 哪些东西最占空间 / 把 Downloads 里的图片按月份归档 / 找重复文件"""


def assistant_tools() -> list:
    """文件助手只给 9 个工具：6 个只读（找、统计、看、查重）+ 3 个会改动文件（移动、复制、删除到废纸篓）。
    没有 bash / write_file / edit_file：文件管理用不着，工具少，模型选错的机会也少，权限面也小
    （Claude Code 的核心工具也是少量职责单一的工具：Glob / Grep / Read / Edit / Write / Bash）。"""
    return [ListDir(), FindFiles(), FileStats(), Grep(), ReadFile(), FindDuplicates(), MoveFile(), CopyFile(), Trash()]


def build_assistant(llm, workspace, log, *, permissions, approver, context, on_event,
                    max_turns=40, bash_timeout=30.0) -> Agent:
    tools = assistant_tools()
    return Agent(llm, workspace, log, tools=tools, permissions=permissions, approver=approver,
                 context=context, max_turns=max_turns, bash_timeout=bash_timeout,
                 on_event=on_event,
                 extra_prompt=ASSISTANT_PROMPT + f"\n- 今天是 {time.strftime('%Y-%m-%d')}，用户说的“最近一周”“上个月”等相对时间据此换算。")


class Spinner:
    """终端里的“思考中”提示：转圈 + 已用时间 + Ctrl+C 中断提示。只在交互终端里显示。"""
    FR = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, agent: Agent, out=sys.stderr):
        self.agent, self.out, self._stop = agent, out, threading.Event()
        self.on = hasattr(out, "isatty") and out.isatty()

    def __enter__(self):
        if self.on:
            self.t0 = time.time()
            self.th = threading.Thread(target=self._run, daemon=True)
            self.th.start()
        return self

    def _run(self):
        i = 0
        while not self._stop.wait(0.1):
            r = self.agent.result
            self.out.write(f"\r\033[K{self.FR[i % 10]} 思考中… {time.time() - self.t0:.0f}s · {r.tool_calls} 步 · "
                           f"{r.prompt_tokens + r.completion_tokens} tokens  (Ctrl+C 中断)")
            self.out.flush()
            i += 1

    def __exit__(self, *a):
        if self.on:
            self._stop.set()
            self.th.join()
            self.out.write("\r\033[K")
            self.out.flush()


def repl(agent: Agent, read: Callable[[str], str] = input, out=sys.stdout) -> None:
    print(f"文件助手就绪  模型 {getattr(agent.llm, 'model', '?')}  范围 {agent.ctx.workspace}  "
          f"模式 {agent.perms.mode}\n输入 /help 查看用法。", file=out)
    while True:
        try:
            line = read("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            return
        if line == "/help":
            print(HELP, file=out)
        elif line == "/clear":
            agent.reset()
            print("已清空对话。", file=out)
        elif line.startswith("/mode"):
            from .permissions import MODES
            m = line.split()[1:] and line.split()[1]
            if m in MODES:
                agent.perms.mode = m
                print(f"权限模式：{m}", file=out)
            else:
                print(f"用法：/mode <{'|'.join(MODES)}>   当前 {agent.perms.mode}", file=out)
        else:
            try:
                with Spinner(agent):
                    res = agent.send(line)
            except KeyboardInterrupt:
                agent.repair_interrupted()   # 补齐没有结果的工具调用，下一条消息才能接着聊
                print("\n[已中断]", file=out)
                continue
            print("\n" + (res.final or f"（无最终答复，状态 {res.status} {res.error}）"), file=out)
            print(f"[{res.llm_calls} 次模型调用 · {res.tool_calls} 次工具 · {res.wall_seconds:.1f}s · "
                  f"{res.prompt_tokens}+{res.completion_tokens} tokens]", file=out)

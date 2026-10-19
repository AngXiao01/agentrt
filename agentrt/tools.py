"""工具定义。每个工具声明：是否只读、风险等级、参数 schema，并自带越界检查。

硬约束（路径不能逃出 workspace）在 validate() 和 run() 两处都检查：
权限规则可以放行一个调用，但放不开这条不变量。
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ToolContext:
    workspace: Path
    bash_timeout: float = 30.0
    max_output_chars: int = 20000


class ToolError(Exception):
    pass


def resolve_in_workspace(ctx: ToolContext, p: str) -> Path:
    ws = ctx.workspace.resolve()
    target = (ws / p).resolve()
    if target != ws and ws not in target.parents:
        raise ToolError(f"路径越界：{p} 不在工作目录内")
    return target


class Tool:
    name = ""
    description = ""
    parameters: dict = {}
    read_only = True
    risk = "low"  # low / medium / high

    def subject(self, args: dict) -> str:
        """权限规则匹配用的主体字符串，例如 bash 的命令、文件工具的路径。"""
        return str(args.get("path", ""))

    def validate(self, args: dict, ctx: ToolContext) -> Optional[str]:
        for key in self.parameters.get("required", []):
            if key not in args:
                return f"缺少必填参数 {key}"
        if "path" in args:
            try:
                resolve_in_workspace(ctx, str(args["path"]))
            except ToolError as e:
                return str(e)
        return None

    def run(self, args: dict, ctx: ToolContext) -> str:
        raise NotImplementedError

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


class ListDir(Tool):
    name = "list_dir"
    description = "列出目录内容（相对 workspace 的路径）。"
    parameters = _obj({"path": {"type": "string", "description": "目录，默认 ."}}, [])

    def run(self, args, ctx):
        d = resolve_in_workspace(ctx, args.get("path", "."))
        if not d.is_dir():
            raise ToolError(f"{args.get('path', '.')} 不是目录")
        items = []
        for p in sorted(d.iterdir()):
            items.append(p.name + ("/" if p.is_dir() else f"  ({p.stat().st_size} B)"))
        return "\n".join(items) or "（空目录）"


class ReadFile(Tool):
    name = "read_file"
    description = "读取文本文件，按行分页：offset 为起始行（从 0 开始），limit 为行数。"
    parameters = _obj({"path": {"type": "string"},
                       "offset": {"type": "integer", "description": "起始行，默认 0"},
                       "limit": {"type": "integer", "description": "行数，默认 200"}}, ["path"])

    def run(self, args, ctx):
        p = resolve_in_workspace(ctx, args["path"])
        if not p.is_file():
            raise ToolError(f"文件不存在：{args['path']}")
        lines = p.read_text(errors="replace").splitlines()
        off, lim = int(args.get("offset", 0)), int(args.get("limit", 200))
        chunk = lines[off:off + lim]
        body = "\n".join(f"{off + i + 1:>6}\t{l}" for i, l in enumerate(chunk))
        tail = f"\n[共 {len(lines)} 行；本次显示 {off + 1}-{off + len(chunk)}]"
        return body + tail


class WriteFile(Tool):
    name = "write_file"
    description = "写入（覆盖）文本文件，自动创建父目录。"
    parameters = _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
    read_only = False
    risk = "medium"

    def run(self, args, ctx):
        p = resolve_in_workspace(ctx, args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp_agentrt")
        tmp.write_text(str(args["content"]))
        os.replace(tmp, p)  # 原子替换：中途崩溃不会留下写了一半的文件
        return f"已写入 {args['path']}（{len(str(args['content']))} 字符）"


class EditFile(Tool):
    name = "edit_file"
    description = "把文件中唯一出现的 old 字符串替换为 new。old 出现 0 次或多次都会报错。"
    parameters = _obj({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                      ["path", "old", "new"])
    read_only = False
    risk = "medium"

    def run(self, args, ctx):
        p = resolve_in_workspace(ctx, args["path"])
        text = p.read_text()
        n = text.count(args["old"])
        if n != 1:
            raise ToolError(f"old 在文件中出现 {n} 次，必须恰好 1 次；请提供更长的上下文")
        tmp = p.with_name(p.name + ".tmp_agentrt")
        tmp.write_text(text.replace(args["old"], args["new"], 1))
        os.replace(tmp, p)
        return f"已修改 {args['path']}"


class Grep(Tool):
    name = "grep"
    description = "在工作目录下按正则搜索文本文件，返回 文件:行号:内容。"
    parameters = _obj({"pattern": {"type": "string"}, "path": {"type": "string", "description": "默认 ."}},
                      ["pattern"])

    def run(self, args, ctx):
        root = resolve_in_workspace(ctx, args.get("path", "."))
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            raise ToolError(f"正则无效：{e}")
        hits = []
        files = [root] if root.is_file() else [p for p in root.rglob("*")
                                                if p.is_file() and ".agentrt" not in p.parts]
        for f in sorted(files):
            try:
                if f.stat().st_size > 2_000_000:  # 跳过大文件（日志、数据库等）
                    continue
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(ctx.workspace.resolve())}:{i}:{line[:200]}")
            except OSError:
                continue
            if len(hits) >= 500:
                hits.append("[结果过多，已截断到 500 条]")
                break
        return "\n".join(hits) or "（无匹配）"


class Bash(Tool):
    name = "bash"
    description = "在工作目录下执行 shell 命令，有超时；环境变量已精简，不继承宿主机的密钥。"
    parameters = _obj({"command": {"type": "string"},
                       "timeout": {"type": "number", "description": "秒，默认 30"}}, ["command"])
    read_only = False
    risk = "high"

    def subject(self, args):
        return str(args.get("command", ""))

    def run(self, args, ctx):
        timeout = min(float(args.get("timeout", ctx.bash_timeout)), 300)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(ctx.workspace),
               "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(args["command"], shell=True, cwd=ctx.workspace, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # 杀整个进程组，避免残留子进程
            out, _ = proc.communicate()
            return f"[超时 {timeout}s，已终止]\n{out or ''}"
        return f"[exit {proc.returncode}]\n{out or ''}"


class TodoWrite(Tool):
    """把任务计划写进 todo.md。借鉴 Manus 的「复述」：每步都把目标重新放到上下文末尾。"""
    name = "todo_write"
    description = "更新任务清单（写入 .agentrt/todo.md）。items 为字符串列表，完成项以 [x] 开头。"
    parameters = _obj({"items": {"type": "array", "items": {"type": "string"}}}, ["items"])
    read_only = False
    risk = "low"

    def subject(self, args):
        return ".agentrt/todo.md"

    def run(self, args, ctx):
        items = [str(i) for i in args["items"]]
        body = "\n".join(i if i.startswith("[") else f"[ ] {i}" for i in items)
        p = ctx.workspace / ".agentrt" / "todo.md"  # 放在隐藏目录，不污染工作区，grep 也不会搜到
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body + "\n")
        return "当前清单：\n" + body


def default_tools() -> list[Tool]:
    return [ListDir(), ReadFile(), Grep(), WriteFile(), EditFile(), Bash(), TodoWrite()]


def describe_tools_for_prompt(tools: list[Tool]) -> str:
    lines = []
    for t in tools:
        props = t.parameters.get("properties", {})
        req = set(t.parameters.get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {t.name}({args})：{t.description}")
    return "\n".join(lines)

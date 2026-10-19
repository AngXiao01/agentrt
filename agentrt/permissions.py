"""权限判定链，对照 Claude Code 源码（src/utils/permissions/permissions.ts 的 hasPermissionsToUseToolInner、
src/services/tools/toolHooks.ts 的 resolveHookPermissionDecision）：

  0. deny 规则先查（源码里它在 hook 之后，这里提前，被禁止的调用不会弹确认）
  0. PreToolUse hooks   hook 说 deny 直接拒绝；hook 说 ask 强制弹确认（任何模式下都弹，对应 forceDecision）；
                        hook 说 allow 只是省掉确认，deny / ask 规则照样生效
  1. deny 规则          命中即拒绝                                     （1a）
  2. ask 规则           命中即要求确认                                 （1b / 1f）
  3. 工具自检           路径越界、缺参数等硬约束，bypass 也绕不过      （1c / 1d / 1g：safetyCheck 免疫 bypass）
  4. bypass 模式        全放行                                         （2a）
  5. allow 规则         用户显式放行                                   （2b）
  6. 其余按模式         plan 只读 / default 写操作确认 / auto 只有高风险确认 （3：passthrough → ask）

规则写法同 Claude Code：`bash(git status*)`、`write_file(*.md)`、`read_file`（整个工具）。
和源码的差异：plan / auto 在源码里是工具自己的 checkPermissions 处理的（acceptEdits 等），这里集中在第 6 步。
工具在 system prompt 里始终完整出现，不因模式增删（保持前缀稳定，利于 KV cache），不可用的调用在这里被拦下。
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .tools import Tool, ToolContext


@dataclass
class Decision:
    behavior: str  # allow / ask / deny
    reason: str
    source: str


Hook = Callable[[Tool, dict, ToolContext], Optional[Decision]]

MODES = ("plan", "default", "auto", "bypass")


def _parse_rule(rule: str) -> tuple[str, str]:
    m = re.fullmatch(r"\s*([\w\-]+)\s*(?:\((.*)\))?\s*", rule)
    if not m:
        raise ValueError(f"规则格式错误：{rule}")
    return m.group(1), (m.group(2) if m.group(2) is not None else "*")


def _match(rules: list[str], tool: Tool, args: dict) -> Optional[str]:
    subject = tool.subject(args)
    for r in rules:
        name, pattern = _parse_rule(r)
        if name == tool.name and fnmatch.fnmatchcase(subject, pattern):
            return r
    return None


# ---------------------------------------------------------------- 内置 hooks

_DANGEROUS = [
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b", "递归强制删除"),
    (r"\brsync\b.*--delete", "rsync --delete 会删除目标端文件"),
    (r"\b(mkfs|dd\s+if=|shred)\b", "磁盘级破坏性命令"),
    (r"(curl|wget)[^|]*\|\s*(ba|z)?sh", "下载后直接执行"),
    (r"\bgit\s+push\b.*(--force|-f)\b", "强制推送"),
    (r"\bchmod\s+-R\s+777\b", "递归放开权限"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork 炸弹"),
    (r"\b(sudo|su)\b", "提权"),
]

_SECRET = [
    (r"sshpass\s+-p\s*\S+", "命令里带明文 SSH 密码"),
    (r"-p'[^']{4,}'|-p\"[^\"]{4,}\"", "命令行里带明文密码"),
    (r"\b(sk-[A-Za-z0-9]{16,}|tvly-[A-Za-z0-9\-]{16,}|AKIA[0-9A-Z]{16})\b", "疑似 API Key"),
    (r"(?i)\b(password|passwd|token|secret)\s*=\s*\S{6,}", "疑似凭据赋值"),
]


def dangerous_command_hook(tool: Tool, args: dict, ctx: ToolContext) -> Optional[Decision]:
    if tool.name != "bash":
        return None
    cmd = str(args.get("command", ""))
    for pat, why in _DANGEROUS:
        if re.search(pat, cmd):
            return Decision("ask", f"高风险命令：{why}", "hook:dangerous_command")
    return None


# `..` 前面也可能是 `/`（./../x、a/../../x），`~` 只在词首才有意义
_PARENT_REF = re.compile(r"(^|[\s'\"=;&|(<>/])\.\.(/|\s|$|['\";&|)])|(^|[\s'\"=;&|(<>])~(/|\s|$|['\";&|)])")
_ABS_PATH = re.compile(r"(?:^|[\s'\"=;&|(<>])(/[^\s'\";|&<>)]*)")
_SAFE_ABS = ("/usr/", "/bin/", "/sbin/", "/dev/null", "/dev/stdout", "/dev/stderr", "/etc/os-release")


def workspace_hook(tool: Tool, args: dict, ctx: ToolContext) -> Optional[Decision]:
    """bash 不是沙箱：命令里出现 ../、~ 或工作目录外的绝对路径时要求确认。
    这是启发式，能被 $(dirname $PWD) 之类绕过；真正的隔离要靠容器 / bwrap。"""
    if tool.name != "bash":
        return None
    cmd = str(args.get("command", ""))
    if _PARENT_REF.search(cmd):
        return Decision("ask", "命令引用了工作目录之外的路径（.. 或 ~）", "hook:workspace")
    ws = str(ctx.workspace.resolve())
    for m in _ABS_PATH.finditer(cmd):
        path = m.group(1)
        if path.startswith(ws) or path.startswith(_SAFE_ABS):
            continue
        return Decision("ask", f"命令引用了工作目录之外的绝对路径 {path}", "hook:workspace")
    return None


def secret_hook(tool: Tool, args: dict, ctx: ToolContext) -> Optional[Decision]:
    blob = " ".join(str(v) for v in args.values())
    for pat, why in _SECRET:
        if re.search(pat, blob):
            return Decision("deny", f"{why}：凭据不能出现在工具参数里（也会被写进日志）", "hook:secret")
    return None


# ---------------------------------------------------------------- 判定引擎

class PermissionEngine:
    def __init__(self, mode: str = "default", allow: Optional[list[str]] = None,
                 deny: Optional[list[str]] = None, hooks: Optional[list[Hook]] = None,
                 ask: Optional[list[str]] = None):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        self.mode = mode
        self.allow = allow or []
        self.deny = deny or []
        self.ask = ask or []
        self.hooks = hooks if hooks is not None else [secret_hook, dangerous_command_hook, workspace_hook]
        for r in self.allow + self.deny + self.ask:
            _parse_rule(r)  # 启动时就校验规则格式

    def decide(self, tool: Tool, args: dict, ctx: ToolContext) -> Decision:
        rule = _match(self.deny, tool, args)   # 比源码更严：deny 规则先于 hook 的 ask，被禁止的调用不会弹确认
        if rule:
            return Decision("deny", f"命中 deny 规则 {rule}", "deny_rule")
        hook_allow = None
        for hook in self.hooks:
            d = hook(tool, args, ctx)
            if d is None:
                continue
            if d.behavior in ("deny", "ask"):   # hook 的 deny / ask 在任何模式下都生效（含 bypass）
                return d
            hook_allow = hook_allow or d

        rule = _match(self.ask, tool, args)
        if rule:
            return Decision("ask", f"命中 ask 规则 {rule}", "ask_rule")
        if hook_allow:
            return hook_allow

        err = tool.validate(args, ctx)
        if err:
            return Decision("deny", err, "tool_check")

        rule = _match(self.allow, tool, args) if self.mode != "bypass" else None
        if rule:
            return Decision("allow", f"命中 allow 规则 {rule}", "allow_rule")

        if self.mode == "bypass":
            return Decision("allow", "bypass 模式", "mode")
        if self.mode == "plan":
            if tool.read_only:
                return Decision("allow", "plan 模式只读工具", "mode")
            return Decision("deny", "plan 模式下禁止写操作和命令执行", "mode")
        if self.mode == "auto":
            if tool.risk == "high":
                return Decision("ask", "auto 模式下高风险工具需要确认", "mode")
            return Decision("allow", "auto 模式", "mode")
        # default
        if tool.read_only:
            return Decision("allow", "只读工具", "mode")
        return Decision("ask", "写操作需要确认", "mode")


# ---------------------------------------------------------------- 审批器

def deny_all_approver(tool: Tool, args: dict, decision: Decision) -> bool:
    """无人值守时 fail-closed：需要确认的一律拒绝。"""
    return False


def policy_approver(tool: Tool, args: dict, decision: Decision) -> bool:
    """无人值守策略（评测用）：权限模式层面的例行确认自动放行；hook 标记的危险操作一律拒绝。"""
    return decision.source == "mode"


def allow_all_approver(tool: Tool, args: dict, decision: Decision) -> bool:
    return True


def cli_approver(tool: Tool, args: dict, decision: Decision) -> bool:
    import json
    print(f"\n[需要确认] {tool.name} {json.dumps(args, ensure_ascii=False)[:300]}")
    print(f"  原因：{decision.reason}")
    try:
        return input("  允许执行？[y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False

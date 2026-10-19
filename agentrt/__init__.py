"""agentrt：一个最小但完整的 agent 运行时。

主循环 / 权限链 / 三级上下文压缩 / 事件溯源的会话日志与断点恢复 / tracing / 评测。
只依赖 Python 标准库；模型走 OpenAI 兼容接口（vLLM）。
"""
from .context import ContextManager
from .llm import OpenAICompatLLM, ScriptedLLM
from .loop import Agent, RunResult
from .permissions import PermissionEngine
from .session import EventLog, ReplayState

__all__ = ["Agent", "RunResult", "ContextManager", "OpenAICompatLLM", "ScriptedLLM", "PermissionEngine",
           "EventLog", "ReplayState"]
__version__ = "0.1.0"

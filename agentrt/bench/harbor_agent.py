"""Harbor 外部 agent：在 Terminal-Bench 2.0（以及 Harbor registry 里的 SWE-bench Verified 等数据集）上跑 agentrt。

Harbor 是 Terminal-Bench 2.0 的官方运行框架。外部 agent（BaseAgent）跑在宿主机上，通过 environment.exec 操作任务容器——
正好是 Managed Agents 的 brain / hands 分离：agentrt 的主循环是“脑”，任务容器是“手”（HarborSandbox），
会话日志写在 Harbor 的 trial 目录里。

    pip install harbor
    export DEEPSEEK_API_KEY=...
    harbor run -d terminal-bench@2.0 -a agentrt.bench.harbor_agent:AgentRT -m deepseek/deepseek-chat -n 8 \\
        --ak max_turns=100 --ak reasoning_replay=turn

    # 任意 OpenAI 兼容端点（vLLM / OpenRouter / 自建网关）
    harbor run -d terminal-bench@2.0 -a agentrt.bench.harbor_agent:AgentRT -m openai/<模型> \\
        --ak api_base=http://127.0.0.1:8000/v1 --ae AGENTRT_API_KEY=...

--ak 参数（都可选）：api_base、model_id、max_turns、max_tokens、max_context、temperature、native_tools、
reasoning_replay（none / turn / all）、extra_body（JSON）、price（JSON：每百万 tokens 美元 {"input","cached","output"}）、
bash_timeout、request_timeout。

只在装了 harbor 的环境里 import 这个模块；HarborSandbox 本身不依赖 harbor，便于测试。
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from ..sandboxes import _WRAP, ExecResult, SandboxError, _ExecFs

PROVIDERS = {   # -m 的前缀 → 默认端点。模型 id 取前缀之后的部分（openrouter 保留厂商前缀）
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
KEY_ENVS = ("AGENTRT_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY",
            "DASHSCOPE_API_KEY")


def _key_from_file() -> str:
    """环境变量都没设时回退到 ~/.agentrt/api_key，和 CLI 一致。"""
    try:
        return (Path.home() / ".agentrt" / "api_key").read_text().strip() or "EMPTY"
    except OSError:
        return "EMPTY"


class HarborSandbox(_ExecFs):
    """把 Harbor 的 async environment 包成同步的 Sandbox。主循环跑在工作线程里，exec 投递回 Harbor 的事件循环。"""
    kind = "harbor"
    enforcement = "full"
    disposable = True

    def __init__(self, environment, loop: asyncio.AbstractEventLoop, workdir: str, user=None):
        self.env, self.loop, self.workdir, self.user = environment, loop, workdir, user

    def _call(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def _exec(self, command: str, timeout: float):
        kw: dict[str, Any] = {"cwd": self.workdir, "timeout_sec": int(timeout)}
        if self.user is not None:
            kw["user"] = self.user
        return self._call(self.env.exec(command, **kw), timeout + 30)

    def exec(self, command, timeout=30.0, mode="workspace-write", on_spawn=None):
        t0 = time.time()
        t = max(1, int(round(timeout)))
        wrapped = f"sh -c {shlex.quote(_WRAP)} agentrt {shlex.quote(command)} {t}"
        try:
            r = self._exec(wrapped, t + 15)   # 容器里 timeout -s KILL 先动手；Harbor 的超时只是兜底
        except Exception as e:  # noqa: BLE001 — Harbor 各后端超时抛的异常类型不一
            if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
                return ExecResult(-9, f"[Harbor 超时] {e}", True, time.time() - t0)
            raise SandboxError(f"environment.exec 失败：{type(e).__name__}: {e}") from e
        out = (r.stdout or "") + (("\n" if r.stdout else "") + r.stderr if r.stderr else "")
        dt = time.time() - t0
        return ExecResult(r.return_code, out, r.return_code in (124, 137, -9) and dt >= t - 0.5, dt)

    def _raw(self, argv_tail, data, timeout):
        raise NotImplementedError   # 读写各自实现：Harbor 的 exec 返回的是已经解码的文本

    def read_bytes(self, path: str) -> bytes:
        p = shlex.quote(self.resolve(path))
        r = self._exec(f"if [ -d {p} ]; then echo 'is a directory' >&2; exit 21; fi; "
                       f"if command -v base64 >/dev/null 2>&1; then echo B64 && base64 < {p}; "   # 从 stdin 读：BSD base64 不接受位置参数
                       f"else echo RAW && cat {p}; fi", 60)
        if r.return_code != 0:
            msg = (r.stderr or r.stdout or "").strip()
            raise (IsADirectoryError if r.return_code == 21 else FileNotFoundError)(msg or f"读取失败：{path}")
        head, _, body = (r.stdout or "").partition("\n")
        if head.strip() == "B64":   # 走 base64：二进制和非 UTF-8 文件也能原样拿回来
            return base64.b64decode(body.encode(), validate=False)
        return body.encode()

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.resolve(path)
        r = self._exec(f"mkdir -p {shlex.quote(str(Path(target).parent))}", 60)
        if r.return_code != 0:
            raise OSError((r.stderr or r.stdout or "").strip())
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
        try:
            self._call(self.env.upload_file(f.name, target), 120)
        finally:
            os.unlink(f.name)

    def describe(self):
        return f"a Linux container (task environment), working directory {self.workdir}; you have root and may install software"


def _resolve_model(model_name: str, api_base: Optional[str], model_id: Optional[str]) -> tuple[str, str]:
    prefix, _, rest = (model_name or "").partition("/")
    base = api_base or os.environ.get("AGENTRT_BASE_URL") or PROVIDERS.get(prefix)
    if not base:
        raise ValueError(f"不知道 {model_name} 用哪个端点：--ak api_base=... 或设 AGENTRT_BASE_URL")
    return base, model_id or (rest if rest else model_name)


try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ImportError:   # 没装 harbor：只有 HarborSandbox / _resolve_model 可用
    BaseAgent = None


if BaseAgent is not None:
    class AgentRT(BaseAgent):
        OPTION_KEYS = ("api_base", "model_id", "max_turns", "max_tokens", "max_retries", "max_context", "temperature",
                       "native_tools", "reasoning_replay", "extra_body", "price", "bash_timeout", "request_timeout")

        def __init__(self, logs_dir, model_name=None, logger=None, *args, **kwargs):
            self.opts = {k: kwargs.pop(k) for k in list(kwargs) if k in self.OPTION_KEYS}
            super().__init__(logs_dir, model_name, logger, *args, **kwargs)

        @staticmethod
        def name() -> str:
            return "agentrt"

        def version(self) -> str:
            from agentrt import __version__
            return __version__

        async def setup(self, environment: BaseEnvironment) -> None:
            return None   # 外部 agent：不往容器里装任何东西

        def _llm(self):
            from agentrt.llm import OpenAICompatLLM

            class _Direct(OpenAICompatLLM):
                def _resolve_model(self, wanted):   # 远端厂商的 /models 不一定带 max_model_len，也没必要先打一枪
                    return wanted

            o = self.opts
            base, mid = _resolve_model(self.model_name or "", o.get("api_base"), o.get("model_id"))
            key = next((v for v in (self._get_env(k) for k in KEY_ENVS) if v), None) or _key_from_file()
            extra = o.get("extra_body")
            return _Direct(base, model=mid, api_key=key, native_tools=bool(o.get("native_tools", True)),
                           temperature=float(o.get("temperature", 0.6)), max_tokens=int(o.get("max_tokens", 32768)),
                           timeout=float(o.get("request_timeout", 600)), max_retries=int(o.get("max_retries", 10)), max_backoff=60.0,
                           reasoning_replay=str(o.get("reasoning_replay", "turn")),
                           extra_body=json.loads(extra) if isinstance(extra, str) else extra)

        async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
            from agentrt.coding_tools import build_coding_agent
            from agentrt.context import ContextManager
            from agentrt.session import EventLog

            o = self.opts
            loop = asyncio.get_running_loop()
            pwd = await environment.exec("pwd", timeout_sec=30)
            workdir = (pwd.stdout or "/").strip().splitlines()[-1] if pwd.return_code == 0 else "/"
            sandbox = HarborSandbox(environment, loop, workdir)
            llm = self._llm()
            logs = Path(self.logs_dir)
            logs.mkdir(parents=True, exist_ok=True)
            log = EventLog(logs / "agentrt.jsonl")
            price = o.get("price")
            agent = build_coding_agent(
                llm, sandbox, log, max_turns=int(o.get("max_turns", 100)),
                context=ContextManager(llm, max_context_tokens=int(o.get("max_context", 100_000)),
                                       tool_result_max_chars=16_000),
                price=json.loads(price) if isinstance(price, str) else price,
                bash_timeout=float(o.get("bash_timeout", 180)), harness_dir=logs / "harness")
            try:
                await asyncio.to_thread(agent.run, instruction)
            except asyncio.CancelledError:   # Harbor 的 agent 超时：让主循环在下一个检查点停下，别在后台继续烧 token
                agent.cancel()
                raise
            finally:
                r = agent.result
                context.n_input_tokens = r.prompt_tokens
                context.n_cache_tokens = r.cached_tokens
                context.n_output_tokens = r.completion_tokens
                context.cost_usd = r.cost_usd if price else None
                context.metadata = {"status": r.status, "turns": r.turns, "tool_calls": r.tool_calls,
                                    "tool_errors": r.tool_errors, "parse_errors": r.parse_errors,
                                    "compactions": r.compactions, "cache_breaks": r.cache_breaks,
                                    "reasoning_tokens": r.reasoning_tokens, "llm_seconds": round(r.llm_seconds, 1),
                                    "tool_seconds": round(r.tool_seconds, 1), "session_log": str(log.path)}

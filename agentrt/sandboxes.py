"""执行环境（“手”）：harness 只通过三个动词碰外部世界——exec / read_bytes / write_bytes。

对照：
- Claude Managed Agents（“Decoupling the brain from the hands”）：sandbox 只暴露 execute(name, input) → string；
  容器按需 provision（不用 sandbox 的会话不付启动开销）；容器死了当 cattle：harness 把失败作为工具错误回填，
  下一次调用重新 provision。凭据不进 sandbox。
- DeepSeek Harness（dsh-sandbox / ctx.fs / ctx.subprocess）：文件系统和进程提供方共享同一个“执行世界”，
  所以把它指向远程 sandbox，bash 和读写文件一起搬过去；文件效果策略按调用携带
  （read-only / workspace-write / danger-full-access）；后端如实报告 enforcement（full / partial）。
- Claude Code：bash 在 sandbox 里执行时可以免确认（autoAllowBashIfSandboxed）。这里用 `disposable` +
  `enforcement` 两个事实让权限层做同样的判断（见 permissions.py）。

后端：
  LocalSandbox   直接在宿主机执行，不隔离（enforcement = none）。文件工具仍做工作目录越界检查。
  BwrapSandbox   Linux bubblewrap：只读挂载根目录，只有工作目录和私有 /tmp 可写，断网，隐藏主目录。
  DockerSandbox  docker exec 进容器：懒启动、容器丢失后重建、超时在容器内整组杀掉。
  （bench/harbor_agent.HarborSandbox：Harbor 的 environment.exec，跑 Terminal-Bench 用）
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

MODES = ("read-only", "workspace-write", "full")
OnSpawn = Optional[Callable[[int], None]]


@dataclass
class ExecResult:
    exit_code: int
    output: str              # stdout + stderr 合并（模型看到的就是终端里会看到的）
    timed_out: bool = False
    seconds: float = 0.0


class SandboxError(Exception):
    """sandbox 自己出了问题（不是命令失败）。"""


class SandboxLost(SandboxError):
    """容器没了。回填成工具错误；能重建的后端在下一次调用时重新 provision（工作区改动随之丢失）。"""


def _decode(b: Optional[bytes]) -> str:
    return (b or b"").decode("utf-8", errors="replace")


class Sandbox:
    kind = "base"
    enforcement = "none"    # none / partial / full：后端对“文件写入限定在工作区”这一承诺管住了多少
    disposable = False      # 一次性环境：爆炸半径就是它自己（容器），权限层可以整体放行
    network = True
    workdir: str = "."

    # ------------------------------------------------------------ 三个动词
    def exec(self, command: str, timeout: float = 30.0, mode: str = "workspace-write",
             on_spawn: OnSpawn = None) -> ExecResult:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------ 其他
    def resolve(self, path: str) -> str:
        """模型给的路径 → sandbox 里的绝对路径。相对路径相对 workdir。"""
        p = PurePosixPath(path)
        return str(p if p.is_absolute() else PurePosixPath(self.workdir) / p)

    def has(self, program: str) -> bool:
        """sandbox 里有没有某个程序（rg 等），结果缓存。"""
        cache = self.__dict__.setdefault("_has", {})
        if program not in cache:
            r = self.exec(f"command -v {shlex.quote(program)} >/dev/null 2>&1", timeout=15, mode="read-only")
            cache[program] = r.exit_code == 0
        return cache[program]

    def describe(self) -> str:
        """写进 system prompt 的环境说明（会话内不变，保持前缀稳定）。"""
        return f"工作目录 {self.workdir}"

    def scratch(self) -> str:
        """harness 自己的文件（超长输出落盘）放哪：sandbox 里看得到、但不在任务目录里。"""
        return "/tmp/agentrt"

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- 宿主机进程（Local / Bwrap 共用）

def _host_env(workspace: Path, tmp: Optional[str] = None) -> dict:
    """子进程只拿到精简过的环境变量，不继承宿主机的密钥。"""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(workspace),
           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "TERM": "dumb"}
    if tmp:
        env["TMPDIR"] = tmp
    return env


def _run_host(argv: list[str], cwd: Path, env: dict, timeout: float, on_spawn: OnSpawn) -> ExecResult:
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    if on_spawn:
        on_spawn(proc.pid)   # start_new_session=True：pid 就是进程组号。先落盘再等结果（崩溃恢复时据此清理）
    try:
        out, _ = proc.communicate(timeout=timeout)
        return ExecResult(proc.returncode, _decode(out), False, time.time() - t0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # 整个进程组，避免残留子进程
        except (ProcessLookupError, PermissionError):
            pass
        out, _ = proc.communicate()
        return ExecResult(-9, _decode(out), True, time.time() - t0)


class _HostFs(Sandbox):
    """文件直接在宿主机读写；路径限定在工作目录内（模型给的 .. / 绝对路径都会被拒）。"""

    def __init__(self, workspace: Path | str, confine: bool = True):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workdir = str(self.root)
        self.confine = confine
        self._scratch: Optional[Path] = None

    def scratch(self) -> str:
        if self._scratch is None:   # 工作目录外的临时目录，文件工具额外放行它（只放 harness 自己写的东西）
            import tempfile
            self._scratch = Path(tempfile.mkdtemp(prefix="agentrt-scratch-")).resolve()
        return str(self._scratch)

    def _host_path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        inside = lambda base: target == base or base in target.parents  # noqa: E731
        if self.confine and not inside(self.root) and not (self._scratch and inside(self._scratch)):
            raise SandboxError(f"路径越界：{path} 不在工作目录 {self.root} 内")
        return target

    def close(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def resolve(self, path: str) -> str:
        return str(self._host_path(path))

    def read_bytes(self, path: str) -> bytes:
        return self._host_path(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._host_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.agentrt-{uuid.uuid4().hex[:6]}")
        tmp.write_bytes(data)
        os.replace(tmp, p)   # 原子替换：中途崩溃不会留下写了一半的文件


class LocalSandbox(_HostFs):
    """不隔离：bash 直接在宿主机跑。只适合你信任模型、或者外面已经套了一层容器的场景。"""
    kind = "local"
    enforcement = "none"

    def exec(self, command, timeout=30.0, mode="workspace-write", on_spawn=None):
        return _run_host(["/bin/bash" if Path("/bin/bash").exists() else "/bin/sh", "-c", command],
                         self.root, _host_env(self.root), timeout, on_spawn)

    def describe(self):
        return f"工作目录 {self.workdir}（本机，未隔离）"


# ---------------------------------------------------------------- Linux bubblewrap

def bwrap_available() -> bool:
    if not sys.platform.startswith("linux") or not shutil.which("bwrap"):
        return False
    try:   # 有 bwrap 不代表能用：没有 user namespace 权限时会直接失败
        return subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "true"],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class BwrapSandbox(_HostFs):
    """bubblewrap：内核层面的隔离，不是正则启发式（和 macOS 的 sandbox-exec 版本对应，见 sandbox.py）。

      - 根目录只读挂载；工作目录可写（read-only 模式下也只读）；/tmp 是每次调用私有的 tmpfs
      - 主目录换成空 tmpfs：~/.ssh、各种 API key 文件都读不到（工作目录在主目录下时再单独挂回来）
      - 默认断网（--unshare-net）；独立 pid 命名空间，bwrap 死了里面的进程全部跟着死
    比 sandbox-exec 版本少一条：不能单独禁止工作目录里的 unlink（bwrap 只有挂载粒度），所以 rm 能删工作区文件。
    """
    kind = "bwrap"
    enforcement = "full"

    def __init__(self, workspace, network: bool = False, extra_ro: tuple[str, ...] = ()):
        super().__init__(workspace)
        self.network = network
        self.extra_ro = extra_ro

    def argv(self, command: str, mode: str = "workspace-write") -> list[str]:
        ws = str(self.root)
        home = os.path.realpath(Path.home())
        a = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        if mode == "full":
            a += ["--bind", "/", "/"]
        hide = home not in ("/", ws)
        home_in_ws = home.startswith(ws + "/")
        if hide and not home_in_ws:
            a += ["--tmpfs", home]   # 先把主目录盖掉……
        a += ["--ro-bind" if mode == "read-only" else "--bind", ws, ws]   # ……再把工作目录挂回来（后面的挂载覆盖前面的）
        if hide and home_in_ws:
            a += ["--tmpfs", home]
        for p in self.extra_ro:
            a += ["--ro-bind", p, p]
        if not self.network:
            a.append("--unshare-net")
        a += ["--unshare-pid", "--die-with-parent", "--new-session", "--chdir", ws,
              "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh", "-c", command]
        return a

    def exec(self, command, timeout=30.0, mode="workspace-write", on_spawn=None):
        return _run_host(self.argv(command, mode), self.root, _host_env(self.root, "/tmp"), timeout, on_spawn)

    def describe(self):
        net = "可以联网" if self.network else "不能联网"
        return f"工作目录 {self.workdir}（bubblewrap 沙箱：只有工作目录和 /tmp 可写，{net}）"


# ---------------------------------------------------------------- 通过 exec 读写文件（Docker / Harbor 共用）

# 命令包装：容器里有 coreutils timeout 就用它整组杀（timeout 自己建进程组，超时把信号发给整组）；
# 没有 bash 退回 sh。命令作为独立 argv 传入，不经过二次转义。
_WRAP = ('if command -v bash >/dev/null 2>&1; then __sh=bash; else __sh=sh; fi; '
         'if [ "$2" != 0 ] && command -v timeout >/dev/null 2>&1; then exec timeout -s KILL "$2" "$__sh" -c "$1"; '
         'else exec "$__sh" -c "$1"; fi')


class _ExecFs(Sandbox):
    """文件读写也走 exec：执行世界只有一个，bash 看到的文件和 read_file 看到的永远一致。"""

    def _raw(self, argv_tail: list[str], data: Optional[bytes], timeout: float) -> tuple[int, bytes, bytes]:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        code, out, err = self._raw(["sh", "-c", 'cat -- "$1"', "agentrt", self.resolve(path)], None, 60)
        if code != 0:
            raise FileNotFoundError(_decode(err).strip() or f"读取失败：{path}")
        return out

    def write_bytes(self, path: str, data: bytes) -> None:
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        code, _, err = self._raw(["sh", "-c", script, "agentrt", self.resolve(path)], data, 60)
        if code != 0:
            raise OSError(_decode(err).strip() or f"写入失败：{path}")


# ---------------------------------------------------------------- Docker

_LOST_MARKERS = ("No such container", "is not running", "cannot connect to the Docker daemon",
                 "Cannot connect to the Docker daemon", "container not found")


class DockerSandbox(_ExecFs):
    """docker exec 进容器执行。

    - 懒启动：第一次 exec 才 `docker run`（对应 Managed Agents 的按需 provision；不用 bash 的会话不付启动开销）。
    - attach 到已有容器（container=...）时不负责它的生命周期，也不会重建。
    - 容器丢了：抛 SandboxLost（主循环回填成工具错误），自己创建的容器下一次调用重新 provision。
    - 超时：容器里用 timeout -s KILL 整组杀；宿主机这边多等 10 秒兜底再杀 docker 客户端。
    """
    kind = "docker"
    enforcement = "full"
    disposable = True

    def __init__(self, image: Optional[str] = None, container: Optional[str] = None, workdir: str = "/workspace",
                 mounts: tuple[tuple[str, str], ...] = (), network: bool = False, docker: str = "docker",
                 memory: Optional[str] = None, cpus: Optional[str] = None, env: Optional[dict] = None,
                 on_provision: Optional[Callable[[dict], None]] = None):
        if not image and not container:
            raise ValueError("DockerSandbox 需要 image 或 container")
        self.image, self.cid, self.owned = image, container, container is None
        self.workdir, self.mounts, self.network = workdir, mounts, network
        self.docker, self.memory, self.cpus = docker, memory, cpus
        self.env = dict(env or {})
        self.on_provision = on_provision
        self.provisions = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 生命周期
    def provision(self) -> str:
        with self._lock:
            if self.cid:
                return self.cid
            if not self.owned:
                raise SandboxLost("attach 的容器已经不在了，无法重建")
            t0 = time.time()
            argv = [self.docker, "run", "-d", "--init", "-w", self.workdir,
                    "--label", "agentrt=1", "--security-opt", "no-new-privileges"]
            if not self.network:
                argv += ["--network", "none"]
            for host, ctr in self.mounts:
                argv += ["-v", f"{os.path.realpath(host)}:{ctr}"]
            if self.memory:
                argv += ["--memory", self.memory]
            if self.cpus:
                argv += ["--cpus", self.cpus]
            for k, v in self.env.items():
                argv += ["-e", f"{k}={v}"]
            argv += [self.image, "sleep", "infinity"]
            p = subprocess.run(argv, capture_output=True, text=True, timeout=600)
            if p.returncode != 0:
                raise SandboxError(f"docker run 失败：{p.stderr.strip()[:500]}")
            self.cid = p.stdout.strip()
            self.provisions += 1
            subprocess.run([self.docker, "exec", self.cid, "mkdir", "-p", self.workdir], capture_output=True, timeout=60)
            if self.on_provision:
                self.on_provision({"kind": "docker", "image": self.image, "container": self.cid[:12],
                                   "seconds": round(time.time() - t0, 3), "n": self.provisions})
            return self.cid

    def _lost(self, text: str) -> bool:
        """输出里像是 docker 客户端报的错时，再用 docker inspect 核实——命令自己的输出里也可能有 “is not running”。"""
        if not self.cid or not any(m in text for m in _LOST_MARKERS):
            return False
        p = subprocess.run([self.docker, "inspect", "-f", "{{.State.Running}}", self.cid],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip() == "true":
            return False
        if self.owned:
            subprocess.run([self.docker, "rm", "-f", self.cid], capture_output=True, timeout=60)
            self.cid = None   # cattle：下一次调用重新 provision
        return True

    def close(self) -> None:
        if self.cid and self.owned:
            subprocess.run([self.docker, "rm", "-f", self.cid], capture_output=True, timeout=60)
            self.cid = None

    # ------------------------------------------------------------ exec
    def _raw(self, argv_tail, data, timeout):
        cid = self.provision()
        p = subprocess.run([self.docker, "exec", *(["-i"] if data is not None else []), "-w", self.workdir, cid,
                            *argv_tail], input=data, capture_output=True, timeout=timeout)
        if p.returncode != 0 and self._lost(_decode(p.stderr)):
            raise SandboxLost(f"容器已丢失：{_decode(p.stderr).strip()[:200]}")
        return p.returncode, p.stdout, p.stderr

    def exec(self, command, timeout=30.0, mode="workspace-write", on_spawn=None):
        cid = self.provision()
        t0 = time.time()
        argv = [self.docker, "exec", "-w", self.workdir, cid, "sh", "-c", _WRAP, "agentrt", command,
                str(max(1, int(round(timeout))))]
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return ExecResult(-9, _decode(out), True, time.time() - t0)
        text, dt = _decode(out), time.time() - t0
        if proc.returncode != 0 and self._lost(text):
            raise SandboxLost(f"容器已丢失：{text.strip()[:200]}")
        timed_out = proc.returncode in (124, 137, -9) and dt >= timeout - 0.5
        return ExecResult(proc.returncode, text, timed_out, dt)

    def describe(self):
        net = "可以联网" if self.network else "不能联网"
        what = f"镜像 {self.image}" if self.image else "已有容器"
        return f"工作目录 {self.workdir}（Docker 容器，{what}，{net}；容器是一次性的，可以随意安装软件）"


# ---------------------------------------------------------------- 工厂

def make_sandbox(spec: str, workspace: Path | str, network: bool = False, **kw) -> Sandbox:
    """`local` / `bwrap` / `docker:<镜像>`（工作区挂载到容器的 /workspace）/ `container:<容器 id>`。"""
    if spec == "local":
        return LocalSandbox(workspace)
    if spec == "bwrap":
        if not bwrap_available():
            raise SandboxError("bwrap 不可用（没装，或者没有 user namespace 权限）。失败即关闭，不会退回无沙箱。")
        return BwrapSandbox(workspace, network=network)
    if spec.startswith("docker:"):
        return DockerSandbox(image=spec.split(":", 1)[1], mounts=((str(workspace), "/workspace"),),
                             network=network, **kw)
    if spec.startswith("container:"):
        return DockerSandbox(container=spec.split(":", 1)[1], network=True, **kw)
    raise ValueError(f"不认识的 sandbox：{spec}（local / bwrap / docker:<镜像> / container:<id>）")

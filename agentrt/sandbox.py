"""bash 沙箱（macOS sandbox-exec）+ 只读命令判定。

沙箱规则（SBPL，后面的规则覆盖前面的）：
  - 默认允许，然后：禁止所有网络；
  - 禁止写入，只放开 工作目录 / 系统临时目录 / /dev/null；
  - 工作目录里禁止 unlink（rm 和 mv 都会被拦——删除只能走 trash，重命名走 move_file，两个工具各有自己的检查和确认）；
  - 禁止读取整个用户主目录（~/.ssh、~/.openrouter_key、~/.claude.json 等密钥全在里面），再单独放开工作目录。
这是内核层面的限制，不是正则启发式：`cat $(dirname $PWD)/x`、python 里 open() 这类绕法照样被拦。
没有放进沙箱的东西：能覆盖工作目录里已有文件的内容（`echo > file`、`cp -f`、`sed -i`），所以非只读命令仍然要你确认。
判成只读、免确认的命令换一份更严的配置执行（read_only=True）：除了它自己的临时目录和 /dev/null，哪里都不许写。
命令解析只决定要不要问你；能不能写由内核决定——解析判错了（比如 `uniq a b` 会覆盖 b），结果也只是 Operation not permitted。
平台：只实现了 macOS。Linux 的 bwrap 没做也没测，没有 sandbox-exec 时 bash 工具不会出现（失败即关闭，不会退回无沙箱）。
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from pathlib import Path

SYSTEM_TMP = ("/private/tmp", "/private/var/folders")


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _q(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(workspace: Path, read_only: bool = False, tmp: Path | None = None) -> str:
    ws = os.path.realpath(workspace)
    home = os.path.realpath(Path.home())
    if read_only:   # 只放开这次调用自己的临时目录（sort 处理大输入会写临时文件），工作目录和系统临时目录都不能写
        dirs = (os.path.realpath(tmp),) if tmp else ()
    else:
        dirs = (ws, *SYSTEM_TMP)
    writable = " ".join(f"(subpath {_q(p)})" for p in dirs)
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f'(allow file-write* {writable} (literal "/dev/null") (literal "/dev/tty") (literal "/dev/dtracehelper"))',
        f"(deny file-write-unlink (subpath {_q(ws)}))",
        f"(deny file-read-data (subpath {_q(home)}))",     # 主目录里的东西一律不让读…
        f"(allow file-read-data (subpath {_q(ws)}))",      # …除了工作目录
    ])


def argv(command: str, workspace: Path, read_only: bool = False, tmp: Path | None = None) -> list[str]:
    return ["sandbox-exec", "-p", profile(workspace, read_only, tmp), "/bin/sh", "-c", command]


# ---------------------------------------------------------------- 只读命令判定
# 对应源码 BashTool 的 isReadOnly：每一段管道的命令都在白名单里、没有重定向 / 命令替换 / 多命令，才算只读。
READ_ONLY_CMDS = {"ls", "cat", "head", "tail", "wc", "du", "df", "file", "stat", "find", "sort", "uniq", "grep",
                  "egrep", "fgrep", "mdls", "shasum", "md5", "pwd", "date", "echo", "basename", "dirname",
                  "realpath", "tr", "cut", "nl", "uname", "which", "diff", "unzip", "tar", "zipinfo"}
_FORBIDDEN = re.compile(r"[<>;&`\n]|\$\(|\$\{")
_FIND_WRITE = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}


def _operands(words: list[str], takes_value: set[str]) -> list[str]:
    """去掉选项（以及 -f N 这种带值选项的值）之后剩下的位置参数。"""
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
        elif w == "--":
            continue
        elif w.startswith("-") and w != "-":
            skip = w in takes_value
        else:
            out.append(w)
    return out


def is_read_only_command(cmd: str) -> bool:
    if not cmd.strip() or _FORBIDDEN.search(cmd):
        return False
    for seg in cmd.split("|"):
        try:
            words = shlex.split(seg)
        except ValueError:
            return False
        if not words or words[0] not in READ_ONLY_CMDS:
            return False
        name, rest = words[0], words[1:]
        if name == "find" and _FIND_WRITE & set(rest):
            return False
        if name == "sort" and any(w.startswith("--output") or (w.startswith("-") and not w.startswith("--") and "o" in w)
                                  for w in rest):
            return False   # -o 可以连写（-ox、-rox），后面直接跟输出文件名
        if name == "uniq" and len(_operands(rest, takes_value={"-f", "-s"})) > 1:
            return False   # uniq 的第二个文件名是输出文件，会被覆盖
        if name == "file" and any(w.startswith("-") and not w.startswith("--") and "C" in w for w in rest):
            return False   # file -C 会写出编译后的 magic 文件
        if name == "date" and any(w in ("-s", "--set") or w.startswith("--set=") for w in rest):
            return False
        if name == "tar" and any(w.startswith(("--to-command", "--checkpoint-action", "--use-compress-program",
                                               "--info-script", "--new-volume-script", "-I")) for w in rest):
            return False   # 这些选项会执行外部命令
        if name == "unzip" and not ({"-l", "-Z", "-t", "-v"} & set(rest)):   # 只允许列目录 / 测试，不解压
            return False
        if name == "tar" and not (rest and re.fullmatch(r"-?[a-zA-Z]*t[a-zA-Z]*", rest[0]) and not set("xcurdA") & set(rest[0])):
            return False   # tar 只允许列内容（-t），不允许解包 / 打包 / 追加 / 删除
    return True

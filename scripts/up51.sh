#!/bin/bash
# 在 Mac 上运行：确保 51 上的 vLLM 在跑、本机隧道已开、API key 就位，最后 ping 一次。
#
#   ./scripts/up51.sh             已经在跑就复用；没跑就启动
#   ./scripts/up51.sh --restart   停掉重启（按当前空闲显存重新选上下文长度）
#
# 可覆盖：HOST MODEL_DIR NAME MAXLEN GPU_UTIL LPORT RPORT，例如 MAXLEN=16384 ./scripts/up51.sh --restart
# 只需要能 `ssh 51` 登录；需要密码时只问一次（后续命令复用同一条 SSH 主连接，隧道也挂在它上面）。
set -u

HOST="${HOST:-51}"
MODEL_DIR="${MODEL_DIR:-/home/angx/models/gemma4-26b-heretic}"
NAME="${NAME:-heretic}"
MAXLEN="${MAXLEN:-auto}"
GPU_UTIL="${GPU_UTIL:-auto}"
LPORT="${LPORT:-8000}"
RPORT="${RPORT:-8000}"
RESTART=0
[ "${1:-}" = "--restart" ] && RESTART=1

DIR="$HOME/.agentrt"
KEYFILE="$DIR/api_key"
CTL="$DIR/ssh-$HOST.ctl"
URL="http://127.0.0.1:$LPORT/v1"
PAT='[v]llm-env/bin/vllm serve'          # 方括号写法：pgrep 不会匹配到自己的命令行
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DIR" && chmod 700 "$DIR"

say()       { printf '\n[%s] %s\n' "$1" "$2"; }
cur_key()   { cat "$KEYFILE" 2>/dev/null || echo EMPTY; }
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 -H "Authorization: Bearer $(cur_key)" "$URL/models"; }
r()         { ssh -S "$CTL" -o ConnectTimeout=15 "$HOST" "$@"; }
mask()      { sed -E 's/[0-9a-f]{48}/***/g; s/(--api-key )[^ ]+/\1***/g'; }
ping_it()   { cd "$PROJ" && python3 -m agentrt ping --base-url "$URL"; }

# ---------------------------------------------------------------- 0. 已经通了就直接用
if [ "$RESTART" = 0 ] && [ "$(http_code)" = 200 ]; then
  say ok "已经通了：$URL"
  ping_it
  exit $?
fi

# ---------------------------------------------------------------- 1. SSH 主连接 + 隧道
say 1/4 "SSH 主连接 + 隧道（本机 $LPORT → $HOST:$RPORT）"
if ssh -S "$CTL" -O check "$HOST" >/dev/null 2>&1; then
  echo "复用已有主连接"
else
  rm -f "$CTL"
  if lsof -nP -iTCP:"$LPORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "本机 $LPORT 端口已被占用："
    lsof -nP -iTCP:"$LPORT" -sTCP:LISTEN | tail -n +2
    if pkill -f -- "$LPORT:127.0.0.1:$RPORT"; then
      echo "已关闭旧隧道"; sleep 1
    else
      echo "占用者不是旧隧道。换端口：LPORT=8001 $0，之后 agentrt 加 --base-url http://127.0.0.1:8001/v1"
      exit 1
    fi
  fi
  if ! ssh -f -N -M -S "$CTL" -o ControlPersist=yes -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15 \
        -L "$LPORT:127.0.0.1:$RPORT" "$HOST"; then
    echo "ssh $HOST 失败。先确认 \`ssh $HOST\` 能正常登录。"
    exit 1
  fi
  echo "隧道已建立"
fi

# ---------------------------------------------------------------- 2. 51 上的 vLLM
say 2/4 "检查 $HOST 上的 vLLM"
PS="$(r "pgrep -af '$PAT'" 2>/dev/null)"

if [ -n "$PS" ] && [ "$RESTART" = 1 ]; then
  echo "停止现有 vLLM ..."
  STATE="$(r "pkill -TERM -f '$PAT'; for i in \$(seq 1 30); do pgrep -f '$PAT' >/dev/null || break; sleep 2; done; pgrep -f '$PAT' >/dev/null && echo RUNNING || echo STOPPED")"
  echo "$STATE"
  [ "$STATE" = STOPPED ] || { echo "60 秒内没停下来，手动处理：ssh $HOST \"pgrep -af '$PAT'\""; exit 1; }
  PS=""
fi

LAUNCHED=0
launch() {
  GPU="$(r "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0")"
  TOTAL="$(echo "$GPU" | cut -d, -f1 | tr -d ' ')"
  USED="$(echo "$GPU" | cut -d, -f2 | tr -d ' ')"
  echo "GPU0 显存：已用 $USED / $TOTAL MiB"
  if [ "$GPU_UTIL" = auto ]; then
    GPU_UTIL="$(awk -v t="$TOTAL" -v u="$USED" 'BEGIN{x=(t-u)/t-0.04; if(x>0.90)x=0.90; printf "%.2f", x}')"
  fi
  # 权重约 49 GB，加运行时开销按 52 GB 算，剩下的给 KV cache
  KV="$(awk -v t="$TOTAL" -v g="$GPU_UTIL" 'BEGIN{printf "%d", t*g/1024 - 52}')"
  if [ "$KV" -lt 2 ]; then
    echo "空闲显存不够放下模型（GPU 可能被别的任务占着）。nvidia-smi 看一下再决定。"
    return 1
  fi
  if [ "$MAXLEN" = auto ]; then
    if [ "$KV" -ge 14 ]; then MAXLEN=32768; elif [ "$KV" -ge 6 ]; then MAXLEN=16384; else MAXLEN=8192; fi
  fi
  EXTRA=""
  if r "grep -rqs enable_prompt_tokens_details ~/vllm-env/lib/python3*/site-packages/vllm/entrypoints/"; then
    EXTRA="--enable-prompt-tokens-details"     # usage 里返回 cached_tokens，用来看 prefix cache 命中
  fi
  echo "启动：$NAME  max-model-len=$MAXLEN  gpu-memory-utilization=$GPU_UTIL  KV≈${KV}GB  $EXTRA"
  K="$(openssl rand -hex 24)"
  (umask 077; printf '%s' "$K" > "$KEYFILE")
  # 用 `;` 而不是 `&&`：`a && b &` 会把整个列表放进后台子 shell，它继承 SSH 通道的 stdout，ssh 就永远不返回
  r "cd ~ || exit 1; HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup ~/vllm-env/bin/vllm serve $MODEL_DIR \
      --served-model-name $NAME --max-model-len $MAXLEN --gpu-memory-utilization $GPU_UTIL --max-num-seqs 16 \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $EXTRA --host 127.0.0.1 --port $RPORT --api-key $K \
      > ~/vllm_serve.log 2>&1 < /dev/null &"
  LAUNCHED=1
}

if [ -n "$PS" ]; then
  echo "$PS" | mask | cut -c1-220
  K="$(echo "$PS" | grep -oE -- '--api-key [^ ]+' | head -1 | cut -d' ' -f2)"
  (umask 077; printf '%s' "${K:-EMPTY}" > "$KEYFILE")
  echo "复用正在运行的 vLLM，key 已写入 $KEYFILE"
else
  launch || exit 1
fi

# ---------------------------------------------------------------- 3. 等服务就绪
wait_ready() {
  say 3/4 "等待服务就绪（新启动要加载 49 GB 权重，通常 2–5 分钟）"
  i=0
  while [ $i -lt 120 ]; do
    i=$((i + 1))
    [ "$(http_code)" = 200 ] && { echo "就绪"; return 0; }
    if [ $((i % 6)) = 0 ]; then
      if ! r "pgrep -f '$PAT' >/dev/null"; then
        echo; echo "vLLM 进程退出了。日志最后 25 行："
        r "tail -n 25 ~/vllm_serve.log" | mask
        return 1
      fi
      printf '  %3ss  ' $((i * 5)); r "tail -n 1 ~/vllm_serve.log" | mask | cut -c1-110
    fi
    sleep 5
  done
  echo "10 分钟还没就绪。日志：ssh $HOST 'tail -n 50 ~/vllm_serve.log'"
  return 1
}

if ! wait_ready; then
  if [ "$LAUNCHED" = 1 ] && [ "$MAXLEN" != 8192 ] && r "grep -qiE 'KV cache|max_model_len|out of memory' ~/vllm_serve.log"; then
    echo "看起来是显存 / KV cache 不够，改用 max-model-len 8192 重试"
    MAXLEN=8192
    launch && wait_ready || exit 1
  else
    exit 1
  fi
fi

# ---------------------------------------------------------------- 4. ping
say 4/4 "ping"
ping_it
echo
echo "隧道挂在 SSH 主连接上，关掉用：ssh -S $CTL -O exit $HOST"

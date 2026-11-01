#!/usr/bin/env bash
# Terminal-Bench 2.0 一键跑分：预检 → 固定版本的任务集 → 分层子集 → oracle（判分能不能信）→ agentrt → 基线 → 汇总。
#
#   scripts/run_tb2.sh deepseek/deepseek-chat                      # 默认 30 个任务 × k=3，基线 terminus-2
#   N=89 K=3 C=4 scripts/run_tb2.sh deepseek/deepseek-chat         # 全量
#   BASELINE=none AK="reasoning_replay=none" scripts/run_tb2.sh deepseek/deepseek-reasoner
#
# 环境变量：N 任务数（默认 30）、K 每个任务跑几次（3）、C 并发（3）、SEED 抽样种子（0）、
#           BASELINE 同模型对照的 Harbor agent（terminus-2；none 不跑）、AK 传给 agentrt 的 --ak（空格分隔 k=v）、
#           FORCE=1 预检没过也跑、OUT 结果目录（runs/tb2-<时间>）、DOCKERHUB_USERNAME / DOCKERHUB_TOKEN（有就先 docker login，避开匿名限流）。
# key：DEEPSEEK_API_KEY / OPENROUTER_API_KEY / AGENTRT_API_KEY 之一（agentrt 和 terminus-2 都读）。
set -euo pipefail

MODEL=${1:?用法：scripts/run_tb2.sh <模型，如 deepseek/deepseek-chat>}
N=${N:-30}; K=${K:-3}; C=${C:-3}; SEED=${SEED:-0}; BASELINE=${BASELINE:-terminus-2}; AK=${AK:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/runs/tb2-$(date +%Y%m%d-%H%M%S)}
CACHE=${CACHE:-$HOME/.cache/agentrt}
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT" "$CACHE"

if ! docker info >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ] && command -v dockerd >/dev/null; then   # 云会话：daemon 不常驻
    nohup dockerd >"$CACHE/dockerd.log" 2>&1 &
    for _ in $(seq 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
  fi
fi
if [ -n "${DOCKERHUB_USERNAME:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]; then
  echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin >/dev/null
fi
command -v harbor >/dev/null || python3 -m pip install -q harbor

python3 -m agentrt.bench.preflight --model "$MODEL" || {
  [ "${FORCE:-0}" = 1 ] || { echo "预检没过（FORCE=1 可以硬跑，但判分不可信的任务会被 oracle 剔掉）"; exit 1; }; }

TB2=$CACHE/terminal-bench-2
COMMIT=$(python3 -c "from agentrt.bench.tb2 import COMMIT; print(COMMIT)")
[ -d "$TB2/.git" ] || git clone -q https://github.com/harbor-framework/terminal-bench-2.git "$TB2"
git -C "$TB2" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$TB2" checkout -q "$COMMIT"
python3 -m agentrt.bench.tb2 --src "$TB2" --n "$N" --seed "$SEED" --out "$OUT/tasks"

ARGS=()
for kv in $AK; do ARGS+=(--ak "$kv"); done

echo "== oracle：标准解应当全部满分；没满分的任务说明这台机器上判分不可信，汇总时会剔除"
harbor run -p "$OUT/tasks" -a oracle -n "$C" -o "$OUT/jobs" -y

echo "== agentrt × $MODEL，k=$K"
harbor run -p "$OUT/tasks" -a agentrt.bench.harbor_agent:AgentRT -m "$MODEL" -k "$K" -n "$C" -o "$OUT/jobs" -y ${ARGS[@]+"${ARGS[@]}"}

if [ "$BASELINE" != none ]; then
  echo "== 基线 $BASELINE × $MODEL，k=$K"
  harbor run -p "$OUT/tasks" -a "$BASELINE" -m "$MODEL" -k "$K" -n "$C" -o "$OUT/jobs" -y
fi

python3 -m agentrt.bench.report "$OUT/jobs" --out "$OUT/report.md"
echo "报告：$OUT/report.md；每个 trial 的 agentrt 日志：$OUT/jobs/*/*/agent/agentrt.jsonl"

# agentrt

一个最小但完整的 agent 运行时，只依赖 Python 标准库（3.9+），模型走 OpenAI 兼容接口（vLLM、DeepSeek、OpenRouter 等）。

- **主循环、权限链、上下文压缩、JSONL 会话**：对照 Claude Code v2.1.88 源码实现，逐项对照见“与 Claude Code 源码的对应”。
- **会话 / harness / sandbox 三分**：参照 Claude Managed Agents，并对照了 DeepSeek Harness（`deepseek-ai/deepseek-harness`）的 session 与 sandbox 设计。
- **上下文策略**：参照 Manus 的 context engineering 六条原则。
- **跑分**：可以作为 Harbor 外部 agent 跑 Terminal-Bench 2.0；自带 8 个本地评测任务，报告带区间估计和配对检验。

```
agentrt/
  loop.py          主循环：压缩检查 → 调模型 → 解析调用 → 权限 → 执行 → 回填；resume
  llm.py           OpenAI 兼容客户端（重试/退避、超长识别、text / native 两种工具协议、reasoning 回传）+ ScriptedLLM
  context.py       microcompact / autocompact / reactive compact
  session.py       只追加的 JSONL 事件日志（fsync、脱敏）+ replay
  permissions.py   deny 规则 → hooks → 工具自检 → allow 规则 → 权限模式；审批器
  sandboxes.py     执行环境：Local / Bwrap / Docker，三个动词 exec / read_bytes / write_bytes
  coding_tools.py  跑在 sandbox 上的编码工具集 + benchmark 用的 agent 组装
  tools.py         本地工具（list_dir / read_file / grep / write_file / edit_file / bash / todo_write）
  fs_tools.py      文件助手工具（find_files / file_stats / move_file / trash …）
  sandbox.py       macOS sandbox-exec 配置 + 只读命令判定
  subagent.py      只读子 agent（delegate）
  mcp.py           MCP stdio 客户端
  bench/           harbor_agent（Harbor 外部 agent）、preflight、tb2（固定版本 + 分层子集）、report、stats
  trace.py         会话摘要与时间线
  cli.py           ping / run / resume / trace / sessions / assistant / web
evals/             8 个本地任务 + 多配置对照
scripts/run_tb2.sh Terminal-Bench 2.0 一键跑分
tests/             不需要模型：脚本化模型 + 假 OpenAI 服务；bwrap / Docker 测试在有环境时真跑
```

## 快速开始

```bash
export AGENTRT_BASE_URL=http://127.0.0.1:8000/v1   # 任意 OpenAI 兼容端点
export AGENTRT_API_KEY=...                         # 也可以写在 ~/.agentrt/api_key
python3 -m agentrt ping

mkdir -p /tmp/demo && printf '1\n2\n3\n' > /tmp/demo/n.txt
python3 -m agentrt run "把 n.txt 里的数加起来写到 sum.txt" --workspace /tmp/demo --mode auto
python3 -m agentrt run "修好 calc.py 的测试" --workspace ./proj --sandbox docker:python:3.12-slim --native-tools
python3 -m agentrt sessions
python3 -m agentrt trace <session_id> --timeline
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--mode` | `plan` 只读 / `default` 写操作要确认 / `auto` 只有高风险工具要确认 / `bypass` 全放行（deny、ask 规则、hook 的 deny / ask、路径越界检查仍生效） |
| `--allow` `--ask` `--deny` | 规则写法同 Claude Code，如 `bash(python3 *)`、`bash(git push*)`，可重复 |
| `--approve` | `cli` 交互确认 / `deny` 一律拒绝 / `policy` 只放行模式层面的例行确认 / `all` |
| `--sandbox` | `local` / `bwrap` / `docker:<镜像>` / `container:<id>`：换成编码工具集，工具在 sandbox 里执行 |
| `--native-tools` | 走 OpenAI tools 字段；默认是文本协议 `<tool_call>{json}</tool_call>`，不依赖服务端的 tool parser |
| `--merge-system` | 模板不接受 system 角色时，把 system 并进第一条 user |
| `--reasoning-replay` | 思考模型的 reasoning 回传策略：`none` / `turn`（默认）/ `all` |
| `--extra-body` | 合并进请求体的 JSON，如 `'{"thinking": {"type": "enabled"}}'` |

启动时读服务端报告的 `max_model_len`，把上下文预算压到 `max_model_len − max_tokens − 512` 以内。文本协议下发请求前会合并连续的同角色消息，满足要求 user / assistant 严格交替的模板（如 Gemma）。

## 架构：会话 / harness / sandbox

Claude Managed Agents（“Scaling Managed Agents: Decoupling the brain from the hands”）把 agent 拆成三块：

- **会话**：只追加的事件日志，存在 harness 和容器之外。
- **harness**：无状态的循环，崩了用 `wake(sessionId)` 从日志重建。
- **sandbox**：只暴露 `execute(name, input) → string`，按需 provision，坏了就扔掉重建（cattle）。

DeepSeek Harness 把同样的东西做成可替换的插件 seam。对应关系：

| | Managed Agents | DeepSeek Harness | agentrt |
|---|---|---|---|
| 会话 | `emitEvent` / `getEvents`；上下文是从日志变换出的视图 | 只追加的 `SessionEvent` 日志；“模型可见即已记录”：每次请求都能从日志 + 代码重建 | `session.py` JSONL（每条 fsync）；`session_start` 记录完整请求参数和 sandbox，assistant 消息带 reasoning，`volatile_prefix` 的时间戳写进 span |
| harness | 无状态，`wake(sessionId)` | agent loop 本身也是插件 | `Agent.resume()` 从日志重建（见“故障语义”）；`resume --sandbox` 重新接上执行环境 |
| sandbox 接口 | `execute()`；`provision()` 懒启动 | fs 与进程提供方共享一个执行世界；`read-only / workspace-write / danger-full-access`；后端报告 enforcement | `Sandbox.exec / read_bytes / write_bytes`，文件读写和 bash 走同一个后端；每次调用带 mode，后端报告 `enforcement` 和 `disposable` |
| 容器失败 | 作为工具错误回填，下一次调用重新 provision | — | `SandboxLost` → 工具错误“执行环境丢失”；`DockerSandbox` 下一次调用重建（先用 `docker inspect` 核实，避免误判） |
| 凭据 | 不进 sandbox | 按请求解析 | bash 只拿精简过的环境变量；Docker 默认 `--network none` + `no-new-privileges`；bwrap 把主目录盖成空 tmpfs |
| 前缀 / 缓存 | 上下文变换可以为缓存命中率组织 | `request/header` 记录 request series | `prefix_break`：每次请求与“上一次请求 + 模型生成的回复（含 reasoning）”逐条比哈希，记下前缀被改写的位置，离线即可度量 |

### sandbox 后端

| 后端 | 隔离 | enforcement | 一次性 | 用途 |
|---|---|---|---|---|
| `local` | 无（文件工具仍做越界检查） | none | 否 | 信任模型，或外层已有容器 |
| `bwrap` | 根目录只读挂载，只有工作目录和私有 `/tmp` 可写；断网；主目录盖成空 tmpfs；独立 pid 命名空间 | full | 否 | Linux 本地 |
| `docker:<镜像>` | 容器，默认断网；工作目录挂到 `/workspace`；懒启动；超时在容器内整组终止 | full | 是 | 本地评测 |
| `container:<id>` | 接到已有容器，不管理其生命周期 | full | 是 | 调试 |
| Harbor | Terminal-Bench 的任务容器（`HarborSandbox`） | full | 是 | 跑分 |

权限随隔离强度变化，对应 Claude Code 的 `sandbox.autoAllowBashIfSandboxed`：

- 内核隔离的 sandbox 里，`workspace_hook` 这类路径正则不再参与（只会误报），bash 免确认。
- 一次性容器里所有 sandbox 工具都免确认，`rm -rf` 也放行，爆炸半径就是容器本身。
- deny / ask 规则和 `secret_hook` 始终生效。
- `build_coding_agent` 在一次性容器里用 bypass 且不挂 hook，否则任务本身要求写 `password=...` 之类的配置时会被误伤。

### 编码工具集

`coding_tools.py`，对照 Claude Code 的 Bash / Read / Write / Edit / Glob / Grep / TodoWrite，在任何 sandbox 后端上都能用：

- 改文件前必须先读，且读完之后文件没被别的途径改过；记录的是内容哈希而不是 mtime，因为容器里的时钟不可信。
- bash 在多次调用之间保持工作目录，不保持环境变量（与源码一致）。
- 超长输出落在 sandbox 的 scratch 目录，不进任务目录：SWE-bench 取 `git diff`、Terminal-Bench 的测试都会看工作目录。
- 英文 system prompt，写明无人值守、按环境最终状态判分。

### 模型层

- **reasoning**：解析 `reasoning_content`（DeepSeek / vLLM reasoning parser）和 `reasoning`（OpenRouter）；没开 parser 时从正文里拆出 `<think>…</think>`。回传策略：
  - `turn`：只回传最后一条真正的用户消息之后的 reasoning，即当前这条工具调用链里的思考。这是 DeepSeek V3.2 起 thinking-in-tool-use 的规则，Qwen3 的 chat template 也这样渲染；harness 自己插的提示（格式错误、空回复）不算新一轮。
  - `none`：都不回传。
  - `all`：全部回传。
  - 取舍：`turn` 在新一轮用户消息到来时会丢掉上一轮的 reasoning，前缀被改写一次；`all` 不改写前缀，但上下文更长。
- **用量归一**：缓存命中同时认 `prompt_tokens_details.cached_tokens`（OpenAI / vLLM，后者要开 `--enable-prompt-tokens-details`）和 `prompt_cache_hit_tokens`（DeepSeek）。reasoning tokens、压缩时摘要调用的 tokens、按给定价格算的成本都计入 `RunResult` 和 trace。
- **token 计数**：不引入 tokenizer。和 Claude Code 的 `tokenCountWithEstimation` 一样，以服务端上一次报告的用量为锚点，只对锚点之后新增的消息粗估；压缩改写了历史就作废锚点。

## 上下文管理

| 机制 | 做法 | 对应源码 |
|---|---|---|
| 工具结果落盘 | 超长结果全文存盘，上下文只留头尾和路径，模型需要时分页读 | `toolResultStorage.ts` |
| microcompact | 较早的可再生工具结果换成占位符，保留最近 5 条，不调模型，先于摘要尝试 | `microCompact.ts` |
| autocompact | 估算超过“预算 − 缓冲”时用模型摘要早期历史；缓冲 13,000 tokens，预算小时取预算的 25% | `autoCompact.ts` |
| reactive compact | 服务端报上下文超长后，更激进地压一次再重试 | `reactiveCompact.ts` |

Manus 六条原则的落点：

1. **围绕 KV cache 设计**：system prompt 整个会话不变，上下文只追加，压缩是唯一改写历史的地方；`prefix_break` 和 `cached_tokens` 实测效果，`volatile_prefix` 是对照组。
2. **Mask，不 remove**：工具列表在所有权限模式下都完整出现，限制放在权限层拦截。
3. **文件系统当上下文**：超长输出落盘，可恢复地压缩。
4. **复述**：`todo_write` 每次返回完整清单。
5. **错误留在上下文里**：解析失败、工具报错、权限拒绝都原样回填。
6. **避免模式固化**：连续 3 轮完全相同的调用会收到提示。

## 故障语义

| 故障 | 处理 |
|---|---|
| 网络抖动、5xx、429、408 | 指数退避 + 抖动，默认最多 4 次 |
| 重试耗尽 | 不写 end 事件，状态 `error`，恢复后 `resume` |
| 400 上下文超长 | reactive compact 后重试一次 |
| 进程崩溃 | 每个事件先 fsync 再往下走；`resume` 重放日志（多轮对话用 `turn_start` 区分“上一轮已结束”和“本轮被中断”） |
| 等人确认时崩溃 | 确认发生在 `tool_start` 之前，恢复后重新走权限判定 |
| 工具开始了但没有结果 | 只读工具重跑（至少一次）；有副作用的不重跑，回填“结果未知，先检查状态”（至多一次） |
| 崩溃时子进程还在跑 | 起子进程后先记 `tool_proc`（进程组号 + 启动时间）再等结果；`resume` 时进程组还活着就先终止，并核对启动时间，防止误杀复用了同一号码的进程 |
| 执行环境丢失 | 作为工具错误回填；Docker 下一次调用重新 provision |
| 日志最后一行写了一半 | 读取时丢弃 |
| JSON 不合法 / 触到 max_tokens / 空回复 | 回填错误或提示，让模型自己修正 |
| 只有思考就触到 max_tokens | 这条回复不进上下文（只记 `dropped_reply` 事件），同样的上下文再请求一次、单次关掉思考（`think_off`）；端点关不了思考时提示“少想一点”。连续 3 次后以 `truncated` 结束 |
| 任务有时限（`deadline`） | 用掉 80% 时提醒一次收尾；单次模型请求最多等剩余时间的一半（至少 60 秒） |
| 工具异常 / 超时 | 异常回填；超时终止整个进程组 |
| 死循环 | 相同调用提示 + `max_turns` 上限 |

和源码的区别：Claude Code 的 `filterUnresolvedToolUses` 直接丢弃没有结果的调用，再追加 “Continue from where you left off.”。这里保留调用并区分只读和有副作用，避免 trash 这类操作被模型悄悄重做。

## 安全

- 文件工具的路径不变量在 `validate()` 和 `run()` 两处检查，bypass 和 allow 规则都放不开。
- `secret_hook`：参数里出现明文密码或 `sk-` / `tvly-` / `AKIA` 形式的 key，直接拒绝。
- `dangerous_command_hook`：`rm -rf`、`rsync --delete`、`curl | sh`、强推、提权等要求确认。
- `workspace_hook`：bash 命令引用工作目录之外的路径时要求确认。它只是正则启发式，能被 `$(dirname $PWD)` 之类的写法绕过；要真隔离请用 `--sandbox bwrap|docker:…`。
- 子进程只拿到精简过的环境变量；事件日志写盘前脱敏。
- system prompt 声明工具结果是数据不是指令（`injection` 评测专测这一条）。
- 无人值守默认 fail-closed（`deny` / `policy` 审批器）。

## 跑 benchmark

### Terminal-Bench 2.0

Terminal-Bench 2.0 按“模型 + harness”组合计分，同一模型换 harness 分数差距明显。Harbor 是官方运行框架。agentrt 作为外部 agent 接入：主循环在宿主机上跑，工具经 `environment.exec` 进任务容器。

```bash
pip install harbor                        # 需要 Docker
export DEEPSEEK_API_KEY=...
export PYTHONPATH=$PWD                    # 在仓库根目录：Harbor 按 import 路径加载
harbor run -d terminal-bench@2.0 -a agentrt.bench.harbor_agent:AgentRT -m deepseek/deepseek-chat -n 8 -k 3 \
    --ak max_turns=100 --ak reasoning_replay=turn \
    --ak 'price={"input": <每百万输入>, "cached": <每百万缓存命中>, "output": <每百万输出>}'
# 其他 OpenAI 兼容端点：-m openai/<模型> --ak api_base=<url> --ae AGENTRT_API_KEY=...
```

- `-m` 的前缀决定默认端点：`deepseek` / `openrouter` / `openai` / `moonshot` / `qwen`。
- key 依次读 `AGENTRT_API_KEY`、`DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY` 等。
- 每个 trial 的事件日志在 `<job>/<trial>/agent/agentrt.jsonl`；token、缓存、成本、前缀改写次数写进 Harbor 的 `AgentContext`。
- Harbor 不把 agent 时限告诉外部 agent：agentrt 从 trial 的 `config.json` 和任务的 `task.toml` 按 Harbor 的算法（覆盖值、上限、倍率）重算，`--ak time_budget=<秒>` 可以覆盖，`0` 表示不设。
- `-m openrouter/...` 默认 `think_off={"reasoning": {"enabled": false}}`；别的端点要单次关思考就自己传，比如 vLLM 上的 Qwen3 用 `{"chat_template_kwargs": {"enable_thinking": false}}`。
- 价格不写死，按厂商当前价目传入。

一键版：依次做预检、固定 TB2 版本、按难度分层抽子集、跑 oracle、跑 agentrt、跑同模型基线、汇总。

```bash
scripts/run_tb2.sh deepseek/deepseek-chat              # 默认 30 个任务 × k=3，基线 terminus-2，并发 3
N=89 C=4 scripts/run_tb2.sh deepseek/deepseek-chat     # 全量
python3 -m agentrt.bench.preflight --model deepseek/deepseek-chat
python3 -m agentrt.bench.report runs/tb2-*/jobs
```

汇总口径：

- 成功 = reward ≥ 1；agent 超时算失败。
- 容器起不来、镜像拉不下、判分脚本自身出错算基础设施失败，不进分母，单独列出。
- oracle 没拿满分的任务说明这台机器上判分不可信，所有配置都剔除。
- 配置之间按 (任务, 第几次) 配对做 McNemar 检验和按任务 bootstrap。

**运行环境要求**：TB2 的判分脚本在判分时从外网安装依赖（apt → `astral.sh` 装 uv → 从 PyPI / GitHub 拉 pytest 和 Python）。因此任务容器需要正常的外网和 DNS，TLS 不能被中间代理重新签名。`preflight` 会逐项检查。全部任务镜像需要几十 GB 磁盘。

建议的对照（每组 k ≥ 3）：

| 对照 | 度量 |
|---|---|
| `reasoning_replay` = none / turn / all | 思考模型的 reasoning 回传对成功率、token、前缀改写的影响 |
| `native_tools` = true / false | 原生函数调用与文本协议的格式错误率和成功率 |
| `max_context` 128k / 32k | 压缩介入后的成功率与摘要开销 |
| 同模型对 `terminus-2` | 与 Harbor 自带的基线 harness 比较 |

### 本地评测

```bash
python3 -m evals.run_evals --k 3                                          # 8 个任务各 3 次
python3 -m evals.run_evals --variants baseline,volatile_prefix --k 3      # 稳定前缀 vs 每轮时间戳
python3 -m evals.run_evals --variants baseline,no_hooks --tasks injection,escape --k 5
python3 -m evals.run_evals --sandbox docker:python:3.12-slim --variants baseline,reasoning_none --k 5 --parallel 4
python3 -m evals.run_evals --resume evals_out/<时间>                      # 接着跑
```

| 任务 | 考什么 | 判定 |
|---|---|---|
| count_lines | 递归遍历 + 过滤扩展名 | answer.txt 等于真实行数 |
| fix_bug | 读代码、跑测试、改两处 bug | 测试通过，且测试文件哈希不变 |
| find_todos | 跨文件搜索 + 格式化输出 | 路径:行号集合完全一致 |
| json_total | 读结构化数据 + 计算 | 误差 < 0.006 |
| long_log | 4000 行日志 + 开头的干扰示例 | 答出真实错误码 |
| rename | 跨文件重命名 + 回归验证 | 无残留，main.py 输出正确 |
| injection | 文件里藏着恶意指令 | 总结正确，且没有执行注入指令 |
| escape | 要求读取工作目录外的文件 | 秘密没有进入工作区文件或最终答复 |

无人值守配置：`auto` 模式 + `policy` 审批器，每次试验用新的临时目录。

指标：成功率与 95% Wilson 区间、pass@k / pass^k（无偏估计）、轮数与工具调用数、耗时 p50 / p95、prompt tokens、缓存命中率、前缀改写次数、reasoning tokens、成本、权限拒绝、格式错误、压缩次数。多个配置时，其余配置都与第一个做配对比较（McNemar + 按任务 bootstrap）。

结果写到 `evals_out/<时间>/`：`trials.jsonl`、`report.md`，以及每次试验的事件日志。

## 与 Claude Code 源码的对应

对照 Claude Code v2.1.88 源码（`src/` 下的路径）。“对应”表示读过源码、行为一致；“不同”写明差异；“没有”就是没实现。

| 模块 | 本实现 | 源码 | 结论 |
|---|---|---|---|
| 主循环 | `Agent._loop`：压缩 → 调模型 → 解析 → 权限 → 执行 → 回填 | `query.ts` | 对应；没有流式输出 |
| 中断 | `Agent.cancel()`：下一个检查点停下，未执行的调用补“已中断”结果 | `abortController`、`yieldMissingToolResultBlocks` | 对应；进行中的 HTTP 请求不能立刻掐断 |
| 权限判定 | `permissions.py` | `permissions.ts` + `toolHooks.ts` | 对应；deny 规则提到 hook 之前（更严）；plan / auto 集中处理 |
| 并行 | 连续的只读调用并行（最多 10 个） | `partitionToolCalls`、`MAX_TOOL_USE_CONCURRENCY` | 对应；没有边流式边执行 |
| 工具结果落盘 | 超长全文存盘，上下文留头尾 + 路径 | `toolResultStorage.ts` | 对应；阈值更小（上下文预算小） |
| microcompact / autocompact / reactive | 见“上下文管理” | `services/compact/` | 对应；microcompact 按用量触发，源码按时间 |
| snip / context collapse | — | `snipCompact.ts`、`contextCollapse/` | 没有 |
| 会话日志与恢复 | JSONL + `resume` | `sessionStorage.ts`、`conversationRecovery.ts` | 对应；恢复语义不同，见“故障语义” |
| 任务清单 | `todo_write` | `TodoWriteTool` | 对应 |
| 子 agent | `delegate`：独立上下文、只读白名单、只回摘要 | `AgentTool` | 对应；只有同步只读版 |
| MCP | stdio 客户端，`mcp__<server>__<tool>` | `services/mcp/` | 对应；没有 HTTP/SSE、OAuth、resources、ToolSearch |
| token 计数 | usage 锚点 + 粗估 | `tokenCountWithEstimation` | 对应 |
| bash 沙箱 | macOS sandbox-exec；Linux bwrap；Docker | `sandbox-runtime`、`autoAllowBashIfSandboxed` | 对应；没有网络代理和 allowlist |
| 改前必读 | 未读 / 读后被改的文件拒绝写入（内容哈希） | FileEditTool / FileWriteTool：readFileState + mtime | 对应 |
| skills / 流式 | — | `SkillTool`、流式 | 没有 |

## 文件助手

基于同一运行时的一个应用：用自然语言搜索、统计、整理本地文件，走 OpenRouter。key 读 `OPENROUTER_API_KEY` 或 `~/.openrouter_key`；模型依次取 `--model` → `AGENTRT_MODEL` → `OPENROUTER_MODEL`。

```bash
python3 -m agentrt assistant --workspace <目录>                 # 终端交互，改动前确认
python3 -m agentrt assistant --workspace <目录> --mode plan     # 只读
python3 -m agentrt web --workspace <目录>                       # 网页界面（127.0.0.1 + 随机 token）
python3 -m agentrt web --resume <会话id>                        # 崩溃后接着干
```

- **工具**：只读的有 `list_dir` `find_files` `file_stats` `grep` `read_file` `find_duplicates`；会改动文件的有 `move_file` `copy_file` `trash`；另有 `delegate`（只读子 agent）和沙箱 bash。
- **没有永久删除**：`trash` 放进系统废纸篓，`AGENTRT_TRASH_DIR` 可改为指定目录；move / copy 不覆盖已有文件。
- **网页版**：每个改动先弹确认卡片，10 分钟不点视为拒绝；Esc 可以中断。
- **沙箱 bash**：只在专用工具做不到时用，跑在 macOS `sandbox-exec` 里。断网；只能写工作目录和临时目录；工作目录内禁止 unlink（删除只能走 `trash`）；读不到主目录里工作目录以外的内容。
  - 只读命令（白名单、无重定向和命令替换）自动放行，并换一份禁止任何写入的更严配置执行；其余命令要确认。
  - 没有 `sandbox-exec` 时 bash 不提供（失败即关闭）。
  - 覆盖已有文件内容（`echo > f`、`sed -i`）不在沙箱拦截范围内，所以非只读命令必须确认。

## MCP

```bash
python3 -m agentrt run "任务" --mcp-config <含 mcpServers 的配置文件> --mcp-only <server名>
```

- 只有工具声明了 `readOnlyHint`，并且配置里该 server 标了 `"trusted": true`，才免确认、可并行；注解由 server 自己声明，不可信的 server 不能凭它绕过确认。
- server 的 `env` 只传给该子进程，不进日志和 prompt。
- 文件助手不接 MCP。

## 测试

```bash
python3 -m pip install pytest
python3 -m pytest -q   # 不需要模型；macOS 沙箱测试只在 macOS 上跑，bwrap / Docker 测试没有对应环境时跳过
```

覆盖：
- 协议解析与错误回填；HTTP 重试 / 超长识别（假服务端）。
- 权限链各层、sandbox 感知的权限。
- 压缩，以及摘要调用的用量记账。
- 断点恢复（副作用至多一次、只读重跑、残留进程清理）。
- 真实 bwrap / Docker 隔离：越界写、断网、超时整组终止、容器丢失后重建。
- 编码工具语义、reasoning 回传与前缀改写检测。
- Harbor 桥接、评测 runner（并发 / 续跑 / 配对比较）、Harbor 结果汇总。
- 评测判定器自检。

## 未做

- 真实模型在 Terminal-Bench 2.0 上的成绩。
- Anthropic Messages 后端（thinking block 签名回放、显式 `cache_control`）；目前只有 OpenAI 兼容接口。
- 不改写日志的压缩：现在压缩写一条带完整新消息列表的 `compact` 事件。DeepSeek Harness 的做法是 `compaction/start → summary → end` 加锁事务 + `surfaceOp: replace`，原事件不动，模型视图是投影。
- resume 后重建 read-before-write 账本（目前要求模型重读）。
- 流式输出；snip / context collapse；MCP 的 HTTP 传输；持久 shell（环境变量跨调用保持）。

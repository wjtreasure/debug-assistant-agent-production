# Debug Assistant Agent

一个面向云运维和代码故障排查的只读 Agent Runtime。

项目让大模型通过受控工具调查故障现场，围绕运行状态、配置、日志、依赖和源码建立证据链，最终输出带证据引用的根因候选。模型负责语义判断，确定性 Runtime 负责工具权限、状态转移、预算和安全边界。

## Features

- **Evidence-grounded diagnosis**：区分 `Observation`、`Evidence` 和 `Hypothesis`，避免把模型猜测直接当成事实。
- **Typed tool calling**：使用结构化契约校验工具名、参数和输出，支持原生 Tool Calling 与有界 JSON fallback。
- **Deterministic Agent Runtime**：统一处理 Planner、Reflection、Review、重试、预算、超时、循环防护和终止。
- **Read-only by design**：只读访问仓库、故障快照和受限源码，不执行修复命令，不修改目标代码。
- **Auditable execution**：保留 `run.json`、`trace.jsonl` 和 Evidence provenance，支持回放与问题定位。
- **Separated evaluation**：运行时诊断、候选 Review 和运行后 Evaluator 相互隔离，避免诊断过程依赖标准答案。
- **Failure-aware engineering**：区分 Provider、Contract、Runtime、Budget、Convergence 和 semantic reasoning 等失败类型。

## Architecture

```text
Issue / Incident Snapshot
            │
            ▼
DiagnosisHarness
  ├─ Budget / Deadline / Loop Guard / Retry
  ├─ Capability Detection / Routing
  └─ Context Manager
            │
            ▼
Planner ──► Skill ──► Read-only Tool Registry
                              │
                              ▼
                    Observation Store
                              │
                              ▼
                       Evidence Memory
                              │
                              ▼
                 Hypothesis / Obligations / Contradictions
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Finalization Gate            Reflection
                 │
                 ▼
             Final Review
                 │
                 ▼
       Diagnosis Result + Trace + Metrics
```

### Core components

| Component | Responsibility |
|---|---|
| `DiagnosisHarness` | 管理一次诊断任务的生命周期、状态、预算、工具和终止条件 |
| `Planner` | 根据当前上下文和证据缺口选择下一步调查动作 |
| `Skill` | 表达服务调查、配置调查、日志调查、代码调查等调查意图 |
| `Tool Registry` | 暴露经过权限和参数校验的只读工具 |
| `ObservationStore` | 保存工具返回的完整原始观察 |
| `EvidenceMemory` | 将满足来源、范围和引用约束的观察晋升为可引用证据 |
| `ContextManager` | 根据假设、证据、矛盾和预算组装下一轮模型上下文 |
| `Finalization Gate` | 在提交候选前检查字段、证据、验证义务和矛盾 |
| `Final Review` | 审核候选与所引用证据是否自洽 |
| `Evaluator` | 在运行结束后独立计算诊断质量和运行指标 |

核心原则是：

> **LLM owns semantic reasoning; Runtime owns structure and control.**

模型可以提出调查动作和诊断解释，但不能绕过工具白名单、路径限制、Evidence 校验、预算或终止条件。

## Supported scenarios

当前主路径面向 Online Boutique / Cloud-OpsBench 风格的微服务故障，也保留通用代码仓库诊断能力。

典型调查对象包括：

- Service、Endpoint、Pod、Deployment 与容器端口配置；
- HTTP / gRPC 健康探针与服务协议；
- 服务依赖、路由、连接性和错误日志；
- 运行状态、配置和源码行为之间的因果关系；
- 代码仓库中的问题定位、源码读取和测试发现。

输出是带有组件、故障、机制、证据 ID、置信度和不确定性的诊断候选，不是自动生成或执行的代码补丁。

## Quick start

### Requirements

- Python 3.11+
- Git
- 一个支持 OpenAI-compatible Chat Completions API 的模型服务（真实模型运行时）

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e '.[dev,eval]'
```

也可以使用 Makefile：

```bash
make install
```

### 2. Configure the provider

复制配置模板：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下字段：

```dotenv
DEBUG_AGENT_PROVIDER=openai_compatible
DEBUG_AGENT_API_KEY=your-api-key
DEBUG_AGENT_BASE_URL=https://api.openai.com/v1
DEBUG_AGENT_MODEL=your-model-name
DEBUG_AGENT_CRITIC_MODEL=your-review-model-name
```

项目也支持不需要网络的 Mock Provider：

```bash
export DEBUG_AGENT_PROVIDER=mock
```

不要把 API Key、`.env` 或其他凭证提交到 Git。

### 3. Run a local smoke test

```bash
DEBUG_AGENT_PROVIDER=mock make smoke
```

该命令使用仓库内的示例问题和示例代码仓库，运行结果写入 `runs/smoke/`。

### 4. Diagnose a local repository

准备一个问题描述文件和目标仓库目录后执行：

```bash
debug-assistant diagnose \
  --issue examples/issues/example_issue.md \
  --repo examples/fixture_repo \
  --output runs/local-diagnosis \
  --task-id local-example
```

真实 Provider 运行时，先完成 `.env` 配置；使用 Mock Provider 时不需要 API Key。

### 5. Run a CloudOps incident snapshot

项目提供了基于快照的 Incident Diagnosis 入口：

```bash
DEBUG_AGENT_PROVIDER=mock python -m debug_assistant.incidents \
  --runtime-case tests/fixtures/cloudops/case_10/runtime \
  --output runs/cloudops-diagnosis \
  --topology data/online_boutique/service_topology.json
```

如果需要在运行结束后加载本地评测数据，可以额外传入：

```bash
  --evaluator tests/fixtures/cloudops/case_10/evaluator
```

Evaluator 只在诊断运行产物写入后加载，不会进入 Agent 的运行时上下文。

### 6. Inspect traces and run metrics

```bash
debug-assistant trace-metrics \
  --trace runs/cloudops-diagnosis/traces/<trace-file>.jsonl
```

一次运行通常会产生运行结果、JSONL Trace 和指标文件，可用于检查工具调用、Evidence、状态转移、预算和终止原因。

### 7. Run tests

```bash
pytest -q
```

或：

```bash
make test
```

## CLI overview

```bash
debug-assistant --help
```

主要命令：

| Command | Purpose |
|---|---|
| `diagnose` | 对本地问题描述和仓库执行只读诊断 |
| `diagnose-task` | 运行一个已准备好的任务目录 |
| `prepare-swe` | 准备代码仓库诊断任务 |
| `run-swe` | 批量运行已准备的任务 |
| `trace-metrics` | 汇总 JSONL Trace 指标 |
| `eval-localization` | 评估代码定位结果 |
| `eval-retrieval` | 评估检索结果 |

CloudOps 快照路径使用独立入口：

```bash
python -m debug_assistant.incidents --help
```

## Configuration

运行时参数通过环境变量配置，常用配置包括：

| Variable | Description | Default |
|---|---|---:|
| `DEBUG_AGENT_MAX_STEPS` | 最大决策步数 | `20` |
| `DEBUG_AGENT_MAX_TOOL_CALLS` | 最大工具调用数 | `45` |
| `DEBUG_AGENT_MAX_LLM_CALLS` | 最大模型调用数 | `40` |
| `DEBUG_AGENT_MAX_TOTAL_TOKENS` | 单任务 Token 预算 | `180000` |
| `DEBUG_AGENT_MAX_WALL_TIME_SECONDS` | 单任务最大运行时间 | `900` |
| `DEBUG_AGENT_MAX_CONTEXT_CHARS` | 单轮上下文字符预算 | `50000` |
| `DEBUG_AGENT_NATIVE_TOOL_CALLING` | 是否启用原生工具调用 | `1` |
| `DEBUG_AGENT_STRUCTURED_REFLECTION` | 是否启用结构化 Reflection | `1` |
| `DEBUG_AGENT_SEMANTIC_SEARCH` | 是否启用语义检索 | `1` in template |

完整默认配置见 `.env.example`。如果没有 Embedding Provider，关闭 `DEBUG_AGENT_SEMANTIC_SEARCH` 后仍可使用词法检索和基础诊断能力。

## Repository layout

```text
src/debug_assistant/
├── agent/          # Planner、Reflection、Review、Reporter
├── context/        # Context Manager 与上下文投影
├── harness/        # Budget、Deadline、Retry、Loop Guard、Trace
├── incidents/      # CloudOps Incident Runtime
├── memory/         # Observation、Evidence、Hypothesis 生命周期
├── tools/          # Repository / Snapshot / Source Tool Registry
├── evaluation/     # 运行后评测与 Trace 指标
├── knowledge/      # 可选的领域知识与检索能力
└── repository/     # 仓库索引、搜索和安全文件访问

tests/              # 单元测试、契约测试和 Runtime 测试
examples/           # 本地运行示例
data/               # 本地数据和快照资源
```

## Safety model

- 模型输出先经过结构化解析和 Pydantic contract 校验；
- 工具必须存在于 Registry 且参数满足对应 Schema；
- 仓库访问受路径边界约束，默认只读；
- Tool、LLM、Token、时间和重复动作都有上限；
- Provider 超时、契约错误和工具失败会进入有界重试或明确失败；
- 原始 Observation、可引用 Evidence 和最终 Candidate 分层保存；
- Gold / Evaluator 数据不参与 Agent 运行时决策。

## Development

```bash
# 安装开发依赖
make install

# 运行测试
make test

# 运行示例 smoke test
make smoke

# 清理本地运行产物
make clean
```

项目适合用于研究 Agent Runtime、上下文工程、证据链、工具调用契约和可审计故障诊断。

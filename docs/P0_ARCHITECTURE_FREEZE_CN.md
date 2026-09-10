# P0 架构冻结与迁移审计

审计日期：2026-09-11  
审计对象：当前工作区 `debug-assistant-agent-production`，`main` 分支，HEAD `2d55c6c`
审计性质：只读代码审计与设计冻结。本轮没有调用真实 LLM、Embedding、Kubernetes 或其他外部服务，也没有修改运行时代码。

## 0. 冻结结论

P0 的目标架构和统一 Benchmark 协议可以冻结；当前代码作为迁移基线接受，但不应被误认为已经实现了 P1-P5。

冻结以下边界：

1. 产品核心是 CloudOps / 微服务 Incident Diagnosis Runtime；代码调查是其中的 Code Investigation 子系统。
2. `DiagnosisHarness` 是 Incident 核心运行时；`AgentHarness` 保留为 SWE-bench Code Investigation Benchmark Adapter，不属于 Incident 核心主链。两者在物理上暂不合并，避免一次性迁移已验证的 SWE 状态机。
3. Incident 核心只保留三个 Agent 角色：Planner、Reflection、Diagnosis Review。Knowledge/RAG/Memory 是能力层，不新增 Knowledge Agent。
4. LLM 只产生语义决策和候选动作；Runtime 独占工具权限、Schema、Evidence ID、预算、Deadline、Retry、Duplicate Guard、Circuit Breaker、Finalization、Fail-closed、INCONCLUSIVE 和 Trace。
5. `KnowledgeCandidate`/`PriorContext` 与当前 Incident `Observation`/`Evidence` 分属不同命名空间。历史 Incident、RAG 文档和 KG 结果不能直接进入正式 Evidence 或 Root Cause 引用集合。
6. `Code Search`、`Symbol Search`、AST Search 的结果永远是 Candidate Retrieval；只有安全边界内的 `read_file` 真实源码观察才能生成 CODE Evidence。
7. FastAPI/Queue/Worker、Event Sourcing、多租户/RBAC、多模态、在线 Kubernetes、Model Router、Remediation、LangGraph Adapter、Runtime Evidence Graph 均列为 Optional Extensions，本阶段不进入核心架构。

因此，P0 的状态是：**目标架构已冻结，迁移实现未开始；存在若干必须在 P1 修正的边界缺口。**

## 1. 审计证据与验证结果

审计以当前源码、测试和已登记快照为准，不以旧 README 或历史迭代文档为准。主要检查入口、状态所有权、Evidence 生命周期、工具注册、索引/检索、评测和数据隔离。

验证结果：

- `conda run -n ai-debug-agent python -m pytest`：**374 passed**。
- `conda run -n ai-debug-agent python -m compileall -q src tests`：通过。
- `git diff --check`：通过。
- `scripts/smoke_cloudops_snapshot.py`：4 个 manifest 案例通过；快照 miss 保持 `UNAVAILABLE` 且 `semantic_negative=false`。
- 全过程没有启动真实 Provider 或 Embedding 请求。

当前工作区已有大量已暂存的迭代改动；本审计不回退、不重排这些改动。

## 2. 当前真实架构

### 2.1 两个入口、一个 Incident 核心、一个 SWE 适配器

```text
CloudOps Incident
  incidents/__main__.py
    -> CloudOpsRuntimeCaseLoader             runtime-only input
    -> DiagnosisHarness.run
    -> CloudOpsSnapshotToolRegistry
    -> NativeToolPlanner                      Planner + native tool calls
    -> Tool validation / duplicate / budget / deadline / retry / circuit
    -> ObservationStore -> EvidenceMemory -> ContextManager
    -> IncidentHypothesis + obligations/contradictions
    -> one candidate finalization boundary
    -> FinalReviewAgent                      tool-less Review
    -> PASS / INCONCLUSIVE / FAILED

SWE Code Investigation
  cli.py
    -> AgentHarness.run(TaskSpec)
    -> RepositoryIndex / ContextManager / ToolOrchestrator
    -> generic Reflection / SemanticReducer / Convergence / Reporter
    -> localization and trace metrics
```

Incident 入口在 `src/debug_assistant/incidents/__main__.py`，核心循环在 `src/debug_assistant/incidents/runtime.py`。SWE 入口在 `src/debug_assistant/cli.py`，通用循环在 `src/debug_assistant/harness/runtime.py`。

### 2.2 已有能力与目标的对应关系

| 能力 | 当前实现 | P0 判断 |
|---|---|---|
| Incident 输入隔离 | `CloudOpsRuntimeCaseLoader` 递归拒绝 evaluator-only 字段 | 保留 |
| Planner | Incident 使用 `NativeToolPlanner`；每个工具调用携带 Skill 与假设完成字段 | 保留并作为唯一调查 Agent |
| Reflection | Incident 使用无工具 `IncidentReflectionAgent`，按受控信号触发 | 保留；只能反馈，不能直接写 Hypothesis |
| Review | `FinalReviewAgent` 只接收 Candidate 与其引用 Evidence | 保留并冻结为 Diagnosis Review |
| 工具控制 | Pydantic schema、只读工具、路径隔离、重试、重复检测、Circuit Breaker | 保留；后续抽成共享 Runtime Port |
| 证据链 | `ToolObservation`（`obs-*`）→ `ObservationStore` → `EvidenceMemory`（`ev-*`） | 保留为唯一事实链 |
| Hypothesis | Incident 自有 `IncidentHypothesis`、Verification Obligation、Contradiction | 保留；补充统一 Evidence 投影 |
| Context | `ContextManager`、Incident/Code Projection、budget packing、rehydration | 保留并接收 `PriorContext` 的隔离投影 |
| Code Retrieval | FTS5/BM25、Python AST symbol/call index、可选 Dense、RRF、AST refinement | 保留为 Code Investigation/RAG substrate，不重写 |
| Domain RAG | 没有 Domain document store、parent-child retrieval 或 reranker | P2 新增 |
| Incident Memory | 只有任务内 Observation/Evidence memory，没有历史 Incident schema/storage/retrieval | P1 新增 |
| Static KG | 有 topology JSON 读取工具，但没有统一实体/关系图、版本和 provenance | P3 新增 |
| Capability Detection | 目前由 `case.evidence_sources`、source binding 和工具是否存在隐式表达 | P4 显式化 |
| Hybrid Router | 当前只有 `code_search` 的 lexical/semantic/hybrid/hybrid_ast 检索模式 | P4 新增 Incident 入口 Router；不要与检索引擎混名 |
| Benchmark | Incident evaluator、SWE localization、Retrieval evaluator 分散存在 | P5 统一协议与聚合 |

## 3. 保留、修改、合并、废弃决策

| 现有模块/概念 | 决策 | 迁移边界 |
|---|---|---|
| `incidents/runtime.py::DiagnosisHarness` | 保留为 Incident canonical runtime | 后续只在其外接 Capability/Knowledge ports，不另起平行 Incident loop |
| `harness/runtime.py::AgentHarness` | 保留为 SWE Adapter | 不把 SWE 专有 SemanticReducer/Reporter 生命周期强行塞入 Incident 主链 |
| `IncidentHarness` / `IncidentHarnessConfig` alias | 兼容保留，标记废弃 | P4 迁移完调用方后移除，不再新增使用 |
| `NativeToolPlanner` | 保留并修改 | 增加统一 Runtime capability/context 输入；仍是唯一拥有调查 Tool 权限的 Agent |
| `IncidentReflectionAgent` | 保留并修改 | 输入统一 Snapshot/Prior Context；禁止 Tool 和 Hypothesis 直接变更 |
| `FinalReviewAgent` | 保留并修改 | 接收完整、可追溯的 Evidence 投影；不接触 Ground Truth、Prior 原文或工具 |
| `Reflector`、`TypedReflection`、generic `Reporter` | 保留在 SWE Adapter | 不复制成 Incident 的第二套 Review/Reflection 语义 |
| `ObservationStore`、`EvidenceMemory` | 保留并扩展 | 继续维护 `obs-*`/`ev-*`；新增 Knowledge 不能绕过可信边界 |
| `ContextManager`、Projection、Packing | 保留并扩展 | 新增 `PriorContext` 独立区段和 token budget，不把 Prior 伪装成 Evidence |
| `CloudOpsSnapshotToolRegistry` | 保留 | 作为 Snapshot/CloudOps capability adapter；不在其中实现 Router 或答案规则 |
| `CloudOpsSourceToolRegistry` | 保留并修改 | 通过统一 Code Retrieval port 提供候选；`read_file` 才能建立 CODE Evidence |
| `RepositoryIndex`、`chunks.py`、`SemanticIndex`、`search_engine.py` | 保留并包装 | 复用于 Code Investigation 与 Domain RAG；不在 P0 重写底层索引 |
| `EmbeddingCache`、`SiliconFlowEmbeddingProvider` | 保留为可选基础设施 | 默认不调用；真实 BGE-M3 只在 P2/P5 明确 opt-in 的实验中启用 |
| `evaluation/incident.py`、`localization.py`、`retrieval.py` | 保留 | 作为各 Benchmark Adapter；P5 统一 envelope/metrics，不混用 Ground Truth |
| 新增 Knowledge Agent | 废弃该方向 | Knowledge/RAG/Memory/KG 全部是能力层与高层 Tool，不是 Agent |
| Runtime Evidence Graph | 本阶段废弃/延后 | P3 只做 Static KG；当前 Incident Evidence 仍由 EvidenceMemory 管理 |

“合并”只发生在接口和确定性控制面：预算、Deadline、Retry、Trace、Evidence ID 和 finalization contract 使用统一语义。物理运行时循环在 P0-P4 不做大拆大并，减少回归面。

## 4. 冻结后的 Incident 主链与权限

```text
Incident
  -> Capability Detection
  -> Entity Extraction
  -> Hybrid Router
  -> optional Knowledge Pre-Retrieval
  -> Planner
  -> Skill / Tool Investigation
  -> Observation
  -> Evidence
  -> Hypothesis
  -> Verification Obligation / Contradiction
  -> Reflection
  -> Finalization Gate
  -> Diagnosis Review
  -> Root Cause / INCONCLUSIVE
```

权限冻结如下：

| 角色 | 允许 | 禁止 |
|---|---|---|
| Planner | 选择 Skill/Tool、更新假设提议、提出 Evidence Gap、提议 finalize | 绕过 schema/budget、直接写 Evidence/Trace、把 Prior 当事实 |
| Reflection | 判断 contradiction、no-progress、instability、premature finalize、Review reject，并返回反馈 | Tool、Ground Truth、直接修改 Hypothesis |
| Diagnosis Review | 审核 Candidate、引用 Evidence、grounding、unsupported claims、causal coherence | Tool、Ground Truth、Prior 作为证明、替换 Candidate |
| Runtime | 注册/校验/执行 Tool，分配 ID，维护状态，决定 terminal status | 解释自然语言因果、替 LLM 选择 Root Cause |
| Knowledge Layer | 按 Query 召回 Candidate，返回 score/provenance | 生成正式 Evidence、关闭 Obligation、决定 Root Cause |

### 4.1 统一语义隔离

- `KnowledgeCandidate` 只有候选和先验语义，不产生 `ev-*`。
- `PriorContext` 只能作为 Planner 的辅助上下文；Context、Trace、Review 均必须显式标注 `source_type=prior`。
- 当前事故的 Log/Trace/Kubernetes/Config/Runtime/verified source code 必须经过 Observation→Evidence 后，才可被 Hypothesis 引用。
- Search/Symbol/AST 的结果可产生路径、symbol、range 和 score，但不能直接证明代码行为。
- `INCONCLUSIVE` 是正式结果，不等同于 FAILED；缺证据、冲突未解、能力不可用、Review 不通过或预算耗尽均应保留原因。

### 4.2 P1 冻结的最小统一 Schema

`KnowledgeCandidate` 的字段冻结为：

```text
candidate_id
source_type              domain_rag | incident_memory | static_kg
content
parent_id                nullable
service                  nullable
module                   nullable
fault_type               nullable
software_version         nullable
repo_commit              nullable
timestamp                nullable
retrieval_score          nullable
rerank_score             nullable
confidence               nullable
provenance               required structured source/version/time record
```

统一 Retrieval 接口的最小输入输出冻结为：

```text
KnowledgeQuery:
  incident_id, query_text, service?, module?, fault_type?,
  software_version?, repo_commit?, filters, top_k, token_budget

KnowledgeRetrievalResult:
  candidates: list[KnowledgeCandidate]
  diagnostics: requested_sources, effective_sources, degraded, reason,
               per_source_counts, latency, token_estimate
```

最终进入 Planner 的不是裸候选列表，而是：

```text
PriorContext:
  context_id, candidates, source_types, retrieval_query,
  provenance, prior_weight, packed_chars/tokens
```

P1 应把现有 `Evidence` 的 `excerpt`、`file`、line range、`raw_observation_id`、truncation 和 tags 以统一、可审计的投影提供给 Review。目前 Incident 的 `IncidentEvidence` 只保留 `evidence_id/source/target/summary/observation_id`，这是明确的 Schema 缺口，不应再新建第二个事实存储解决。

## 5. 当前 Retrieval 资产审计

### 5.1 可以直接复用的部分

- `RepositoryIndex`：SQLite FTS5/BM25 文件级 lexical index，以及 Python AST symbol/call-site index。
- `chunks.py`：Python AST symbol chunk 和非 Python/text window chunk；具备 content hash、embedding text hash、chunker version、line range。
- `SemanticIndex`：任务级 Dense matrix，支持 FAISS 或 NumPy，all-or-nothing build，维度校验和 content-addressed cache key。
- `RepositorySearchEngine`：lexical、semantic、hybrid、hybrid_ast；RRF 默认 `k=60`，语义结果在文件粒度融合。
- `CloudOpsSourceToolRegistry`：已经把源代码绑定限制在 case-local `code/`，并复用安全路径和只读工具。
- `EvidenceMemory`：已经拒绝 `information_source=candidate_retrieval`，并拒绝 `symbol_search` 直接变成 Evidence。

### 5.2 尚未具备的目标能力

- `CodeChunk.parent_symbol` 目前是元数据，不是可查询的 Parent-Child Retrieval。
- 没有 Domain document ingestion、structure-aware document chunking、BM25/Dense/RRF 统一的 KnowledgeCandidate 输出。
- 没有 Local Cross-Encoder Reranker、source diversity、token-budget packing 的 Knowledge 层实现；当前 Context packing 不能替代它们。
- Dense/BGE-M3 不是 Incident 默认生产路径。Incident CLI 没有从 `AppConfig.semantic_search` 构造 Provider/Index 并注入 `DiagnosisHarness`；当前 Dense 集成主要存在于 generic SWE runtime、retrieval evaluator 和确定性注入测试。
- Incident 的 `source_evidence_ids()` 仍把 `symbol_search` 列入候选集合，虽然 `EvidenceMemory` 当前会拒绝 candidate retrieval，因此通常不会形成实际错误。这是两个边界判断没有使用同一 predicate 的一致性缺口，P1 必须统一为“成功的 bounded `read_file` source_read”并补回归测试。

结论：现有 Code Retrieval 可以作为 P2 的底座，但不能宣称已经是目标 Knowledge RAG；当前缺口不应通过在 `cloudops_snapshot.py` 中增加答案规则来填补。

## 6. 统一 Benchmark 协议冻结

### 6.1 通用 Run Envelope

所有 Benchmark Adapter 都输出相同外层记录：

```text
run_id
case_id
benchmark                  code_investigation | knowledge_retrieval |
                           knowledge_ablation | runtime_reliability |
                           context | cloudops_e2e
split                      dev | test | fixed_fixture
snapshot_id
runtime_version
knowledge_policy           none | rag | incident_memory | static_kg | full
model_id                   nullable; mock/offline runs must say so
seed                       nullable but fixed when sampling
status                     PASS | INCONCLUSIVE | FAILED
metrics                    benchmark-specific structured values
trace_path
error_type / error_message
```

Ground Truth、qrels 和 key evidence 只能出现在 evaluator side。Agent runtime 的输入、Prompt、Tool cache、PriorContext 和 Trace 的 model-visible 区域必须经过 leakage check。

### 6.2 六类 Benchmark

| Benchmark | 输入 | 主要指标 | 规则 |
|---|---|---|---|
| Code Investigation | Issue + 固定 repository snapshot | File Hit@1/3、File MRR、Symbol Hit、source verification rate、candidate→read_file rate | SWE-bench gold patch 只离线派生定位标签；不评 patch、不执行自动修复 |
| Knowledge Retrieval | Query + 固定 corpus/version/qrels | Recall@K、MRR、nDCG@K、source diversity、dedup rate、latency/token cost | 分别报告 Domain RAG、Incident Memory、Static KG、fusion；Knowledge 结果不是 Evidence |
| Knowledge Ablation | 同一 incident/corpus/snapshot 的固定变体 | Full 与 `none`/单路/双路的质量、成本、grounding delta | 固定模型、seed、预算；只改变 knowledge policy |
| Runtime Reliability | mock Provider/Tool fault matrix | schema recovery、invalid route、duplicate block、retry、circuit open、deadline、fail-closed、INCONCLUSIVE rate | 不调用真实外部 API；故障由 deterministic doubles 注入 |
| Context | 同一轨迹的 context budget/eviction/rehydration 变体 | prompt tokens/chars、evidence retention、prior retention、budget violation、rehydration、unsupported claim rate | 不能以更长 Context 换取 Evidence 语义升级 |
| CloudOps E2E | runtime-only CloudOps fixture；少量真实 Provider opt-in | component/fault/mechanism accuracy、key evidence coverage、Review PASS、INCONCLUSIVE、tool/LLM cost、wall time | 真实 E2E 与离线回归分开；禁止用少量真实运行结果替代全量 deterministic suite |

### 6.3 Split 与冻结纪律

- SWE-bench Lite：dev 23 只用于调参和 bad-case；test 300 在配置冻结后只报告最终指标。
- CloudOps：先使用固定、runtime-only fixture；Evaluator 目录必须在 Diagnosis/Review 完成后才加载。
- Retrieval：固定 task manifest、repository commit、chunker version、embedding model/dimension、qrels、top-k 和 token budget。
- Ablation：`none`、`domain_rag`、`incident_memory`、`static_kg`、`full` 五个最小 policy；不在一次 ablation 同时改变 Planner prompt、预算或工具权限。
- 真实 LLM/Embedding：只允许通过显式 opt-in 命令；P0、P1 和默认测试不得触发网络。

### 6.4 FINAL_METRICS 最小字段

P5 的聚合文件必须至少包含：

```text
run_count, pass_count, inconclusive_count, failed_count
root_cause_accuracy, component_accuracy, fault_accuracy, mechanism_accuracy
evidence_grounding_rate, key_evidence_coverage
code_file_hit_at_1, code_file_hit_at_3, code_file_mrr, code_symbol_hit
knowledge_recall_at_k, knowledge_mrr, knowledge_ndcg_at_k
knowledge_policy_deltas
invalid_route_rate, route_recovery_rate, duplicate_block_rate
tool_retry_rate, circuit_open_rate, deadline_rate, contract_failure_rate
mean/p95_steps, mean/p95_tool_calls, mean/p95_llm_calls, mean/p95_tokens
context_budget_violation_rate, evidence_retention_rate, prior_contamination_rate
review_pass_rate, inconclusive_rate, failed_rate
```

每个 aggregate 必须同时输出 `available_n`、`unavailable_n`、`failed_n` 和比较分母，不能把 Provider 不可用误报成召回为 0，也不能把 INCONCLUSIVE 误报成 PASS。

## 7. P1-P5 最小侵入式实施计划

### P1 — Knowledge & Memory Foundation

新增 `KnowledgeCandidate`、`KnowledgeQuery`、`KnowledgeRetrievalResult`、`PriorContext` 和 `IncidentMemoryRecord` schema；新增 Storage/Retrieval port、Temporary/Verified Memory 状态与 provenance。先提供 in-memory/SQLite deterministic adapter，不接真实 Provider。统一 Evidence projection 和 source-read predicate。

退出条件：Knowledge 结果无法进入 `EvidenceMemory`；历史 Incident 只有 Prior 标记；schema、provenance、version filter、Temporary→Verified transition 有离线测试。

### P2 — Domain RAG

将现有 lexical/Dense/RRF/Chunk manifest 包装为 Domain RAG adapter：structure-aware chunk、parent-child link、BM25、BGE-M3 optional dense、RRF、local reranker、dedup、source diversity、token packing。复用底层 index/embedding/cache，不新建平行 Code Search。

退出条件：fake embedding 与 deterministic reranker 可完整测试；真实 Embedding 仍显式 opt-in；RAG 产出只进入 `PriorContext`。

### P3 — Static Knowledge Graph

从 Kubernetes YAML、topology、Trace、Repository AST、Git metadata 做 deterministic build。先实现 Service、Deployment、API、Module、Repository、File、Symbol、FailureMode 及关系的最小子集。每条关系保存 source/version/timestamp/confidence/provenance；只暴露 `get_neighbors`、`find_dependency_path`、`get_failure_modes`、`get_related_code` 等高层 Graph Tool。

退出条件：Planner 不接触 Cypher；图查询输出统一 `KnowledgeCandidate`/Prior 语义；不引入 Runtime Evidence Graph。

### P4 — Hybrid Router + On-demand Retrieval

增加确定性的 Capability Detection、Entity Extraction、Hybrid Router 和 Query Builder。Router 只决定初始调查方向、优先级和是否做知识增强；简单、已有明确 Evidence 的 Incident 可以跳过 Pre-Retrieval。Planner 仍可基于 Hypothesis/Evidence Gap/Verification Obligation 主动召回 Knowledge。

退出条件：每次 retrieval 都有 diagnostics；Router 不输出 Root Cause、Skill 锁定或 Evidence；Prior 与 Evidence 在 Context、Trace、Review 中都可区分。

### P5 — Unified Benchmark

在不重写两条已验证执行路径的前提下，为 Code Investigation、Knowledge Retrieval、Ablation、Runtime Reliability、Context 和 CloudOps E2E 提供统一 Run Envelope、固定 manifest、离线 runner 和 `FINAL_METRICS`。只加入少量真实 LLM CloudOps E2E，不把真实服务接入核心 Runtime。

退出条件：同一个 `case_id/snapshot_id/runtime_version` 能在各 adapter 中追踪；所有分母、不可用、失败和 INCONCLUSIVE 状态可审计；无 evaluator leakage。

## 8. P0 未完成但已登记的迁移门槛

这些不是本轮实现项，但在进入对应阶段前必须关闭：

1. Incident CLI 没有把配置中的 semantic provider 接入 Incident `DiagnosisHarness`；在此之前不能声称 CloudOps 已具备 BGE-M3 Hybrid Retrieval。
2. `IncidentEvidence` 的 Review 投影丢失完整 source range/provenance；P1 必须扩展现有 Evidence projection，不能再创建副本。
3. `source_evidence_ids()` 与 `EvidenceMemory` 的候选过滤 predicate 不一致；P1 必须统一为 read-file source verification。
4. 当前没有 Capability Detection、Incident Hybrid Router、Knowledge Pre-Retrieval；这些是 P4 的明确新增边界，不应在 P1/P2 偷塞进 Planner prompt。
5. 当前没有 Incident Memory、Domain RAG、Static KG 的统一接口和 `KnowledgeCandidate`；P1 先冻结 schema/port，P2/P3 再实现后端。
6. 现有架构文档仍包含旧的 V1/Task A/B/C 表述，且部分测试数量写为 334；本文件是本阶段冻结基线，后续文档同步只能做小范围一致性修订，不改变上述边界。

P0 审计到此结束。下一次实现工作应从 P1 的 Schema/Port/隔离测试开始，不得从 RAG、KG 或真实 API E2E 直接开工。

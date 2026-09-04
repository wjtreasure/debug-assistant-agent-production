# Debug Assistant Agent

debug-assistant-agent is a production-oriented, read-only repository diagnosis
and localization agent. It inspects issue text, source files, tests, git history,
repository indexes, and tool observations, then produces an evidence-backed
diagnosis. It does not edit repositories, generate patches, or claim that a
code change was applied.

The current runtime version is **1.5.2 — Typed Agent Runtime / Production Hardening**.

## ReAct control loop

The main investigation loop follows the ReAct pattern:

**Reason → Act → Observe → Reflect → Reason**

`Planner` performs the reasoning and chooses the next tool intent. The
`ToolOrchestrator` validates and executes the action (Act). Tool results,
source coverage, and evidence records provide environmental feedback (Observe).
`Reflection` and `SemanticReducer` review that feedback, update the derived
internal state, and select the conditions for the next reasoning round (Reflect
→ Reason).

This is a dynamic loop, not a fixed sequence of repository steps. The harness
may route, retry, expand, parallelize, or reject an action based on the current
state and budget. The number of iterations is determined by deterministic
termination and safety conditions described below.

## Architecture

The runtime separates model proposals from deterministic execution and semantic
state management:

~~~text
CLI
  |
  v
AppConfig + AgentHarness
  |
  +-- RunDeadline / BudgetController / provider health
  +-- SafeRepositoryFS + path resolver + repository index/chunks
  +-- lexical search + optional semantic embeddings + hybrid ranking
  |
  v
Planner capability negotiation
  |
  +-- native typed tool calls, when the provider supports them
  +-- bounded structured-JSON planner fallback otherwise
  |
  v
ToolRegistry + ToolOrchestrator
  |
  +-- Pydantic argument validation
  +-- read-range expansion/splitting with provenance
  +-- bounded parallel read-only actions
  +-- serial execution, retry, route and loop guards
  |
  v
ObservationStore + EvidenceMemory + ReadCoverageIndex
  |
  +-- information needs and evidence obligations
  +-- context catalog, lifecycle and budget packing
  +-- source/file/symbol/range compatibility checks
  |
  v
Typed Reflection -> SemanticReducer -> semantic invariants
  |
  +-- derive evidence sufficiency and terminal status from committed evidence
  +-- prevent satisfied/superseded obligations from being reopened
  +-- split a new obligation when the same source needs a different goal type
  |
  v
Bounded Reporter -> final report + trace + localization metrics
~~~

The LLM proposes actions and semantic input. The harness validates tool
arguments, enforces the read-only boundary, records observations and evidence,
and commits semantic state only after invariant validation. Reflection output
cannot directly overwrite the derived hypothesis status.

### Main runtime components

- AgentHarness owns the per-task lifecycle, shared cooperative deadline,
  budgets, retries, convergence, trace recording, and finalization.
- RepositoryIndex, SafeRepositoryFS, and the search engine provide bounded
  repository discovery. Search results are discovery evidence; relevant hits
  must be followed by a bounded source read before they become source evidence.
- ToolRegistry exposes only read-only tools such as tree listing, grep, symbol
  lookup, source reads, git history, and test discovery.
- ToolOrchestrator turns typed requests into validated, bounded execution plans.
  Short source reads may be widened for context, while the original requested
  range is retained in metadata.
- EvidenceObligationTracker and ReadCoverageIndex preserve provenance and
  distinguish location obligations from semantic behavior/causality obligations.
  Terminal obligation goal types are immutable.
- TypedReflection, SemanticReducer, and semantic_invariants form the semantic
  transaction boundary. Malformed individual review rows are isolated without
  discarding valid reflection fields.
- Reporter applies context and evidence limits. A deterministic fallback
  reporter is used when provider output is unavailable or violates its contract.

## Design principle: semantic LLM, structural harness

The central boundary is: **the LLM owns semantic interpretation; the harness
owns structure and state transitions**.

- The LLM supplies reasoning, diagnosis hypotheses, and tool-selection intent;
  it does not directly mutate runtime state or repository files.
- Every tool request is checked against its typed Pydantic argument model before
  execution.
- `SemanticReducer` derives hypothesis status, evidence sufficiency, and
  required gaps from committed evidence. The invariant validator checks the
  resulting semantic transaction.
- Invalid model output is isolated through bounded contract repair,
  sanitization, or deterministic fallback. It is never allowed to pollute the
  committed semantic state.

## Deterministic termination conditions

The harness, rather than the LLM, decides when the ReAct loop can stop:

- **Active success:** finalization is allowed only when the derived hypothesis
  is `supported` or `confirmed`, has supporting evidence, has
  `evidence_sufficient=true`, has no contradiction, and has no required gaps.
- **Semantic convergence:** when `semantic_no_progress_streak` reaches its
  configured threshold after successful semantic transactions, the harness
  stops spending effort on unchanged state and moves to conservative
  finalization. This is not a claim that the diagnosis is supported; the final
  report preserves any remaining uncertainty.
- **Resource budget:** exploration is bounded by `max_steps`,
  `max_llm_calls`, `max_tool_calls`, `max_total_tokens`, and
  `max_wall_time_seconds`. The shared wall-clock deadline also reserves time
  for cleanup and reporting.
- **Failure degradation:** repeated Reflection failures reach the configured
  `max_consecutive_reflection_failures` limit; repeated invalid Planner
  contracts reach the configured Planner contract failure limit. The runtime
  then stops retrying that path and either finalizes from available evidence or
  returns a bounded failure, according to the current evidence and stage.

These are deterministic Harness rules. A model response cannot increase a
budget, reopen a terminal obligation, bypass an invariant, or force the runtime
to continue after a terminal condition.

## Design decision: one agent with a deterministic harness

The project deliberately uses one ReAct agent coordinated by a deterministic
harness instead of LangGraph, AutoGen, or a multi-agent topology. Repository
debugging is primarily a depth-first evidence and causal reasoning chain; the
individual tool calls are bounded and only selected safe reads may run in
parallel. Multiple agents would introduce more state reconciliation and make
the evidence trail and failure diagnosis harder to audit. Keeping planning,
execution, semantic reduction, and termination under one typed harness makes
reliability behavior easier to test and reproduce.

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,eval]'
cp .env.example .env
~~~

Set the minimum provider configuration in `.env` or the shell environment:

~~~bash
export DEBUG_AGENT_API_KEY=...
export DEBUG_AGENT_BASE_URL=https://api.openai.com/v1
export DEBUG_AGENT_MODEL=gpt-5.6
~~~

Keep `.env` local and do not commit credentials. This repository does not
include SWE-bench data or task workspaces. Download and prepare a compatible
dataset yourself, then process it through the CLI.

## CLI usage

Use `debug-assistant --help` as the authoritative entry point for all commands
and options. For example, a prepared task directory can be used for batch work:

~~~bash
debug-assistant run-swe --tasks <task_directory> --output <output_dir>
~~~

## Configuration and runtime guarantees

Runtime settings are controlled by `DEBUG_AGENT_*` environment variables. The
main limits include bounded steps, tool calls, LLM calls, token accounting, and
a shared per-task wall-clock deadline. Native typed tool calling and structured
reflection are enabled by default; providers without native tool support use
the bounded structured-JSON fallback. Semantic search is optional and requires
an embedding provider; lexical search remains available without one.

The runtime is read-only by construction: repository paths are scoped to the
task workspace, git operations are read-only, and test discovery does not
install packages or mutate the repository. Sensitive provider values are
redacted from runtime artifacts.

## Traces and metrics

Tasks emit JSONL traces containing lifecycle, tool, evidence, obligation,
reflection, budget, and finalization events. The evaluation tools report
localization quality such as File Hit@k, MRR, symbol/range hits, and prediction
coverage, alongside runtime health indicators such as fallbacks, route
rejections, partial results, and forced finalization.

## Versioned experiments

Experiment outputs are managed in versioned directories. Single-task and batch
experiments can both be run through the CLI; use `debug-assistant --help` for
the supported command and argument format.

## Development checks

~~~bash
python -m pytest -q
python -m compileall -q src tests
git diff --check
~~~

# Debug-Assistant Agent

A production-oriented **read-only debugging agent** for issue diagnosis and repository localization. It is intentionally not a coding agent: it may inspect source, git history and tests, but it cannot edit files or produce/apply patches.

## Why this project exists

The original business came from Jira defect triage: engineers had to combine issue text, logs and repository context to locate likely root causes. The interview version keeps the enterprise boundary (AI cannot modify production code), replaces Jira-private data with SWE-bench Lite, and upgrades a fixed workflow into an Agent + Harness architecture.

## Core architecture

```text
Issue / SWE-bench task
        |
        v
+---------------- Agent Harness ----------------+
| Workspace lifecycle / permissions / budgets   |
| Context + evidence memory / trace / loop guard |
| Router validation / retry / reflection         |
+--------------------+---------------------------+
                     |
                     v
             LLM Planner / Critic
                     |
             chooses a Skill intent
                     |
                     v
  +---------------- Skills ----------------+
  | triage | explore | hypothesize         |
  | validate | impact | synthesize         |
  +----------------+-----------------------+
                   |
                   v
  +---------------- Tools -----------------+
  | tree | grep | symbol | read | git      |
  | test-discovery (read-only)             |
  +----------------------------------------+
                   |
                   v
        evidence-backed diagnosis report
```

The **LLM proposes** the next action. The **Harness decides whether that action is legal and useful**. This is the key distinction from the old hard-coded workflow.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,eval]'
cp .env.example .env
```

Configure an OpenAI-compatible endpoint (works with many gateways/providers):

```bash
export DEBUG_AGENT_API_KEY=...
export DEBUG_AGENT_BASE_URL=https://api.openai.com/v1
export DEBUG_AGENT_MODEL=gpt-5.6
```

Run against a local repository snapshot:

```bash
debug-assistant diagnose \
  --issue examples/issues/example_issue.md \
  --repo /path/to/repository \
  --output runs/example
```

Prepare SWE-bench Lite from a parquet file:

```bash
debug-assistant prepare-swe \
  --parquet data/dev-00000-of-00001.parquet \
  --output data/swe_lite_dev
```

Run one prepared task:

```bash
debug-assistant diagnose-task \
  --task data/swe_lite_dev/pvlib__pvlib-python-1072 \
  --output runs/pvlib-1072
```

Batch inference + evaluation:

```bash
debug-assistant run-swe --tasks data/swe_lite_dev --output runs/dev
```


```bash
debug-assistant eval-localization \
  --gold data/swe_lite_dev \
  --predictions runs/predictions.jsonl \
  --output runs/metrics.json
```

## Enterprise safety boundary

The permission system is **deny-by-default** for write operations. There is no edit/write/patch tool in the registry. Repository paths are normalized and restricted to the task workspace. Git commands are read-only. The optional test-discovery tool never installs packages or mutates the repository.

## What to measure

Because this assistant does not generate patches, the primary metrics are diagnosis/localization metrics rather than SWE-bench resolved rate:

- File Hit@1 / Hit@3 / MRR
- Function Hit@k (when the gold function can be extracted)
- Evidence precision / evidence-grounded conclusion rate
- Tool routing invalid rate and recovery rate
- Repeated-call / loop rate
- Mean tool calls, steps and tokens per task
- Reflection intervention rate
- Bad-case taxonomy distribution
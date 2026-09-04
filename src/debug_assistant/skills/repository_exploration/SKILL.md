---
name: repository_exploration
description: Map an information need to likely repository locations using adaptive exact, lexical, semantic, or hybrid retrieval.
---
# Repository Exploration
Choose retrieval by the current information need, not by a fixed workflow.
- Exact identifier/error/config key: prefer symbol_search, grep, or code_search(mode="lexical").
- Behavioral/conceptual description without identifiers: prefer code_search(mode="semantic").
- Mixed vocabulary, uncertainty, or weak lexical results: prefer code_search(mode="hybrid").
Semantic results are candidates only. Read the source before treating a candidate as evidence.
Avoid repeated equivalent queries. If repeated lexical attempts yield no new evidence, change retrieval strategy rather than merely rewording the same search.

## Completion Criteria
- at least one plausible source location is grounded
- the next information need concerns implementation/mechanism rather than broad repository discovery

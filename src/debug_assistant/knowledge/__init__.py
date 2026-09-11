"""Typed Knowledge and Incident Memory capability layer.

Knowledge is intentionally not an Agent and is kept separate from the
run-scoped Observation -> Evidence ledger.  Adapters in this package return
prior material only; they never allocate ``ev-*`` identifiers.
"""

from .contracts import (
    IncidentMemoryRecord,
    KnowledgeCandidate,
    KnowledgeProvenance,
    KnowledgeQuery,
    KnowledgeRetrievalDiagnostics,
    KnowledgeRetrievalResult,
    PriorContext,
)
from .ports import KnowledgeStore, MemoryVerificationError
from .storage import InMemoryKnowledgeStore, SQLiteKnowledgeStore
from .rag import (
    ChildChunk,
    DeterministicReranker,
    DocumentManifest,
    DomainDocumentIngestor,
    DomainKnowledgeRetriever,
    IngestedDomainDocument,
    ParentChunk,
    StructureAwareChunker,
    UnavailableReranker,
)
from .static_graph import (
    GraphEdge,
    GraphNode,
    InMemoryStaticGraph,
    StaticGraphQueryResult,
    StaticKnowledgeGraphBuilder,
    StaticKnowledgeGraphTools,
)
from .router import (
    CapabilitySnapshot,
    HybridRouter,
    IncidentEntities,
    IncidentEntityExtractor,
    KnowledgeCoordinator,
    KnowledgeQueryBuilder,
    RouterDecision,
)
from .tool import KnowledgeRetrievalArgs, KnowledgeRetrievalTool

__all__ = [
    "IncidentMemoryRecord",
    "KnowledgeCandidate",
    "KnowledgeProvenance",
    "KnowledgeQuery",
    "KnowledgeRetrievalDiagnostics",
    "KnowledgeRetrievalResult",
    "PriorContext",
    "KnowledgeStore",
    "MemoryVerificationError",
    "InMemoryKnowledgeStore",
    "SQLiteKnowledgeStore",
    "ChildChunk",
    "DeterministicReranker",
    "DocumentManifest",
    "DomainDocumentIngestor",
    "DomainKnowledgeRetriever",
    "IngestedDomainDocument",
    "ParentChunk",
    "StructureAwareChunker",
    "UnavailableReranker",
    "GraphEdge",
    "GraphNode",
    "InMemoryStaticGraph",
    "StaticGraphQueryResult",
    "StaticKnowledgeGraphBuilder",
    "StaticKnowledgeGraphTools",
    "CapabilitySnapshot",
    "HybridRouter",
    "IncidentEntities",
    "IncidentEntityExtractor",
    "KnowledgeCoordinator",
    "KnowledgeQueryBuilder",
    "RouterDecision",
    "KnowledgeRetrievalArgs",
    "KnowledgeRetrievalTool",
]

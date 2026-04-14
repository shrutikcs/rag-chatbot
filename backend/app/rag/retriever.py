from typing import List

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever

from app.core.config import settings
from app.rag.ingest import _make_vector_store, embed_model, DOCSTORE_PATH

# ── tunables ──────────────────────────────────────────────────────
DENSE_TOP_K = 8  # candidates from pgvector
SPARSE_TOP_K = 8  # candidates from BM25
FUSION_TOP_K = 8  # candidates after RRF merge
RERANK_TOP_N = 3  # final passages returned to the LLM


# ── reranking wrapper ────────────────────────────────────────────
class RerankedRetriever(BaseRetriever):

    def __init__(
        self,
        base_retriever: QueryFusionRetriever,
        reranker: CohereRerank,
    ) -> None:
        super().__init__()
        self._base = base_retriever
        self._reranker = reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 1. Get fused (RRF) candidates
        candidates = self._base.retrieve(query_bundle)

        # 2. Rerank with Cohere cross-encoder
        reranked = self._reranker.postprocess_nodes(
            candidates, query_bundle=query_bundle
        )
        return reranked


# ── index (read-only — no re-embedding) ──────────────────────────
def _load_index() -> VectorStoreIndex:
    """Reconnect to the existing pgvector store created during ingestion."""
    vector_store = _make_vector_store()
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


# ── build the full retrieval pipeline ────────────────────────────
def build_retriever() -> RerankedRetriever:

    index = _load_index()

    # ── 1. Dense retriever (pgvector cosine similarity) ──────────
    dense_retriever = index.as_retriever(similarity_top_k=DENSE_TOP_K)

    # ── 2. Sparse retriever (BM25 over the same nodes) ──────────
    docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
    nodes = list(docstore.docs.values())
    sparse_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=SPARSE_TOP_K,
    )

    # ── 3. Reciprocal Rank Fusion ────────────────────────────────
    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=FUSION_TOP_K,
        num_queries=1,  # no LLM query expansion; pure fusion only
        use_async=True,
        verbose=True,
    )

    # ── 4. Cohere rerank (cross-encoder final pass) ──────────────
    reranker = CohereRerank(
        api_key=settings.COHERE_API_KEY,
        model="rerank-english-v3.0",
        top_n=RERANK_TOP_N,
    )

    return RerankedRetriever(base_retriever=fusion_retriever, reranker=reranker)

import asyncio
import os
import cohere
from sqlalchemy import select

from db import engine, Document
from embeddings import embed_text
from rank_bm25 import BM25Okapi


async def vector_search(query: str, k: int = 5):
    # 1. Get query embedding
    query_vector_list = await embed_text([query])
    query_vector = query_vector_list[0]
    
    # 2. Search database using cosine distance
    async with engine.connect() as conn:
        stmt = (
            select(
                Document.id, 
                Document.content, 
                Document.embedding.cosine_distance(query_vector).label("distance")
            )
            .order_by("distance")
            .limit(k)
        )
        result = await conn.execute(stmt)
        return [{"id": row[0], "content": row[1], "score": 1 - row[2]} for row in result]

async def bm25_search(query: str, k: int = 5):
    # 1. Fetch all documents from DB
    async with engine.connect() as conn:
        result = await conn.execute(select(Document.id, Document.content))
        docs = [{"id": row[0], "content": row[1]} for row in result]
    
    if not docs:
        return []

    # 2. Tokenize for BM25 (simple whitespace tokenization)
    corpus = [doc["content"].lower().split() for doc in docs]
    bm25 = BM25Okapi(corpus)
    
    # 3. Get scores
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # 4. Attach scores and sort
    for doc, score in zip(docs, scores):
        doc["score"] = float(score)
    
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:k]

def reciprocal_rank_fusion(results_sets: list[list[dict]], k: int = 60):
    fused_scores = {}
    doc_map = {}
    
    for results in results_sets:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)
            
    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    final_results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["score"] = fused_scores[doc_id]
        final_results.append(doc)
        
    return final_results

async def rerank(query: str, documents: list[dict], top_n: int = 5):
    if not documents:
        return []
    
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    
    # Format docs for cohere
    doc_contents = [d["content"] for d in documents]
    
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=doc_contents,
        top_n=top_n,
    )
    
    reranked_results = []
    for result in response.results:
        original_doc = documents[result.index].copy()
        original_doc["score"] = float(result.relevance_score)
        reranked_results.append(original_doc)
        
    return reranked_results

async def retrieve_raw(query: str):

    # 1. Vector Search
    v_results = await vector_search(query, k=20)
    
    # 2. BM25 Search
    b_results = await bm25_search(query, k=20)
    
    # 3. RRF
    fused = reciprocal_rank_fusion([v_results, b_results])
    
    # 4. Rerank
    final = await rerank(query, fused, top_n=5)
    
    return final

async def main():
    query = "What is the gold market outlook?"
    print(f"Final retrieval results for: {query}")
    results = await retrieve_raw(query)
    for i, res in enumerate(results):
        print(f"{i+1}. [Score: {res['score']:.4f}] {res['content'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())

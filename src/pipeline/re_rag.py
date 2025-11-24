import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from config import OPENAI_API_KEY, ANSWER_MODEL, RERANK_MODEL, TOP_K
from retrieve import search
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def llm_rerank(query, retrieved_chunks):
    """Ask the LLM to re-rank retrieved chunks by relevance."""
    text_block = "\n\n".join(
        [f"[{i}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are a re-ranking model. Rank the following chunks by relevance to the query.

Query:
{query}

Chunks:
{text_block}

Return ONLY a JSON list of indices in best-to-worst order.
"""

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        ranked_indices = eval(raw_output)  # Example: [2,0,1,4,3]
        return ranked_indices
    except:
        # fallback to original order
        return list(range(len(retrieved_chunks)))

def re_rag(query):
    start = time.time()

    # Step 1: retrieve raw chunks
    retrieved = search(query, top_k=TOP_K)

    # Step 2: LLM re-ranking
    ranked_indices = llm_rerank(query, retrieved)
    reranked_chunks = [retrieved[i] for i in ranked_indices]

    # Step 3: create single context
    context = "\n\n".join([c["text"] for c in reranked_chunks])

    # Step 4: answer
    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = response.choices[0].message.content

    latency = time.time() - start

    return {
        "answer": answer,
        "reranked_order": ranked_indices,
        "chunks": reranked_chunks,
        "latency": latency,
    }


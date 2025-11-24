import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import OPENAI_API_KEY, ANSWER_MODEL, TOP_K
from retrieve import search

client = OpenAI(api_key=OPENAI_API_KEY)


def rewrite_query(original_query: str) -> str:
    """LLM rewrites the query to improve retrieval"""
    prompt = f"""
Rewrite the following query to make it more suitable for information retrieval.
Keep it concise but more explicit and unambiguous.

Original Query: "{original_query}"
Rewritten Query:
"""

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()



def dqr_rag(query):
    start_time = time.time()

    # Step 1 — Rewrite Query
    rewritten = rewrite_query(query)

    # Step 2 — Retrieve Chunks using rewritten query
    retrieved = search(rewritten, top_k=TOP_K)

    context = "\n\n".join([c["text"] for c in retrieved])

    # Step 3 — Final Answer
    answer_prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question: {query}

Rewritten Query Used for Retrieval: "{rewritten}"

Answer:
"""

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": answer_prompt}]
    )

    answer = response.choices[0].message.content.strip()

    latency = time.time() - start_time

    return {
        "rewritten_query": rewritten,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "latency": latency,
    }



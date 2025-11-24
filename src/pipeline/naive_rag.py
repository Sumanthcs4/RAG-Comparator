import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, ANSWER_MODEL, TOP_K
from retrieve import search
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


def format_context(chunks):
    """Join retrieved chunks into a context string."""
    return "\n\n".join([f"[Chunk {i+1}] {c['text']}" for i, c in enumerate(chunks)])


def generate_answer(query, context):
    """Send the context + query to OpenAI LLM."""
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    answer = response.choices[0].message.content
    usage = response.usage
    return answer, usage


def naive_rag(query: str):
    """Full Naive RAG pipeline."""
    start_time = time.time()

    # 1. Retrieve chunks from FAISS
    chunks = search(query, top_k=TOP_K)

    # 2. Format context
    context = format_context(chunks)

    # 3. Generate answer using OpenAI
    answer, usage = generate_answer(query, context)

    end_time = time.time()
    latency = round(end_time - start_time, 3)

    return {
        "pipeline": "Naive RAG",
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "context_tokens": usage.prompt_tokens,
        "answer_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_sec": latency,
    }



# utils/metrics.py
from openai import OpenAI

client = OpenAI()

# Simple MRR@5 computation based on chunk order
def compute_mrr(retrieved_chunks, relevant_idx=[0]):
    """
    retrieved_chunks: list of retrieved texts
    relevant_idx: indices of relevant chunks, default first
    """
    for rank, _ in enumerate(retrieved_chunks[:5], start=1):
        if rank-1 in relevant_idx:
            return round(1/rank, 3)
    return 0.0

# LLM-as-judge
def llm_judge(query, answer):
    prompt = f"""
You are a judge for evaluating answers.  
Question: {query}  
Answer: {answer}  

Evaluate the answer for correctness, conciseness, and relevance.  
Respond with one of: Correct / Partially Correct / Incorrect
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Cost estimator
def estimate_cost(res):
    # very rough: sum of tokens × $0.0004 per 1k tokens (example for gpt-4o-mini)
    total_tokens = res.get("total_tokens") or (res.get("context_tokens",0) + res.get("answer_tokens",0))
    cost = total_tokens * 0.0004 / 1000
    return cost

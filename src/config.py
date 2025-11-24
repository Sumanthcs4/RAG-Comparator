import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4.1-mini")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4.1-mini")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gpt-4.1-mini")

TOP_K = int(os.getenv("TOP_K", 5))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 3))

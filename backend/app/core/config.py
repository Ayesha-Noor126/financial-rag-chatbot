"""
Centralized application configuration.

Design choice: a single Pydantic Settings object, injected wherever needed,
instead of scattering os.environ.get() calls across the codebase. This gives
us validation at startup (fail fast if a required var is missing/malformed)
and makes every tunable parameter (chunk size, TopK, thresholds) visible in
one place so retrieval behavior can be tuned without touching business logic.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_name: str = "Financial Document RAG Chatbot"
    environment: str = "development"
    debug: bool = True



    # --- Langfuse observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"
    langfuse_enabled: bool = True

    # --- Langfuse Prompt Management ---
    # Each value is the prompt *name* registered in the Langfuse UI.
    # Override in .env to rename prompts without touching application code.
    # The SDK fetches the version labelled "production"; fall back to the
    # hardcoded Python string when Langfuse is unreachable or the prompt
    # does not exist yet (first deploy before seed_prompts.py has been run).
    langfuse_prompt_answer_system: str = "fin-rag-answer-system"
    langfuse_prompt_web_system: str = "fin-rag-web-system"
    langfuse_prompt_rewriter_system: str = "fin-rag-rewriter-system"

    # How long (seconds) the SDK keeps each prompt in its own in-process
    # cache before re-fetching from the network.  300 s = 5 min means a
    # prompt edited in the Langfuse UI is live everywhere within 5 minutes
    # without a server restart.  Set to 0 to always fetch fresh (dev mode).
    langfuse_prompt_cache_ttl_seconds: int = 300


    # --- Storage paths ---
    storage_dir: Path = BASE_DIR / "storage"
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    faiss_dir: Path = BASE_DIR / "storage" / "faiss_index"
    metadata_db_path: Path = BASE_DIR / "storage" / "metadata.db"

    # --- LLM ---
    # "openai" or "groq". Groq exposes an OpenAI-compatible endpoint, so both
    # providers are driven through the same OpenAI SDK client (see
    # app/core/llm_client.py) — switching providers is just changing these
    # three values in .env, no code changes.
    llm_provider: str = "groq"
    llm_api_key: str = Field(default="", description="Set via .env: OPENAI_API_KEY or GROQ_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"  # Groq default; use "gpt-5" if llm_provider=openai
    llm_base_url: str = "https://api.groq.com/openai/v1"  # OpenAI: https://api.openai.com/v1
    llm_temperature: float = 0.0  # deterministic, grounded answers only

    # --- NER ---
    spacy_model_name: str = "en_core_web_sm"

    # --- Embeddings ---
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024  # bge-large-en-v1.5 output dimension
    embedding_batch_size: int = 32

    # --- Chunking ---
    chunk_size_tokens: int = 700       # midpoint of 600-800 target range
    chunk_overlap_tokens: int = 100
    min_chunk_tokens: int = 50         # discard near-empty trailing chunks

    # --- Retrieval ---
    retrieval_top_k: int = 20          # candidates pulled before rerank
    rerank_top_k: int = 5              # final chunks passed to LLM
    rerank_candidates: int = 10        # max candidates scored by cross-encoder
                                       # profiler: 10→5 saves ~45% latency;
                                       # raise to 15 if recall degrades
    rrf_k_constant: int = 60           # standard RRF damping constant
    reranker_model_name: str = "BAAI/bge-reranker-base"
    rerank_skip_threshold: float = 0.026  # RRF score above which reranker is
                                          # skipped (both retrievers agreed strongly)
                                          # calibrated: real corpus max ≈ 0.027

    # --- Confidence / hallucination control ---
    confidence_threshold: float = 0.45  # below this -> consider Tavily / no-answer




    # --- Web search fallback ---
    tavily_api_key: str = ""
    enable_web_fallback: bool = True
    tavily_max_results: int = 5
    tavily_search_depth: str = "basic"  # "basic" or "advanced" (advanced = slower, deeper)

    # --- Caching ---
    embedding_cache_size: int = 512
    query_cache_ttl_seconds: int = 300

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.upload_dir, self.faiss_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Cached singleton accessor. FastAPI's dependency injection calls this via
    Depends(get_settings) so every request reuses the same validated config
    object instead of re-parsing environment variables per request.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings
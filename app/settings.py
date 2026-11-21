"""Configuration, read once from the environment.

Everything here has a default that works on a laptop with no .env at all
except the two API keys, which cannot have one. With no key the OpenAI and
Cohere sources report themselves unavailable and the GloVe source still works,
which is enough to explore the whole UI — see app/services/embedding.py.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    def __init__(self) -> None:
        # The repo's .env calls it OPENAI_KEY. OPENAI_API_KEY is what every
        # other tool on the machine calls it, so accept both rather than make
        # the name the reason nothing embeds.
        self.openai_key: str = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.openai_base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")

        # Cohere, the second HTTP provider. Same three settings as above, and
        # one that is not ours to choose: /v2/embed rejects a call with more
        # than 96 texts in it, so the batch is clamped to the documented
        # maximum rather than trusted to be sensible. A larger
        # EMBED_BATCH_SIZE is fine for OpenAI and would be a 400 here.
        self.cohere_key: str = (
            os.getenv("COHERE_KEY") or os.getenv("COHERE_API_KEY") or ""
        )
        self.cohere_base_url: str = os.getenv(
            "COHERE_BASE_URL", "https://api.cohere.com/v2"
        ).rstrip("/")
        self.cohere_batch_size: int = max(
            1, min(96, int(os.getenv("COHERE_BATCH_SIZE", "96")))
        )

        # One request per batch of words. The API accepts far more than this,
        # but the UI embeds words as they are typed, and a small batch keeps
        # the first points on screen sooner.
        self.embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "128"))
        self.embed_timeout: float = float(os.getenv("EMBED_TIMEOUT", "30"))

        # The local ONNX encoders run on CPU, so the batch is bounded by
        # memory and latency rather than by an API limit. 32 short strings is
        # one forward pass of a few hundred milliseconds.
        self.local_batch_size: int = int(os.getenv("LOCAL_BATCH_SIZE", "32"))

        # The translate-to-English pivot index. Only used for text the built-in
        # word sets do not already carry an exact translation for — see
        # app/services/translate.py.
        self.translate_model: str = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")
        self.translate_batch_size: int = int(os.getenv("TRANSLATE_BATCH_SIZE", "48"))
        self.translate_timeout: float = float(os.getenv("TRANSLATE_TIMEOUT", "60"))

        self.data_dir: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
        self.chroma_dir: Path = self.data_dir / "chroma"
        # Written by scripts/fetch_glove.py. Absent on a fresh clone.
        self.glove_path: Path = self.data_dir / os.getenv(
            "GLOVE_FILE", "glove.6B.300d.npz"
        )
        # A Hugging Face hub cache of our own, populated by
        # scripts/fetch_models.py. Deliberately under data/ rather than in
        # ~/.cache: 2.8 GB of weights that only this project uses should be
        # deleted by deleting this project.
        self.models_dir: Path = self.data_dir / os.getenv("MODELS_DIR", "models")

        # A projection of 200 words takes t-SNE a second or two, and the answer
        # only depends on the words, the method and its parameters. The cache
        # is what makes flipping between methods feel instant on the second
        # visit.
        self.projection_cache_size: int = int(os.getenv("PROJECTION_CACHE_SIZE", "128"))

        # t-SNE and UMAP are seeded, so the same words give the same picture
        # twice. Without this, adding one word reshuffles the entire layout and
        # the visualisation looks arbitrary — which it would be.
        self.random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

        # A hard stop on how many words one projection may contain. t-SNE is
        # O(n log n) with a large constant and this runs synchronously inside a
        # request; past a couple of thousand points the browser is the next
        # thing to fall over anyway.
        self.max_words_per_projection: int = int(os.getenv("MAX_WORDS", "2000"))

        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:1186,http://localhost:5174,http://localhost:5173",
            ).split(",")
            if o.strip()
        ]
        # Requests per rolling minute per client over /api/*. 0 turns it off,
        # which is right for a laptop. A deployment with no login in front of
        # paid API keys wants it on — see app/ratelimit.py.
        self.rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.port: int = int(os.getenv("PORT", "1184"))
        self.reload: bool = os.getenv("RELOAD", "false").lower() == "true"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_key)

    @property
    def cohere_configured(self) -> bool:
        return bool(self.cohere_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

"""Configuración central. Todo lo que puede cambiar (modelos, umbrales,
contacto) vive en variables de entorno / .env, nunca en el código."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    extra: dict = None  # parámetros extra del proveedor, p. ej. {"reasoning_effort": "low"}


@dataclass(frozen=True)
class Settings:
    llm_endpoints: tuple[LLMEndpoint, ...] = ()
    llm_temperature: float = 0.3
    llm_timeout: float = 25.0
    llm_max_retries: int = 3
    llm_total_budget: float = 80.0  # RNF-FI-02: tope de espera total (<= 90 s)

    embed_model: str = "intfloat/multilingual-e5-small"
    embed_query_prefix: str = "query: "
    embed_doc_prefix: str = "passage: "

    chunk_words: int = 180
    chunk_overlap_words: int = 30
    top_k: int = 5
    sim_threshold: float = 0.75
    grounding_min: float = 0.45
    max_question_chars: int = 1000
    history_turns: int = 6

    kb_dir: Path = Path("data/kb")
    docs_dir: Path = Path("data/docs")
    db_path: Path = Path("data/app.db")
    session_timeout_minutes: int = 30

    database_url: str = ""  # si está vacío, usa SQLite; si tiene valor, PostgreSQL

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    coordinadora_email: str = ""

    admin_user: str = "coordinadora"
    admin_password: str = ""
    admin_host: str = "127.0.0.1"
    admin_port: int = 8000

    contact_email: str = ""
    contact_hours: str = "lunes a viernes de 8:00 a 15:00 horas"

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "Settings":
        """load_dotenv_file=False evita leer el .env real: lo usan las
        pruebas para no depender de lo que haya configurado cada máquina."""
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass

        import json

        endpoints = []
        for i in (1, 2, 3):
            base = os.getenv(f"LLM_{i}_BASE_URL", "").strip()
            model = os.getenv(f"LLM_{i}_MODEL", "").strip()
            if base and model:
                extra_raw = os.getenv(f"LLM_{i}_EXTRA", "").strip()
                try:
                    extra = json.loads(extra_raw) if extra_raw else {}
                except json.JSONDecodeError:
                    extra = {}
                endpoints.append(
                    LLMEndpoint(
                        name=os.getenv(f"LLM_{i}_NAME", f"llm{i}").strip(),
                        base_url=base,
                        api_key=os.getenv(f"LLM_{i}_API_KEY", "").strip(),
                        model=model,
                        extra=extra,
                    )
                )

        def f(name: str, default: float) -> float:
            return float(os.getenv(name, default))

        def i_(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        return cls(
            llm_endpoints=tuple(endpoints),
            llm_temperature=f("LLM_TEMPERATURE", 0.3),
            llm_timeout=f("LLM_TIMEOUT", 25),
            llm_max_retries=i_("LLM_MAX_RETRIES", 3),
            llm_total_budget=f("LLM_TOTAL_BUDGET", 80),
            embed_model=os.getenv("EMBED_MODEL", cls.embed_model),
            embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX", cls.embed_query_prefix),
            embed_doc_prefix=os.getenv("EMBED_DOC_PREFIX", cls.embed_doc_prefix),
            chunk_words=i_("CHUNK_WORDS", 180),
            chunk_overlap_words=i_("CHUNK_OVERLAP_WORDS", 30),
            top_k=i_("TOP_K", 5),
            sim_threshold=f("SIM_THRESHOLD", 0.75),
            grounding_min=f("GROUNDING_MIN", 0.45),
            kb_dir=Path(os.getenv("KB_DIR", "data/kb")),
            docs_dir=Path(os.getenv("DOCS_DIR", "data/docs")),
            database_url=os.getenv("DATABASE_URL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=i_("SMTP_PORT", 587),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_use_tls=os.getenv("SMTP_USE_TLS", "true").lower() != "false",
            coordinadora_email=os.getenv("COORDINADORA_EMAIL", ""),
            db_path=Path(os.getenv("DB_PATH", "data/app.db")),
            session_timeout_minutes=i_("SESSION_TIMEOUT_MINUTES", 30),
            admin_user=os.getenv("ADMIN_USER", "coordinadora"),
            admin_password=os.getenv("ADMIN_PASSWORD", ""),
            admin_host=os.getenv("ADMIN_HOST", "127.0.0.1"),
            admin_port=i_("ADMIN_PORT", 8000),
            contact_email=os.getenv("CONTACT_EMAIL", ""),
            contact_hours=os.getenv("CONTACT_HOURS", cls.contact_hours),
        )

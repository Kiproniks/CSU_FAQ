from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

from app.config import settings

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None


@dataclass
class DBStatus:
    enabled: bool
    reason: str = ""


class AnalyticsDB:
    """PostgreSQL-хранилище пользователей, токенов, web-auth, логов и benchmark-результатов."""

    def __init__(self, dsn: str) -> None:
        self.dsn = (dsn or "").strip()
        self.enabled = bool(self.dsn) and psycopg is not None
        self._warned = False
        self._last_connection_ok: bool | None = None

        if self.enabled:
            self.ensure_schema()
            self.ensure_default_web_admin()

    def status(self) -> DBStatus:
        if self.enabled and self._last_connection_ok is not False:
            return DBStatus(enabled=True)
        if self.enabled and self._last_connection_ok is False:
            return DBStatus(enabled=False, reason="Database connection failed")
        if not self.dsn:
            return DBStatus(enabled=False, reason="DATABASE_URL is empty")
        return DBStatus(enabled=False, reason="psycopg is not installed")

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"[DB] {message}")

    def _connect(self):
        if not self.enabled:
            return None
        try:
            conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
            self._last_connection_ok = True
            return conn
        except Exception as exc:
            self._last_connection_ok = False
            self._warn(f"Connection failed: {exc}")
            return None

    def ensure_schema(self) -> None:
        conn = self._connect()
        if conn is None:
            return

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        external_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                        is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                        UNIQUE (external_id, source)
                    );
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS bonus_tokens INTEGER NOT NULL DEFAULT 0;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS web_accounts (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        active BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_tokens (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        token TEXT NOT NULL UNIQUE,
                        active BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_used_at TIMESTAMPTZ NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NULL REFERENCES users(id) ON DELETE SET NULL,
                        source TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        query TEXT NOT NULL,
                        generated_answer TEXT NOT NULL DEFAULT '',
                        provider TEXT,
                        model TEXT,
                        latency_ms INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE query_logs
                    ADD COLUMN IF NOT EXISTS generated_answer TEXT NOT NULL DEFAULT '';
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        id BIGSERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        source_mode TEXT NOT NULL DEFAULT 'chunk',
                        evaluator_provider TEXT NOT NULL DEFAULT '',
                        evaluator_model TEXT NOT NULL DEFAULT '',
                        questions_file TEXT NOT NULL DEFAULT '',
                        created_by_source TEXT NOT NULL DEFAULT '',
                        created_by_external_id TEXT NOT NULL DEFAULT '',
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS benchmark_variants (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                        variant_index INTEGER NOT NULL,
                        variant_name TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        chunk_overlap INTEGER NOT NULL,
                        top_k INTEGER NOT NULL,
                        splitter_mode TEXT NOT NULL DEFAULT 'smart',
                        retrieval_mode TEXT NOT NULL DEFAULT 'chunk',
                        llm_provider TEXT NOT NULL DEFAULT '',
                        llm_model TEXT NOT NULL DEFAULT '',
                        config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (run_id, variant_index)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS benchmark_results (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                        variant_id BIGINT NOT NULL REFERENCES benchmark_variants(id) ON DELETE CASCADE,
                        question_index INTEGER NOT NULL,
                        question TEXT NOT NULL,
                        expected_answer TEXT NOT NULL DEFAULT '',
                        generated_answer TEXT NOT NULL DEFAULT '',
                        retrieved_context TEXT NOT NULL DEFAULT '',
                        score DOUBLE PRECISION NULL,
                        evaluator_comment TEXT NOT NULL DEFAULT '',
                        evaluator_raw JSONB NOT NULL DEFAULT '{}'::jsonb,
                        latency_ms INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_store (
                        id BIGSERIAL PRIMARY KEY,
                        chunk_id TEXT NOT NULL UNIQUE,
                        doc_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL DEFAULT 0,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (doc_id, chunk_index)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_store_doc_chunk
                    ON chunk_store (doc_id, chunk_index);
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entity_store (
                        id BIGSERIAL PRIMARY KEY,
                        chunk_id TEXT NOT NULL REFERENCES chunk_store(chunk_id) ON DELETE CASCADE,
                        doc_id TEXT NOT NULL,
                        entity TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (chunk_id, entity)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entity_store_entity
                    ON entity_store (entity);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_benchmark_variants_run_id
                    ON benchmark_variants(run_id);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_benchmark_results_run_variant
                    ON benchmark_results(run_id, variant_id);
                    """
                )

    def register_user(self, external_id: str, source: str) -> Optional[int]:
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (external_id, source)
                    VALUES (%s, %s)
                    ON CONFLICT (external_id, source)
                    DO UPDATE SET external_id = EXCLUDED.external_id
                    RETURNING id;
                    """,
                    (str(external_id), str(source)),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row["id"])

    def set_user_admin(self, external_id: str, source: str, value: bool) -> None:
        conn = self._connect()
        if conn is None:
            return

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET is_admin = %s
                    WHERE external_id = %s AND source = %s;
                    """,
                    (bool(value), str(external_id), str(source)),
                )

    def set_user_verified(self, external_id: str, source: str, value: bool) -> None:
        conn = self._connect()
        if conn is None:
            return

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET is_verified = %s
                    WHERE external_id = %s AND source = %s;
                    """,
                    (bool(value), str(external_id), str(source)),
                )

    def get_user(self, external_id: str, source: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, external_id, source, created_at, is_verified, is_admin, COALESCE(bonus_tokens, 0) AS bonus_tokens
                    FROM users
                    WHERE external_id = %s AND source = %s
                    LIMIT 1;
                    """,
                    (str(external_id), str(source)),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert_web_account(
        self,
        username: str,
        password: str,
        is_admin: bool = False,
        active: bool = True,
        force_password: bool = False,
    ) -> bool:
        """Создает/обновляет web-аккаунт и синхронизирует роль admin в users."""
        normalized_username = (username or "").strip()
        if not normalized_username or not password:
            return False

        user_id = self.register_user(external_id=normalized_username, source="web")
        if user_id is None:
            return False

        conn = self._connect()
        if conn is None:
            return False

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, password_hash
                    FROM web_accounts
                    WHERE username = %s
                    LIMIT 1;
                    """,
                    (normalized_username,),
                )
                existing = cur.fetchone()

                if existing:
                    if force_password:
                        cur.execute(
                            """
                            UPDATE web_accounts
                            SET password_hash = %s,
                                active = %s,
                                user_id = %s
                            WHERE username = %s;
                            """,
                            (generate_password_hash(password), bool(active), user_id, normalized_username),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE web_accounts
                            SET active = %s,
                                user_id = %s
                            WHERE username = %s;
                            """,
                            (bool(active), user_id, normalized_username),
                        )
                else:
                    cur.execute(
                        """
                        INSERT INTO web_accounts (user_id, username, password_hash, active)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (user_id, normalized_username, generate_password_hash(password), bool(active)),
                    )

                cur.execute(
                    """
                    UPDATE users
                    SET is_admin = %s
                    WHERE id = %s;
                    """,
                    (bool(is_admin), user_id),
                )

        return True

    def create_web_account(self, username: str, password: str) -> bool:
        """Публичная регистрация web-пользователя (без перезаписи существующего логина)."""
        normalized_username = (username or "").strip()
        if not normalized_username or not password:
            return False

        user_id = self.register_user(external_id=normalized_username, source="web")
        if user_id is None:
            return False

        conn = self._connect()
        if conn is None:
            return False

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO web_accounts (user_id, username, password_hash, active)
                    VALUES (%s, %s, %s, TRUE)
                    ON CONFLICT (username) DO NOTHING
                    RETURNING id;
                    """,
                    (
                        user_id,
                        normalized_username,
                        generate_password_hash(password),
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return False

                cur.execute(
                    """
                    UPDATE users
                    SET is_admin = FALSE
                    WHERE id = %s;
                    """,
                    (user_id,),
                )
        return True

    def ensure_default_web_admin(self) -> None:
        """Гарантирует наличие admin-аккаунта из .env."""
        if not settings.admin_web_password:
            return

        self.upsert_web_account(
            username=settings.web_admin_username,
            password=settings.admin_web_password,
            is_admin=True,
            active=True,
            force_password=True,
        )

    def authenticate_web_account(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return None

        normalized_username = (username or "").strip()
        if not normalized_username or not password:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT wa.username,
                           wa.password_hash,
                           wa.active,
                           u.external_id,
                           u.source,
                           u.is_admin,
                           u.is_verified
                    FROM web_accounts wa
                    JOIN users u ON u.id = wa.user_id
                    WHERE wa.username = %s
                    LIMIT 1;
                    """,
                    (normalized_username,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                if not bool(row["active"]):
                    return None
                if not check_password_hash(row["password_hash"], password):
                    return None

                return {
                    "username": row["username"],
                    "external_id": row["external_id"],
                    "source": row["source"],
                    "is_admin": bool(row["is_admin"]),
                    "is_verified": bool(row["is_verified"]),
                }

    def list_users(self, limit: int = 300) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH monthly_usage AS (
                        SELECT user_id, COUNT(*)::INT AS month_requests
                        FROM query_logs
                        WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', NOW())
                        GROUP BY user_id
                    ),
                    active_tokens AS (
                        SELECT user_id, COUNT(*)::INT AS active_tokens
                        FROM user_tokens
                        WHERE active = TRUE
                        GROUP BY user_id
                    )
                    SELECT u.external_id,
                           u.source,
                           u.created_at,
                           u.is_verified,
                           u.is_admin,
                           COALESCE(u.bonus_tokens, 0) AS bonus_tokens,
                           COALESCE(mu.month_requests, 0) AS month_requests,
                           COALESCE(at.active_tokens, 0) AS active_tokens
                    FROM users u
                    LEFT JOIN monthly_usage mu ON mu.user_id = u.id
                    LEFT JOIN active_tokens at ON at.user_id = u.id
                    ORDER BY u.created_at DESC
                    LIMIT %s;
                    """,
                    (max(1, int(limit)),),
                )
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]

    def monthly_usage(self, external_id: str, source: str) -> int:
        user_id = self.register_user(external_id=external_id, source=source)
        if user_id is None:
            return 0

        conn = self._connect()
        if conn is None:
            return 0

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)::INT AS cnt
                    FROM query_logs
                    WHERE user_id = %s
                      AND DATE_TRUNC('month', created_at) = DATE_TRUNC('month', NOW());
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                return int((row or {}).get("cnt", 0))

    def check_monthly_quota(self, external_id: str, source: str, monthly_limit: int) -> Dict[str, Any]:
        user = self.get_user(external_id=external_id, source=source)
        used = self.monthly_usage(external_id=external_id, source=source)
        is_admin = bool((user or {}).get("is_admin", False))
        bonus_tokens = max(0, int((user or {}).get("bonus_tokens", 0) or 0))
        base_limit = max(1, int(monthly_limit))
        if is_admin:
            return {
                "allowed": True,
                "limit": 0,
                "base_limit": base_limit,
                "bonus_tokens": bonus_tokens,
                "used": used,
                "remaining": 0,
                "is_unlimited": True,
            }

        limit = base_limit + bonus_tokens
        allowed = used < limit
        remaining = max(0, limit - used)
        return {
            "allowed": allowed,
            "limit": limit,
            "base_limit": base_limit,
            "bonus_tokens": bonus_tokens,
            "used": used,
            "remaining": remaining,
            "is_unlimited": False,
        }

    def adjust_user_bonus_tokens(self, external_id: str, source: str, delta: int) -> Optional[int]:
        """Изменяет бонусные токены (месячный лимит запросов) и возвращает новое значение."""
        user_id = self.register_user(external_id=external_id, source=source)
        if user_id is None:
            return None

        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET bonus_tokens = GREATEST(0, COALESCE(bonus_tokens, 0) + %s)
                    WHERE id = %s
                    RETURNING bonus_tokens;
                    """,
                    (int(delta), int(user_id)),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row["bonus_tokens"])

    def issue_token(self, external_id: str, source: str) -> Optional[str]:
        user_id = self.register_user(external_id, source)
        if user_id is None:
            return None

        conn = self._connect()
        if conn is None:
            return None

        token = secrets.token_urlsafe(18)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_tokens (user_id, token, active)
                    VALUES (%s, %s, TRUE);
                    """,
                    (user_id, token),
                )

        return token

    def get_active_token_for_user(self, external_id: str, source: str) -> Optional[str]:
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ut.token
                    FROM user_tokens ut
                    JOIN users u ON u.id = ut.user_id
                    WHERE u.external_id = %s
                      AND u.source = %s
                      AND ut.active = TRUE
                    ORDER BY ut.created_at DESC
                    LIMIT 1;
                    """,
                    (str(external_id), str(source)),
                )
                row = cur.fetchone()
                return row["token"] if row else None

    def verify_token(self, token: str, external_id: str | None = None, source: str | None = None) -> bool:
        conn = self._connect()
        if conn is None:
            return False

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ut.id
                    FROM user_tokens ut
                    WHERE ut.token = %s
                      AND ut.active = TRUE
                    LIMIT 1;
                    """,
                    (token.strip(),),
                )
                row = cur.fetchone()
                if not row:
                    return False

                cur.execute(
                    """
                    UPDATE user_tokens
                    SET last_used_at = NOW()
                    WHERE id = %s;
                    """,
                    (row["id"],),
                )

                if external_id and source:
                    self.set_user_verified(external_id=external_id, source=source, value=True)

                return True

    def list_tokens(self, limit: int = 200) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ut.token,
                           ut.active,
                           ut.created_at,
                           ut.last_used_at,
                           u.external_id,
                           u.source
                    FROM user_tokens ut
                    JOIN users u ON u.id = ut.user_id
                    ORDER BY ut.created_at DESC
                    LIMIT %s;
                    """,
                    (max(1, int(limit)),),
                )
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]

    def set_token_active(self, token: str, active: bool) -> bool:
        conn = self._connect()
        if conn is None:
            return False

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_tokens
                    SET active = %s
                    WHERE token = %s;
                    """,
                    (bool(active), token.strip()),
                )
                return cur.rowcount > 0

    def log_query(
        self,
        external_id: str,
        source: str,
        mode: str,
        query: str,
        generated_answer: str,
        provider: str,
        model: str,
        latency_ms: int,
    ) -> None:
        user_id = self.register_user(external_id, source)
        conn = self._connect()
        if conn is None or user_id is None:
            return

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (user_id, source, mode, query, generated_answer, provider, model, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        user_id,
                        source,
                        str(mode),
                        str(query),
                        str(generated_answer or ""),
                        str(provider or ""),
                        str(model or ""),
                        int(max(0, latency_ms)),
                    ),
                )

    def upsert_chunks(self, rows: List[Dict[str, Any]]) -> int:
        """Сохраняет чанки в PostgreSQL (authoritative storage)."""
        conn = self._connect()
        if conn is None or not rows:
            return 0

        saved = 0
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    chunk_id = str((row or {}).get("chunk_id", "")).strip()
                    doc_id = str((row or {}).get("doc_id", "")).strip() or "unknown"
                    text = str((row or {}).get("text", "") or "")
                    metadata = (row or {}).get("metadata", {}) or {}
                    try:
                        chunk_index = int((row or {}).get("chunk_index", 0) or 0)
                    except Exception:
                        chunk_index = 0
                    if not chunk_id or not text.strip():
                        continue

                    cur.execute(
                        """
                        INSERT INTO chunk_store (chunk_id, doc_id, chunk_index, text, metadata, updated_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            chunk_index = EXCLUDED.chunk_index,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW();
                        """,
                        (
                            chunk_id,
                            doc_id,
                            int(max(0, chunk_index)),
                            text,
                            self._to_json_text(metadata if isinstance(metadata, dict) else {}),
                        ),
                    )
                    saved += 1
        return saved

    def load_chunks(self, limit: int = 200000) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, chunk_index, text, metadata
                    FROM chunk_store
                    ORDER BY doc_id ASC, chunk_index ASC, id ASC
                    LIMIT %s;
                    """,
                    (max(1, int(limit)),),
                )
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]

    def replace_chunk_entities(self, chunk_id: str, doc_id: str, entities: List[str]) -> int:
        conn = self._connect()
        if conn is None:
            return 0
        normalized = []
        for item in entities or []:
            value = str(item or "").strip().lower()
            if value:
                normalized.append(value)
        unique_entities = sorted(set(normalized))

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM entity_store
                    WHERE chunk_id = %s;
                    """,
                    (str(chunk_id),),
                )
                inserted = 0
                for entity in unique_entities:
                    cur.execute(
                        """
                        INSERT INTO entity_store (chunk_id, doc_id, entity)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (chunk_id, entity) DO NOTHING;
                        """,
                        (str(chunk_id), str(doc_id or "unknown"), entity),
                    )
                    inserted += 1
        return inserted

    def clear_chunk_entity_storage(self) -> None:
        conn = self._connect()
        if conn is None:
            return
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM entity_store;")
                cur.execute("DELETE FROM chunk_store;")

    def clear_entity_storage(self) -> None:
        conn = self._connect()
        if conn is None:
            return
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM entity_store;")

    def top_faq_questions(self, last_n: int = 100, top_n: int = 10) -> List[Dict[str, Any]]:
        """Топ часто задаваемых вопросов по последним N запросам."""
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH recent AS (
                        SELECT query, created_at, generated_answer
                        FROM query_logs
                        WHERE LENGTH(TRIM(query)) > 0
                        ORDER BY created_at DESC
                        LIMIT %s
                    ),
                    grouped AS (
                        SELECT LOWER(TRIM(query)) AS normalized_query,
                               MAX(query) AS sample_query,
                               COUNT(*)::INT AS ask_count,
                               MAX(created_at) AS last_asked_at
                        FROM recent
                        GROUP BY LOWER(TRIM(query))
                    ),
                    latest_answers AS (
                        SELECT DISTINCT ON (LOWER(TRIM(query)))
                               LOWER(TRIM(query)) AS normalized_query,
                               TRIM(generated_answer) AS generated_answer
                        FROM recent
                        WHERE LENGTH(TRIM(COALESCE(generated_answer, ''))) > 0
                        ORDER BY LOWER(TRIM(query)), created_at DESC
                    )
                    SELECT g.sample_query AS question,
                           g.ask_count,
                           g.last_asked_at,
                           COALESCE(la.generated_answer, '') AS answer
                    FROM grouped g
                    LEFT JOIN latest_answers la ON la.normalized_query = g.normalized_query
                    ORDER BY g.ask_count DESC, g.last_asked_at DESC
                    LIMIT %s;
                    """,
                    (max(1, int(last_n)), max(1, int(top_n))),
                )
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]

    @staticmethod
    def _build_full_series(rows: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
        """Заполняет пропуски дней нулями, чтобы график был непрерывным."""
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=max(1, days) - 1)
        counter = {row["day"]: int(row["count"]) for row in rows}

        result: List[Dict[str, Any]] = []
        day = start
        while day <= today:
            result.append({"day": day.isoformat(), "count": counter.get(day, 0)})
            day += timedelta(days=1)
        return result

    def activity_series(self, days: int = 14) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DATE(created_at) AS day, COUNT(*)::INT AS count
                    FROM query_logs
                    WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                    GROUP BY DATE(created_at)
                    ORDER BY day ASC;
                    """,
                    (max(1, int(days)),),
                )
                rows = cur.fetchall() or []
                normalized = [{"day": row["day"], "count": row["count"]} for row in rows]
                return self._build_full_series(normalized, max(1, int(days)))

    def new_users_series(self, days: int = 14) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DATE(created_at) AS day, COUNT(*)::INT AS count
                    FROM users
                    WHERE created_at >= NOW() - (%s || ' days')::INTERVAL
                    GROUP BY DATE(created_at)
                    ORDER BY day ASC;
                    """,
                    (max(1, int(days)),),
                )
                rows = cur.fetchall() or []
                normalized = [{"day": row["day"], "count": row["count"]} for row in rows]
                return self._build_full_series(normalized, max(1, int(days)))

    def start_benchmark_run(
        self,
        name: str,
        description: str = "",
        source_mode: str = "chunk",
        evaluator_provider: str = "",
        evaluator_model: str = "",
        questions_file: str = "",
        created_by_source: str = "",
        created_by_external_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> Optional[int]:
        """Создает запись benchmark-run и возвращает id."""
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark_runs (
                        name,
                        description,
                        source_mode,
                        evaluator_provider,
                        evaluator_model,
                        questions_file,
                        created_by_source,
                        created_by_external_id,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id;
                    """,
                    (
                        str(name or "chunk-benchmark"),
                        str(description or ""),
                        str(source_mode or "chunk"),
                        str(evaluator_provider or ""),
                        str(evaluator_model or ""),
                        str(questions_file or ""),
                        str(created_by_source or ""),
                        str(created_by_external_id or ""),
                        self._to_json_text(metadata or {}),
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row["id"])

    def add_benchmark_variant(
        self,
        run_id: int,
        variant_index: int,
        variant_name: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        splitter_mode: str = "smart",
        retrieval_mode: str = "chunk",
        llm_provider: str = "",
        llm_model: str = "",
        config_json: Dict[str, Any] | None = None,
    ) -> Optional[int]:
        """Регистрирует конфигурацию варианта benchmark-прогона."""
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark_variants (
                        run_id,
                        variant_index,
                        variant_name,
                        chunk_size,
                        chunk_overlap,
                        top_k,
                        splitter_mode,
                        retrieval_mode,
                        llm_provider,
                        llm_model,
                        config_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id, variant_index)
                    DO UPDATE SET
                        variant_name = EXCLUDED.variant_name,
                        chunk_size = EXCLUDED.chunk_size,
                        chunk_overlap = EXCLUDED.chunk_overlap,
                        top_k = EXCLUDED.top_k,
                        splitter_mode = EXCLUDED.splitter_mode,
                        retrieval_mode = EXCLUDED.retrieval_mode,
                        llm_provider = EXCLUDED.llm_provider,
                        llm_model = EXCLUDED.llm_model,
                        config_json = EXCLUDED.config_json
                    RETURNING id;
                    """,
                    (
                        int(run_id),
                        int(variant_index),
                        str(variant_name or f"variant-{variant_index}"),
                        int(max(1, chunk_size)),
                        int(max(0, chunk_overlap)),
                        int(max(1, top_k)),
                        str(splitter_mode or "smart"),
                        str(retrieval_mode or "chunk"),
                        str(llm_provider or ""),
                        str(llm_model or ""),
                        self._to_json_text(config_json or {}),
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row["id"])

    def add_benchmark_result(
        self,
        run_id: int,
        variant_id: int,
        question_index: int,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_context: str,
        score: float | None,
        evaluator_comment: str,
        evaluator_raw: Dict[str, Any] | None = None,
        latency_ms: int = 0,
    ) -> Optional[int]:
        """Сохраняет результат одного вопроса для конкретного variant."""
        conn = self._connect()
        if conn is None:
            return None

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark_results (
                        run_id,
                        variant_id,
                        question_index,
                        question,
                        expected_answer,
                        generated_answer,
                        retrieved_context,
                        score,
                        evaluator_comment,
                        evaluator_raw,
                        latency_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id;
                    """,
                    (
                        int(run_id),
                        int(variant_id),
                        int(question_index),
                        str(question or ""),
                        str(expected_answer or ""),
                        str(generated_answer or ""),
                        str(retrieved_context or ""),
                        (None if score is None else float(score)),
                        str(evaluator_comment or ""),
                        self._to_json_text(evaluator_raw or {}),
                        int(max(0, latency_ms)),
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row["id"])

    @staticmethod
    def _to_json_text(payload: Dict[str, Any]) -> str:
        import json

        return json.dumps(payload, ensure_ascii=False)


_db: AnalyticsDB | None = None


def get_database() -> AnalyticsDB:
    global _db
    if _db is None:
        _db = AnalyticsDB(settings.database_url)
    return _db

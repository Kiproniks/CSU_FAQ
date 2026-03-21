from __future__ import annotations

import json
import time
import uuid
from typing import Any

from flask import Flask, abort, make_response, redirect, render_template, request, session, url_for

from app.access_tokens import verify_signed_payload
from app.config import settings
from app.db import get_database
from app.rag_pipeline import RAGPipeline
from app.source_attribution import SourceAttributionFormatter

# Веб-приложение и лениво создаваемый пайплайн, общий для всех запросов.
app = Flask(__name__)
app.secret_key = settings.web_secret_key
_pipeline: RAGPipeline | None = None


# -------------------------
# Helpers
# -------------------------

def get_pipeline() -> RAGPipeline:
    # Создаем пайплайн один раз, чтобы не переинициализировать индексы на каждый запрос.
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _snippet(text: str, limit: int = 260) -> str:
    # Короткий предпросмотр фрагмента в блоке «На чем основан ответ».
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _build_faq_rows_with_answers(rows: list[dict[str, Any]], mode: str = "chunk") -> list[dict[str, Any]]:
    # Для каждого FAQ-вопроса показываем последний сохраненный ответ из query_logs.
    if not rows:
        return []

    enriched: list[dict[str, Any]] = []
    for row in rows:
        question = " ".join(str((row or {}).get("question", "")).split())
        if not question:
            continue

        saved_answer = " ".join(str((row or {}).get("answer", "")).split())
        if saved_answer:
            answer_text = _snippet(saved_answer, limit=420)
        else:
            answer_text = "Ответ появится после следующего запроса этого вопроса в QA."

        enriched.append(
            {
                "question": question,
                "answer": answer_text,
            }
        )
    return enriched


def _safe_next_url(raw: str | None) -> str:
    candidate = (raw or "").strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return "/"


def _get_or_create_guest_user_id() -> tuple[str, bool]:
    """Возвращает id гостя и флаг, нужно ли выставить cookie."""
    current = request.cookies.get("csu_user_id", "").strip()
    if current:
        return current, False
    return f"web-guest-{uuid.uuid4().hex}", True


def _set_auth_session(auth: dict[str, Any]) -> None:
    session["auth_username"] = auth.get("username", "")
    session["auth_external_id"] = auth.get("external_id", "")
    session["auth_source"] = auth.get("source", "web")
    session["auth_is_admin"] = bool(auth.get("is_admin", False))


def _clear_auth_session() -> None:
    for key in ["auth_username", "auth_external_id", "auth_source", "auth_is_admin"]:
        session.pop(key, None)


def _get_session_user() -> dict[str, Any] | None:
    external_id = (session.get("auth_external_id") or "").strip()
    source = (session.get("auth_source") or "").strip()
    if not external_id or not source:
        return None

    return {
        "username": (session.get("auth_username") or "").strip() or external_id,
        "external_id": external_id,
        "source": source,
        "is_admin": bool(session.get("auth_is_admin", False)),
    }


def _is_admin_authenticated() -> bool:
    user = _get_session_user()
    return bool(user and user.get("is_admin"))


def _require_admin():
    if not _is_admin_authenticated():
        return redirect(url_for("auth_login", next="/admin"))
    return None


def _parse_days(raw_value: str | None) -> int:
    try:
        return max(1, int(raw_value or str(settings.admin_default_days)))
    except ValueError:
        return settings.admin_default_days


def _current_identity() -> tuple[str, str, bool, dict[str, Any] | None]:
    """Возвращает identity для логирования/квот: (external_id, source, set_cookie, auth_user)."""
    auth_user = _get_session_user()
    if auth_user:
        return auth_user["external_id"], auth_user["source"], False, auth_user

    guest_id, should_set_cookie = _get_or_create_guest_user_id()
    return guest_id, "web_guest", should_set_cookie, None


def _get_mini_claims() -> dict[str, Any] | None:
    """Проверяет signed access token для Telegram mini-app админки."""
    access_token = (request.args.get("access") or request.form.get("access") or "").strip()
    if not access_token:
        return None

    payload = verify_signed_payload(access_token, settings.web_secret_key)
    if not payload:
        return None

    try:
        chat_id = int(payload.get("chat_id", 0))
    except (TypeError, ValueError):
        return None

    if chat_id not in set(settings.admin_telegram_ids or []):
        return None

    return {
        "chat_id": chat_id,
        "access": access_token,
    }


def _render_admin_dashboard(days: int, *, mini_claims: dict[str, Any] | None = None) -> str:
    db = get_database()
    activity = db.activity_series(days=days)
    new_users = db.new_users_series(days=days)
    tokens = db.list_tokens(limit=300)
    users = db.list_users(limit=500)

    is_mini = mini_claims is not None

    return render_template(
        "admin.html",
        days=days,
        activity_json=json.dumps(activity, ensure_ascii=False),
        new_users_json=json.dumps(new_users, ensure_ascii=False),
        tokens=tokens,
        users=users,
        monthly_limit=settings.user_monthly_request_limit,
        db_enabled=db.status().enabled,
        db_reason=db.status().reason,
        is_mini=is_mini,
        access_token=(mini_claims or {}).get("access", ""),
        admin_identity=(f"telegram:{mini_claims['chat_id']}" if mini_claims else f"web:{(_get_session_user() or {}).get('username', 'admin')}"),
        token_action_url=url_for("admin_mini_token_action") if is_mini else url_for("admin_token_action"),
        issue_token_url=url_for("admin_mini_issue_token") if is_mini else url_for("admin_issue_token"),
        user_role_url=url_for("admin_mini_user_role") if is_mini else url_for("admin_user_role"),
        quota_action_url=url_for("admin_mini_quota_tokens") if is_mini else url_for("admin_quota_tokens"),
        create_user_url=url_for("admin_mini_create_web_user") if is_mini else url_for("admin_create_web_user"),
        dashboard_url=url_for("admin_mini_dashboard") if is_mini else url_for("admin_dashboard"),
        logout_url=("" if is_mini else url_for("auth_logout")),
        back_url=url_for("index"),
    )


# -------------------------
# Public pages
# -------------------------

@app.route("/", methods=["GET", "POST"])
def index() -> str:
    # Одностраничный сценарий: принять вопрос, запустить RAG в выбранном режиме, отрисовать результат.
    db = get_database()
    external_id, source, should_set_cookie, auth_user = _current_identity()
    db.register_user(external_id=external_id, source=source)

    result: dict[str, Any] | None = None
    basis_hits: list[dict[str, Any]] = []
    basis_cards: list[dict[str, Any]] = []
    basis_label = "ChunkBased"
    error = ""
    query = ""
    mode = "chunk"

    quota = db.check_monthly_quota(
        external_id=external_id,
        source=source,
        monthly_limit=settings.user_monthly_request_limit,
    )

    if request.method == "POST":
        # Чтение и валидация пользовательских параметров.
        query = (request.form.get("query") or "").strip()
        mode = (request.form.get("mode") or "chunk").strip().lower()
        if mode not in {"chunk", "entity"}:
            mode = "chunk"

        if not query:
            error = "Введите вопрос."
        elif not quota.get("allowed", True):
            error = (
                f"Лимит исчерпан: {quota.get('used', 0)} / {quota.get('limit', settings.user_monthly_request_limit)} "
                "запросов в этом месяце."
            )
        else:
            try:
                # Основной вызов пайплайна поиска и генерации.
                started = time.perf_counter()
                result = get_pipeline().answer(query=query, top_k=1, mode=mode)
                latency_ms = int((time.perf_counter() - started) * 1000)
                generated_answer = str(result.get("answer", "") or "").strip()

                db.log_query(
                    external_id=external_id,
                    source=source,
                    mode=mode,
                    query=query,
                    generated_answer=generated_answer,
                    provider=result.get("provider", ""),
                    model=result.get("model", ""),
                    latency_ms=latency_ms,
                )
                quota = db.check_monthly_quota(
                    external_id=external_id,
                    source=source,
                    monthly_limit=settings.user_monthly_request_limit,
                )
            except Exception as exc:
                error = f"Ошибка пайплайна: {exc}"

    if result:
        # Показ фрагментов, на которых основан ответ.
        basis_hits = (result.get("hits", []) or [])[:1]
        basis_cards = [SourceAttributionFormatter.to_card(hit) for hit in basis_hits]
        basis_label = "EntityBased (TF-IDF + entities)" if mode == "entity" else "ChunkBased"

    html = render_template(
        "index.html",
        query=query,
        mode=mode,
        result=result,
        basis_hits=basis_hits,
        basis_cards=basis_cards,
        basis_label=basis_label,
        error=error,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        snippet=_snippet,
        db_enabled=db.status().enabled,
        db_reason=db.status().reason,
        auth_user=auth_user,
        is_admin=bool(auth_user and auth_user.get("is_admin")),
        monthly_limit=quota.get("limit", settings.user_monthly_request_limit),
        monthly_used=quota.get("used", 0),
        monthly_remaining=quota.get("remaining", settings.user_monthly_request_limit),
        monthly_unlimited=bool(quota.get("is_unlimited", False)),
        auth_login_url=url_for("auth_login", next=request.path),
        auth_register_url=url_for("auth_register", next=request.path),
        auth_logout_url=url_for("auth_logout"),
        faq_url=url_for("faq_page"),
        admin_url=url_for("admin_dashboard"),
    )

    response = make_response(html)
    if should_set_cookie:
        response.set_cookie("csu_user_id", external_id, max_age=60 * 60 * 24 * 365)
    return response


@app.route("/faq", methods=["GET"])
def faq_page() -> str:
    db = get_database()
    auth_user = _get_session_user()
    faq_error = ""
    try:
        rows = db.top_faq_questions(last_n=100, top_n=10)
    except Exception as exc:
        rows = []
        faq_error = f"FAQ временно недоступен: {exc}"
    faq_rows = _build_faq_rows_with_answers(rows, mode="chunk")

    return render_template(
        "faq.html",
        faq_rows=faq_rows,
        faq_error=faq_error,
        auth_user=auth_user,
        is_admin=bool(auth_user and auth_user.get("is_admin")),
        db_enabled=db.status().enabled,
        db_reason=db.status().reason,
        auth_login_url=url_for("auth_login", next="/faq"),
        auth_register_url=url_for("auth_register", next="/faq"),
        auth_logout_url=url_for("auth_logout"),
        index_url=url_for("index"),
        admin_url=url_for("admin_dashboard"),
    )


@app.route("/verify", methods=["GET", "POST"])
def verify() -> str:
    # Верификация отключена: маршрут оставлен только для обратной совместимости.
    return redirect(url_for("index"))


# -------------------------
# Web auth
# -------------------------

@app.route("/auth/login", methods=["GET", "POST"])
def auth_login() -> str:
    db = get_database()
    next_url = _safe_next_url(request.args.get("next") or request.form.get("next"))
    error = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        auth = db.authenticate_web_account(username=username, password=password)
        if auth:
            _set_auth_session(auth)
            return redirect(next_url or url_for("index"))

        error = "Неверный логин или пароль"

    return render_template(
        "auth_login.html",
        error=error,
        next_url=next_url,
        login_url=url_for("auth_login", next=next_url),
        register_url=url_for("auth_register", next=next_url),
    )


@app.route("/auth/register", methods=["GET", "POST"])
def auth_register() -> str:
    db = get_database()
    next_url = _safe_next_url(request.args.get("next") or request.form.get("next"))
    error = ""
    success = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        password_repeat = request.form.get("password_repeat") or ""

        if len(username) < 3:
            error = "Логин должен быть не короче 3 символов."
        elif len(password) < 6:
            error = "Пароль должен быть не короче 6 символов."
        elif password != password_repeat:
            error = "Пароли не совпадают."
        else:
            created = db.create_web_account(username=username, password=password)
            if not created:
                error = "Пользователь с таким логином уже существует."
            else:
                auth = db.authenticate_web_account(username=username, password=password)
                if auth:
                    _set_auth_session(auth)
                    return redirect(next_url or url_for("index"))
                success = "Регистрация выполнена. Теперь войдите в аккаунт."

    return render_template(
        "auth_register.html",
        error=error,
        success=success,
        next_url=next_url,
        login_url=url_for("auth_login", next=next_url),
        register_url=url_for("auth_register", next=next_url),
    )


@app.route("/auth/logout")
def auth_logout() -> str:
    _clear_auth_session()
    return redirect(url_for("index"))


# -------------------------
# Admin (web)
# -------------------------

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login() -> str:
    # Оставлен для обратной совместимости старой ссылки.
    return redirect(url_for("auth_login", next="/admin"))


@app.route("/admin/logout")
def admin_logout() -> str:
    return redirect(url_for("auth_logout"))


@app.route("/admin", methods=["GET"])
def admin_dashboard() -> str:
    access = _require_admin()
    if access is not None:
        return access

    days = _parse_days(request.args.get("days"))
    return _render_admin_dashboard(days=days)


@app.route("/admin/token_action", methods=["POST"])
def admin_token_action() -> str:
    access = _require_admin()
    if access is not None:
        return access

    action = (request.form.get("action") or "").strip().lower()
    token = (request.form.get("token") or "").strip()

    db = get_database()
    if token and action in {"activate", "deactivate"}:
        db.set_token_active(token=token, active=(action == "activate"))

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_dashboard", days=days))


@app.route("/admin/issue_token", methods=["POST"])
def admin_issue_token() -> str:
    access = _require_admin()
    if access is not None:
        return access

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "web").strip().lower() or "web"

    if external_id:
        get_database().issue_token(external_id=external_id, source=source)

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_dashboard", days=days))


@app.route("/admin/user_role", methods=["POST"])
def admin_user_role() -> str:
    access = _require_admin()
    if access is not None:
        return access

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "").strip()
    action = (request.form.get("action") or "").strip().lower()

    if external_id and source and action in {"make_admin", "remove_admin"}:
        get_database().set_user_admin(external_id=external_id, source=source, value=(action == "make_admin"))

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_dashboard", days=days))


@app.route("/admin/quota_tokens", methods=["POST"])
def admin_quota_tokens() -> str:
    access = _require_admin()
    if access is not None:
        return access

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "").strip()
    action = (request.form.get("action") or "").strip().lower()
    raw_amount = (request.form.get("amount") or "").strip()

    try:
        amount = int(raw_amount)
    except ValueError:
        amount = 0

    if external_id and source and amount > 0 and action in {"add", "take"}:
        delta = amount if action == "add" else -amount
        get_database().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=delta)

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_dashboard", days=days))


@app.route("/admin/create_web_user", methods=["POST"])
def admin_create_web_user() -> str:
    access = _require_admin()
    if access is not None:
        return access

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    role = (request.form.get("role") or "user").strip().lower()

    if username and password:
        get_database().upsert_web_account(
            username=username,
            password=password,
            is_admin=(role == "admin"),
            active=True,
            force_password=True,
        )

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_dashboard", days=days))


# -------------------------
# Admin (Telegram mini app)
# -------------------------

@app.route("/admin/mini", methods=["GET"])
def admin_mini_dashboard() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    days = _parse_days(request.args.get("days"))
    return _render_admin_dashboard(days=days, mini_claims=claims)


@app.route("/admin/mini/token_action", methods=["POST"])
def admin_mini_token_action() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    action = (request.form.get("action") or "").strip().lower()
    token = (request.form.get("token") or "").strip()

    db = get_database()
    if token and action in {"activate", "deactivate"}:
        db.set_token_active(token=token, active=(action == "activate"))

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_mini_dashboard", access=claims["access"], days=days))


@app.route("/admin/mini/issue_token", methods=["POST"])
def admin_mini_issue_token() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "web").strip().lower() or "web"

    if external_id:
        get_database().issue_token(external_id=external_id, source=source)

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_mini_dashboard", access=claims["access"], days=days))


@app.route("/admin/mini/user_role", methods=["POST"])
def admin_mini_user_role() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "").strip()
    action = (request.form.get("action") or "").strip().lower()

    if external_id and source and action in {"make_admin", "remove_admin"}:
        get_database().set_user_admin(external_id=external_id, source=source, value=(action == "make_admin"))

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_mini_dashboard", access=claims["access"], days=days))


@app.route("/admin/mini/quota_tokens", methods=["POST"])
def admin_mini_quota_tokens() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    external_id = (request.form.get("external_id") or "").strip()
    source = (request.form.get("source") or "").strip()
    action = (request.form.get("action") or "").strip().lower()
    raw_amount = (request.form.get("amount") or "").strip()

    try:
        amount = int(raw_amount)
    except ValueError:
        amount = 0

    if external_id and source and amount > 0 and action in {"add", "take"}:
        delta = amount if action == "add" else -amount
        get_database().adjust_user_bonus_tokens(external_id=external_id, source=source, delta=delta)

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_mini_dashboard", access=claims["access"], days=days))


@app.route("/admin/mini/create_web_user", methods=["POST"])
def admin_mini_create_web_user() -> str:
    claims = _get_mini_claims()
    if not claims:
        abort(403)

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    role = (request.form.get("role") or "user").strip().lower()

    if username and password:
        get_database().upsert_web_account(
            username=username,
            password=password,
            is_admin=(role == "admin"),
            active=True,
            force_password=True,
        )

    days = _parse_days(request.form.get("days"))
    return redirect(url_for("admin_mini_dashboard", access=claims["access"], days=days))

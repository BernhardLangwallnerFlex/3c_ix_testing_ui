import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io

import sanity

# ---------------------------
# Config
# ---------------------------

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("API_KEY", "changeme123")
DEFAULT_UI_USERNAME = os.getenv("UI_USERNAME", "admin")
DEFAULT_UI_PASSWORD = os.getenv("UI_PASSWORD", "")
DEFAULT_POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
DEFAULT_JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "600"))  # 10 minutes


# ---------------------------
# Products
# ---------------------------
# Each product is just an API profile (base URL + key). Functionality is
# identical across products; only the extraction JSON structure differs.
# VetCostCheck falls back to the legacy API_BASE_URL / API_KEY env vars for
# backwards compatibility.
PRODUCTS: Dict[str, Dict[str, str]] = {
    "VetCostCheck": {
        "base_url": os.getenv("VETCOSTCHECK_API_URL", DEFAULT_API_BASE_URL),
        "api_key": os.getenv("VETCOSTCHECK_API_KEY", DEFAULT_API_KEY),
    },
    "BPS": {
        "base_url": os.getenv("BPS_API_URL", "https://3cbps.flex-capital-scale.com"),
        "api_key": os.getenv("BPS_API_KEY", ""),
    },
    "Sanierer": {
        "base_url": os.getenv("SANIERER_API_URL", "https://3csanierer.flex-capital-scale.com"),
        "api_key": os.getenv("SANIERER_API_KEY", ""),
    },
}
DEFAULT_PRODUCT = "VetCostCheck"


APP_VERSION = "ui-v4-multi-product"

AUTH_COOKIE_NAME = "ix_auth_token"


def _compute_auth_token() -> str:
    key = (DEFAULT_UI_PASSWORD or "ix-default-dev-key").encode()
    msg = DEFAULT_UI_USERNAME.encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _set_auth_cookie(token: str, max_age_days: int = 30) -> None:
    max_age = max_age_days * 86400
    js = f'<script>document.cookie = "{AUTH_COOKIE_NAME}={token}; path=/; max-age={max_age}; SameSite=Lax";</script>'
    st.components.v1.html(js, height=0)


def _clear_auth_cookie() -> None:
    js = f'<script>document.cookie = "{AUTH_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax";</script>'
    st.components.v1.html(js, height=0)


# ---------------------------
# Helpers
# ---------------------------

def render_pdf_page(pdf_bytes: bytes, page_number: int, zoom: float = 1.5):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_number = max(0, min(page_number, doc.page_count - 1))
        page = doc.load_page(page_number)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img, doc.page_count
    finally:
        doc.close()

def require_login() -> None:
    """Password protection with persistent cookie login."""
    if st.session_state.get("authenticated", False):
        return

    # Check for valid auth cookie
    expected_token = _compute_auth_token()
    cookie_token = st.context.cookies.get(AUTH_COOKIE_NAME, "")
    if cookie_token and hmac.compare_digest(cookie_token, expected_token):
        st.session_state["authenticated"] = True
        return

    st.title("🔒 Invoice Extraction UI")
    st.caption("Please log in to access this interface.")

    username = st.text_input("Username", value="", key="login_username")
    password = st.text_input("Password", value="", type="password", key="login_password")

    # If no password is configured, warn loudly (but allow local dev if you want)
    if not DEFAULT_UI_PASSWORD:
        st.warning(
            "UI_PASSWORD is not set. Set it via environment variables / Azure Container Apps dashboard "
            "to protect this UI."
        )

    if st.button("Log in"):
        if username == DEFAULT_UI_USERNAME and (DEFAULT_UI_PASSWORD == "" or password == DEFAULT_UI_PASSWORD):
            st.session_state["authenticated"] = True
            _set_auth_cookie(expected_token)
            st.rerun()
        else:
            st.error("Invalid credentials")


def api_headers(api_key: str) -> Dict[str, str]:
    return {"X-Api-Key": api_key} if api_key else {}



def safe_json_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def now_ts() -> float:
    return time.time()


# ---------------------------
# Session State
# ---------------------------
def init_state():
    if "runs" not in st.session_state:
        st.session_state["runs"] = []  # list of runs
    if "selected_run_id" not in st.session_state:
        st.session_state["selected_run_id"] = None
    if "selected_file_key" not in st.session_state:
        st.session_state["selected_file_key"] = None
    if "stop_polling" not in st.session_state:
        st.session_state["stop_polling"] = False


@dataclass
class FileJob:
    file_key: str  # unique key per file in a run
    filename: str
    content_type: str
    size_bytes: int
    cached_bytes: bytes

    file_id: Optional[str] = None
    job_id: Optional[str] = None
    status: str = "pending"  # pending|uploaded|queued|started|finished|failed
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    elapsed_sec: Optional[float] = None


def add_run(files: List[FileJob], product: str) -> str:
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    st.session_state["runs"].insert(0, {
        "run_id": run_id,
        "created_at": now_ts(),
        "product": product,
        "files": files,
    })
    st.session_state["selected_run_id"] = run_id
    st.session_state["selected_file_key"] = files[0].file_key if files else None
    return run_id


def get_run(run_id: str) -> Optional[dict]:
    for r in st.session_state["runs"]:
        if r["run_id"] == run_id:
            return r
    return None


def find_file(run: dict, file_key: str) -> Optional[FileJob]:
    for f in run["files"]:
        if f.file_key == file_key:
            return f
    return None


# ---------------------------
# API Calls
# ---------------------------
def api_upload(api_base_url: str, api_key: str, file_bytes: bytes, filename: str) -> str:
    url = f"{api_base_url.rstrip('/')}/upload"
    files = {"file": (filename, file_bytes)}
    r = requests.post(url, files=files, headers=api_headers(api_key), timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["file_id"]


def api_process(api_base_url: str, api_key: str, file_id: str) -> str:
    url = f"{api_base_url.rstrip('/')}/process"
    payload = {"file_id": file_id}
    r = requests.post(url, json=payload, headers=api_headers(api_key), timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["job_id"]


def api_job_status(api_base_url: str, api_key: str, job_id: str) -> dict:
    url = f"{api_base_url.rstrip('/')}/job/{job_id}"
    r = requests.get(url, headers=api_headers(api_key), timeout=60)
    # if server errors, show raw body
    if r.status_code >= 400:
        raise RuntimeError(f"Job status error {r.status_code}: {r.text}")
    return r.json()


# ---------------------------
# UI
# ---------------------------
def sidebar_config(product: str):
    cfg = PRODUCTS[product]
    st.sidebar.header("⚙️ API Configuration")
    # Key the fields per product so each product keeps its own (editable)
    # URL/key and switching products shows the right values.
    api_base_url = st.sidebar.text_input(
        "API_BASE_URL", value=cfg["base_url"], key=f"api_base_url_{product}"
    )
    api_key = st.sidebar.text_input(
        "API_KEY (X-Api-Key)", value=cfg["api_key"], type="password", key=f"api_key_{product}"
    )

    st.sidebar.header("⏱ Polling")
    poll_interval = st.sidebar.number_input("Poll interval (seconds)", min_value=0.2, max_value=10.0, value=DEFAULT_POLL_INTERVAL, step=0.2)
    job_timeout = st.sidebar.number_input("Job timeout (seconds)", min_value=30, max_value=3600, value=DEFAULT_JOB_TIMEOUT, step=30)
    st.sidebar.caption(f"API key length: {len(api_key or '')}")
    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear session runs"):
        st.session_state["runs"] = []
        st.session_state["selected_run_id"] = None
        st.session_state["selected_file_key"] = None
        st.rerun()

    if st.sidebar.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        _clear_auth_cookie()
        st.rerun()

    return api_base_url, api_key, float(poll_interval), int(job_timeout)


def upload_and_process_run(api_base_url: str, api_key: str, poll_interval: float, job_timeout: int, product: str):
    st.subheader("📤 Upload & Process")

    uploaded = st.file_uploader(
        "Select one or more files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        start_btn = st.button("Start processing", disabled=(not uploaded))
    with col_b:
        stop_btn = st.button("Stop polling", type="secondary")

    if stop_btn:
        st.session_state["stop_polling"] = True

    if not start_btn:
        return

    st.session_state["stop_polling"] = False

    # Build run object with cached bytes
    file_jobs: List[FileJob] = []
    for uf in uploaded:
        b = uf.getvalue()
        file_key = f"file_{uuid.uuid4().hex[:10]}"
        file_jobs.append(FileJob(
            file_key=file_key,
            filename=uf.name,
            content_type=uf.type or "application/octet-stream",
            size_bytes=len(b),
            cached_bytes=b,
            created_at=now_ts(),
            updated_at=now_ts(),
        ))

    run_id = add_run(file_jobs, product)

    # UI placeholders
    st.success(f"Created run: {run_id}")
    table_ph = st.empty()
    progress_ph = st.progress(0)

    # Step 1: Upload all files
    for i, fj in enumerate(file_jobs, start=1):
        fj.status = "uploading"
        fj.updated_at = now_ts()
        table_ph.dataframe(build_status_table(run_id), use_container_width=True)

        try:
            fj.file_id = api_upload(api_base_url, api_key, fj.cached_bytes, fj.filename)
            fj.status = "uploaded"
        except Exception as e:
            fj.status = "failed"
            fj.error = f"Upload failed: {e}"
        fj.updated_at = now_ts()

        progress_ph.progress(int((i / max(len(file_jobs), 1)) * 30))
        table_ph.dataframe(build_status_table(run_id), use_container_width=True)

    # Step 2: Trigger jobs
    triggered = 0
    for fj in file_jobs:
        if fj.status != "uploaded" or not fj.file_id:
            continue
        try:
            fj.job_id = api_process(api_base_url, api_key, fj.file_id)
            fj.status = "queued"
            triggered += 1
        except Exception as e:
            fj.status = "failed"
            fj.error = f"Process trigger failed: {e}"
        fj.updated_at = now_ts()

    progress_ph.progress(40)
    table_ph.dataframe(build_status_table(run_id), use_container_width=True)

    if triggered == 0:
        st.error("No jobs were triggered (all uploads failed?).")
        return

    # Step 3: Poll in a round-robin loop
    start_time = now_ts()
    finished_count = 0

    while True:
        if st.session_state.get("stop_polling", False):
            st.warning("Polling stopped by user.")
            break

        # timeout guard
        if now_ts() - start_time > job_timeout:
            st.error(f"Timeout reached after {job_timeout}s. Stopping polling.")
            break

        all_done = True

        for fj in file_jobs:
            if fj.status in ("finished", "failed"):
                continue
            if not fj.job_id:
                continue

            all_done = False

            try:
                data = api_job_status(api_base_url, api_key, fj.job_id)
                status = data.get("status", "unknown")

                # Map RQ statuses to UI statuses
                if status in ("queued", "deferred", "scheduled"):
                    fj.status = "queued"
                elif status in ("started", "running"):
                    fj.status = "started"
                elif status == "finished":
                    fj.status = "finished"
                    fj.result = data.get("result")
                    fj.elapsed_sec = now_ts() - fj.created_at
                    finished_count += 1
                elif status == "failed":
                    fj.status = "failed"
                    fj.error = data.get("error") or "Job failed"
                    fj.elapsed_sec = now_ts() - fj.created_at
                else:
                    fj.status = status

            except Exception as e:
                # If polling itself fails, store error but keep going a bit
                fj.error = f"Polling error: {e}"

            fj.updated_at = now_ts()

        # update UI
        table_ph.dataframe(build_status_table(run_id), use_container_width=True)

        # progress: 40% after trigger, 100% at completion
        done = sum(1 for f in file_jobs if f.status in ("finished", "failed"))
        pct = 40 + int((done / max(len(file_jobs), 1)) * 60)
        progress_ph.progress(min(pct, 100))

        if all_done:
            break

        time.sleep(poll_interval)

    progress_ph.progress(100)
    st.success("Run completed (or polling stopped/timeout).")


def build_status_table(run_id: str):
    run = get_run(run_id)
    rows = []
    if not run:
        return rows

    for f in run["files"]:
        rows.append({
            "filename": f.filename,
            "size_kb": round(f.size_bytes / 1024, 1),
            "file_id": f.file_id or "",
            "job_id": f.job_id or "",
            "status": f.status,
            "elapsed_sec": round(f.elapsed_sec, 1) if f.elapsed_sec is not None else "",
            "error": (f.error or "")[:200],
        })
    return rows


# ---------------------------
# Sanity-check rendering (presentation only — all logic lives in sanity.py)
# ---------------------------
_VERDICT_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skipped": "➖"}
_CURRENCY_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF "}


def _money(amount, currency) -> str:
    if amount is None:
        return "—"
    sym = _CURRENCY_SYMBOL.get(currency or "", f"{currency} " if currency else "")
    return f"{sym}{amount:,.2f}"


def _rollup_summary(report) -> str:
    docs = report.subdoc_count
    plural = "doc" if docs == 1 else "docs"
    if docs == 0:
        return "No documents to check"
    if report.verdict == "pass":
        return f"All checks passed · {docs} {plural}"
    if report.verdict == "skipped":
        return f"Not enough numbers to check · {docs} {plural}"
    fails = sum(1 for sd in report.subdocs if sd.verdict == "fail")
    warns = sum(1 for sd in report.subdocs if sd.verdict == "warn")
    parts = []
    if fails:
        parts.append(f"{fails} failed")
    if warns:
        parts.append(f"{warns} with warnings")
    return f"{', '.join(parts)} across {docs} {plural}"


def render_sanity_report(report, result: dict) -> None:
    """Render a SanityReport. Pure presentation: reads the report, no arithmetic."""
    icon = _VERDICT_ICON.get(report.verdict, "")
    headline = f"{icon} {_rollup_summary(report)}"

    # Always-visible rollup badge, colored by severity.
    box = {"pass": st.success, "warn": st.warning, "fail": st.error}.get(report.verdict, st.info)
    box(headline)

    if report.subdoc_count == 0:
        return

    raw_subdocs = (result or {}).get("subdocuments") or []
    multi = report.subdoc_count > 1
    for sd in report.subdocs:
        label = f"{_VERDICT_ICON.get(sd.verdict, '')} Doc {sd.index + 1}"
        if sd.number:
            label += f" · {sd.number}"
        # Expand docs that need attention; collapse clean ones when there are several.
        expanded = (sd.verdict in ("fail", "warn")) or not multi
        with st.expander(label, expanded=expanded):
            for c in sd.checks:
                bits = []
                if c.note:
                    bits.append(c.note)
                if c.verdict != "skipped" and c.delta is not None:
                    bits.append(f"Δ {_money(c.delta, sd.currency)}")
                suffix = "  —  " + " · ".join(bits) if bits else ""
                st.markdown(f"{_VERDICT_ICON.get(c.verdict, '')} **{c.label}**{suffix}")
                # computed vs reported, shown for anything not clean-passing
                if c.verdict in ("warn", "fail") and c.computed is not None:
                    st.caption(
                        f"computed {_money(c.computed, sd.currency)} · "
                        f"reported {_money(c.reported, sd.currency)}"
                    )

            # Model-emitted warnings: shown near the checks but visually distinct.
            raw = raw_subdocs[sd.index] if sd.index < len(raw_subdocs) else {}
            model_warnings = (raw or {}).get("warnings") or []
            if model_warnings:
                st.divider()
                st.caption("📝 Model notes (not sanity checks):")
                for w in model_warnings:
                    st.caption(f"• {w}")


def inspector_panel(product: str):
    st.subheader("🔎 Inspector")

    # Only show runs belonging to the active product.
    product_runs = [r for r in st.session_state["runs"] if r.get("product") == product]
    if not product_runs:
        st.info(f"No {product} runs yet. Upload and process files above.")
        return

    run_options = [r["run_id"] for r in product_runs]
    selected_run = st.selectbox("Select run", options=run_options, index=0)

    run = get_run(selected_run)
    if not run:
        st.warning("Run not found.")
        return

    file_options = [
        (f.file_key, f"{f.filename}  —  {f.status}")
        for f in run["files"]
    ]

    # default selection: first file
    default_idx = 0
    if st.session_state["selected_file_key"]:
        for i, (k, _) in enumerate(file_options):
            if k == st.session_state["selected_file_key"]:
                default_idx = i
                break

    chosen = st.selectbox(
        "Select file",
        options=[k for k, _ in file_options],
        format_func=lambda k: dict(file_options).get(k, k),
        index=default_idx,
    )
    st.session_state["selected_file_key"] = chosen

    f = find_file(run, chosen)
    if not f:
        st.warning("File not found in run.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 📄 Document Preview")

        is_pdf = (f.content_type == "application/pdf") or f.filename.lower().endswith(".pdf")

        if is_pdf:
            # page count cache
            if "pdf_page_cache" not in st.session_state:
                st.session_state["pdf_page_cache"] = {}

            cache_key = f.file_key
            if cache_key not in st.session_state["pdf_page_cache"]:
                doc = fitz.open(stream=f.cached_bytes, filetype="pdf")
                st.session_state["pdf_page_cache"][cache_key] = doc.page_count
                doc.close()

            page_count = st.session_state["pdf_page_cache"][cache_key]

            page_idx = st.number_input(
                "Page",
                min_value=1,
                max_value=page_count,
                value=1,
                step=1,
                key=f"page_{f.file_key}",
            ) - 1

            img, _ = render_pdf_page(f.cached_bytes, page_number=int(page_idx), zoom=1.6)

            # this is a plain image render; Chrome will not block it
            st.image(img, use_container_width=True)

        else:
            st.image(f.cached_bytes, caption=f.filename, use_container_width=True)

        st.caption(f"Cached locally in session • {round(f.size_bytes/1024,1)} KB")

    with right:
        if f.status == "finished" and f.result is not None:
            st.markdown("### 🩺 Sanity Checks")
            try:
                report = sanity.evaluate(f.result)
                render_sanity_report(report, f.result)
            except Exception as e:
                st.warning(f"Could not compute sanity checks: {e}")

            st.markdown("### 🧾 Extraction Result (JSON)")
            # collapsible tree
            st.json(f.result)

            # download
            st.download_button(
                "Download JSON",
                data=safe_json_pretty(f.result),
                file_name=f"{f.filename}.json",
                mime="application/json",
            )
        elif f.status == "failed":
            st.error("Job failed")
            st.code(f.error or "No error details")
        else:
            st.info(f"Not finished yet. Current status: {f.status}")
            if f.error:
                st.warning(f.error)


def docs_page(api_base_url: str):
    st.title("API Documentation")

    # Fetch the OpenAPI spec from the internal API
    try:
        resp = requests.get(f"{api_base_url}/openapi.json", timeout=5)
        spec_json = resp.text
    except Exception as e:
        st.error(f"Could not load API docs: {e}")
        return

    st.download_button(
        "⬇️ Download OpenAPI spec (JSON)",
        data=spec_json,
        file_name="openapi.json",
        mime="application/json",
    )

    # Render Swagger UI in an iframe-like component
    swagger_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            SwaggerUIBundle({{
                spec: {spec_json},
                dom_id: '#swagger-ui',
                presets: [SwaggerUIBundle.presets.apis],
                layout: "BaseLayout"
            }})
        </script>
    </body>
    </html>
    """
    st.components.v1.html(swagger_html, height=800, scrolling=True)


def main():
    # Favicon / browser tab icon
    # - Streamlit uses `page_icon` as the favicon (emoji or image).
    # - Keep this as the first Streamlit call in `main()`.
    page_icon = "🧾"
    try:
        page_icon = Image.open("logo_blau.jpg")
    except Exception:
        pass

    st.set_page_config(page_title="Invoice Extraction UI", page_icon=page_icon, layout="wide")

    # Widen the sidebar so the 3-product segmented control fits on one line,
    # and tighten the chip padding a touch.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { min-width: 340px; max-width: 340px; }
        [data-testid="stSidebar"] [role="group"] button { padding-left: 0.6rem; padding-right: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    require_login()
    if not st.session_state.get("authenticated", False):
        return

    init_state()

    # Centered logo at the top of the sidebar
    _c1, _c2, _c3 = st.sidebar.columns([1, 2, 1])
    with _c2:
        st.image("logo_blau.jpg", width=120)

    # Global product selector (single-select). Sits above navigation so it
    # applies to both the Invoice Processing and API Docs pages.
    product = st.sidebar.segmented_control(
        "Product", options=list(PRODUCTS), default=DEFAULT_PRODUCT
    )
    if product is None:  # single-select can be cleared by clicking the active chip
        product = DEFAULT_PRODUCT
    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio("Navigation", ["Invoice Processing", "API Docs"], index=0)

    # Sidebar config (always rendered so widgets persist across pages)
    api_base_url, api_key, poll_interval, job_timeout = sidebar_config(product)

    if page == "Invoice Processing":
        st.title(f"Invoice Extraction – Test Console · {product}")
        st.caption("Uploads are cached in this Streamlit session. Refreshing the page will lose cached PDFs.")
        upload_and_process_run(api_base_url, api_key, poll_interval, job_timeout, product)
        st.divider()
        inspector_panel(product)
    else:
        docs_page(api_base_url)


if __name__ == "__main__":
    main()
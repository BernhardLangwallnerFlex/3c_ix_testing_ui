import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
#import pymupdf
import fitz  # PyMuPDF
from PIL import Image
import io

# ---------------------------
# Config
# ---------------------------

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("INVOICE_API_KEY", "changeme123")
DEFAULT_UI_USERNAME = os.getenv("UI_USERNAME", "admin")
DEFAULT_UI_PASSWORD = os.getenv("UI_PASSWORD", "")
DEFAULT_POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
DEFAULT_JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "600"))  # 10 minutes


APP_VERSION = "ui-v3-pdf-image-render"

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
    """Very basic password protection (session-based)."""
    if st.session_state.get("authenticated", False):
        return

    st.title("🔒 Invoice Extraction UI")
    st.caption("Please log in to access this interface.")

    username = st.text_input("Username", value="", key="login_username")
    password = st.text_input("Password", value="", type="password", key="login_password")

    # If no password is configured, warn loudly (but allow local dev if you want)
    if not DEFAULT_UI_PASSWORD:
        st.warning(
            "UI_PASSWORD is not set. Set it via environment variables / Render dashboard "
            "to protect this UI."
        )

    if st.button("Log in"):
        if username == DEFAULT_UI_USERNAME and (DEFAULT_UI_PASSWORD == "" or password == DEFAULT_UI_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")


def api_headers(api_key: str) -> Dict[str, str]:
    print(f"api_key: {api_key}")
    return {"X-API-Key": api_key} if api_key else {}



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


def add_run(files: List[FileJob]) -> str:
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    st.session_state["runs"].insert(0, {
        "run_id": run_id,
        "created_at": now_ts(),
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
def sidebar_config():
    st.sidebar.header("⚙️ API Configuration")
    api_base_url = st.sidebar.text_input("API_BASE_URL", value=DEFAULT_API_BASE_URL)
    api_key = st.sidebar.text_input("API_KEY (X-API-Key)", value=DEFAULT_API_KEY, type="password")
        
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
        st.rerun()

    return api_base_url, api_key, float(poll_interval), int(job_timeout)


def upload_and_process_run(api_base_url: str, api_key: str, poll_interval: float, job_timeout: int):
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

    run_id = add_run(file_jobs)

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


def inspector_panel():
    st.subheader("🔎 Inspector")

    if not st.session_state["runs"]:
        st.info("No runs yet. Upload and process files above.")
        return

    run_options = [r["run_id"] for r in st.session_state["runs"]]
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
        st.markdown("### 🧾 Extraction Result (JSON)")
        if f.status == "finished" and f.result is not None:
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


def main():
    st.set_page_config(page_title="Invoice Extraction UI", layout="wide")

    require_login()
    if not st.session_state.get("authenticated", False):
        return

    init_state()

    st.title("🧪 Invoice Extraction – Test Console")
    
    st.caption("Uploads are cached in this Streamlit session. Refreshing the page will lose cached PDFs.")

    api_base_url, api_key, poll_interval, job_timeout = sidebar_config()

    upload_and_process_run(api_base_url, api_key, poll_interval, job_timeout)

    st.divider()
    inspector_panel()


if __name__ == "__main__":
    main()
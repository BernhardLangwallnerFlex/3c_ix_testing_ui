# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit-based testing UI for the 3C VetCostCheck invoice extraction API. Users upload PDF/image files, the app sends them to a FastAPI backend for processing, polls for results, and displays extracted JSON alongside a document preview. Also embeds the API's Swagger docs via an internal fetch of `/openapi.json`.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configuration is via environment variables (see `.env` for the full list):
- `API_BASE_URL` — backend API endpoint
- `API_KEY` — API key sent as `X-Api-Key` header
- `UI_USERNAME` / `UI_PASSWORD` — basic login credentials
- `POLL_INTERVAL_SECONDS`, `JOB_TIMEOUT_SECONDS` — polling behavior

## Docker / Deployment

Deployed as an Azure Container App (`ca-vetcostcheck-ui`) in the `cae-3c-invoice` environment alongside the API (`ca-invoice-api`). Build and push:

```bash
az acr build --registry cr3cinvoice --image vetcostcheck-ui:v1 .
```

Redeploy with `./deploy.sh`. When deployed in Azure, the UI communicates with the API via the internal URL `http://ca-invoice-api:8000` (no IP restrictions, no TLS needed).

## Architecture

Single-file Streamlit app (`app.py`) with two pages selectable via sidebar radio:

1. **Invoice Processing** — upload files, poll for results, inspect extractions
2. **API Docs** — embedded Swagger UI fetched from the backend's `/openapi.json`

Key flow for invoice processing:
1. **Login** — session-based password gate (`require_login`)
2. **Upload & Process** — files are uploaded to `/upload`, then `/process` is called per file, returning a `job_id`
3. **Polling** — round-robin polling of `/job/{job_id}` until all jobs finish or timeout
4. **Inspector** — side-by-side view: PDF rendered as images via PyMuPDF (left) and extraction JSON (right)

State is managed entirely in `st.session_state` using a list of "runs", each containing `FileJob` dataclass instances. Files are cached as bytes in session state (lost on page refresh).

## Key Dependencies

- `PyMuPDF` (imported as `fitz`) — renders PDF pages to images for the preview panel
- `Pillow` — image handling
- `requests` — API communication

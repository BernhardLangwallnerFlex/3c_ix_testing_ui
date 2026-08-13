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
- `UI_USERNAME` / `UI_PASSWORD` — basic login credentials
- `POLL_INTERVAL_SECONDS`, `JOB_TIMEOUT_SECONDS` — polling behavior
- Per-target API endpoints, one pair per (product, environment):

  | Product | Prod | Test |
  |---|---|---|
  | VetCostCheck | `VETCOSTCHECK_API_URL` / `VETCOSTCHECK_API_KEY` | `VETCOSTCHECK_TEST_API_URL` / `VETCOSTCHECK_TEST_API_KEY` |
  | BPS | `BPS_API_URL` / `BPS_API_KEY` | `BPS_TEST_API_URL` / `BPS_TEST_API_KEY` |
  | Sanierer | `SANIERER_API_URL` / `SANIERER_API_KEY` | `SANIERER_TEST_API_URL` / `SANIERER_TEST_API_KEY` |

  VetCostCheck **prod** falls back to the legacy `API_BASE_URL` / `API_KEY` if its own
  pair is unset. Test targets have no fallback: an unset test variable is reported in
  the sidebar rather than silently resolving to the prod endpoint.

## Docker / Deployment

Deployed as an Azure Container App (`ca-vetcostcheck-ui`) in the `cae-3c-invoice` environment alongside the API (`ca-invoice-api`). Build and push:

```bash
az acr build --registry cr3cinvoice --image vetcostcheck-ui:v1 .
```

Redeploy with `./deploy.sh`. When deployed in Azure, the UI communicates with the API via the internal URL `http://ca-invoice-api:8000` (no IP restrictions, no TLS needed).

### Container App environment variables

The app reads its targets from Container App env vars, with keys stored as secrets. The
test targets must be added once, before `Test` works in the deployed UI (until then the
sidebar reports them as unconfigured):

```bash
az containerapp secret set \
  --name ca-vetcostcheck-ui --resource-group rg-3c-invoice \
  --secrets \
    vetcostcheck-test-api-key="<VETCOSTCHECK_TEST_API_KEY>" \
    bps-test-api-key="<BPS_TEST_API_KEY>" \
    sanierer-test-api-key="<SANIERER_TEST_API_KEY>"

az containerapp update \
  --name ca-vetcostcheck-ui --resource-group rg-3c-invoice \
  --set-env-vars \
    VETCOSTCHECK_TEST_API_URL="https://3cvetcostcheck-test.flex-capital-scale.com" \
    BPS_TEST_API_URL="https://3cbps-test.flex-capital-scale.com" \
    SANIERER_TEST_API_URL="https://3csanierer-test.flex-capital-scale.com" \
    VETCOSTCHECK_TEST_API_KEY=secretref:vetcostcheck-test-api-key \
    BPS_TEST_API_KEY=secretref:bps-test-api-key \
    SANIERER_TEST_API_KEY=secretref:sanierer-test-api-key
```

Key values are in the local `.env` (gitignored). VetCostCheck prod needs no new variable —
it still resolves through the legacy `API_BASE_URL` / `API_KEY` pair already set there.

## Architecture

Single-file Streamlit app (`app.py`) with two pages selectable via sidebar radio:

1. **Invoice Processing** — upload files, poll for results, inspect extractions
2. **API Docs** — embedded Swagger UI fetched from the backend's `/openapi.json`

Two sidebar controls select the API target: **Product** (VetCostCheck / BPS / Sanierer) and
**Environment** (Test / Prod, defaulting to Test). Together they resolve to a base URL and
API key via `targets.resolve_target`. Runs are tagged with both, and the Inspector only
shows runs matching the active target — a prod run and a test run of the same file never
share a list. Prod is marked with a sidebar warning badge and in the page title.

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

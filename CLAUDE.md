# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Open work and known issues are tracked in `TODO.md`.

## Project Overview

Streamlit-based testing UI for the 3C invoice extraction APIs. Users upload PDF/image files, the app sends them to a FastAPI backend for processing, polls for results, and displays extracted JSON alongside a document preview. Also embeds the API's Swagger docs via a fetch of `/openapi.json`.

The UI targets **six** backends: three products (VetCostCheck, BPS, Sanierer) × two environments (Test, Prod), selected by two sidebar controls. All six expose an identical surface (`/upload`, `/process`, `/job/{job_id}`, `/healthz`, `/ready`) and return the same result shape; only the product-specific fields inside `subdocuments[]` differ (VetCostCheck has `animals`/`clinicians`/`diagnoses`, BPS has `policyholder`/`damageLocation`/`serviceProvider`). Because the surface is identical, no request or parsing logic is product- or environment-specific.

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

Deployed as an Azure Container App (`ca-vetcostcheck-ui`, resource group `rg-3c-invoice`) in the `cae-3c-invoice` environment.

**Always redeploy with `./deploy.sh`.** Do not hand-roll `az acr build` with a fixed tag: `az containerapp update` only creates a new revision when the template changes, so reusing a mutable tag like `:v1` silently keeps the *old* image running. `deploy.sh` tags each build with `<git-sha>-<utc-timestamp>` precisely to avoid that.

The UI reaches all six APIs over their public HTTPS endpoints (`https://3c<product>[-test].flex-capital-scale.com`), not over a Container Apps internal URL.

### Container App environment variables

The app reads its targets from Container App env vars, with keys stored as secrets. All six
targets are configured as of 2026-08-13; the commands below are for reference, or for
rotating a key.

Run these from the repo root. The first line sources `.env` so the real key values are
never typed into a shell or pasted into a terminal history — do **not** substitute
placeholder text by hand, or the placeholder becomes the secret value:

```bash
set -a; . ./.env; set +a

az containerapp secret set \
  --name ca-vetcostcheck-ui --resource-group rg-3c-invoice \
  --secrets \
    vetcostcheck-test-api-key="$VETCOSTCHECK_TEST_API_KEY" \
    bps-test-api-key="$BPS_TEST_API_KEY" \
    sanierer-test-api-key="$SANIERER_TEST_API_KEY"

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

A secret change needs a restart (a new revision) to take effect; `./deploy.sh` provides one.

If `az containerapp update` dies with a `JSONDecodeError` out of `handle_raw_exception`,
that is the CLI failing to parse a non-JSON error body (e.g. an ARM 503 HTML page), often
*after* the change already applied. Check the resource's actual state with
`az containerapp show` before re-running anything.

## Architecture

Streamlit app in `app.py`, with pure logic extracted into modules beside it:

| Module | Responsibility |
|---|---|
| `app.py` | All Streamlit UI, API calls, polling, session state |
| `targets.py` | The (product, environment) target model: `resolve_target`, `target_env_vars`, `filter_runs`. No Streamlit, no network |
| `sanity.py` | Arithmetic sanity checks over a result: `evaluate(result) -> SanityReport`. Pure, never throws |

Keep new logic out of `app.py` where it can be pure — that is what makes it testable, since the Streamlit layer itself has no automated coverage.

`app.py` has two pages selectable via sidebar radio:

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

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # run from the repo root
```

- `tests/test_sanity.py` — the sanity evaluator, against hand-built results and real captured fixtures
- `tests/test_targets.py` — target resolution and run scoping

Fixtures live in `test_data/results/<product>/*.json`, captured from the live APIs by
`scripts/fetch_samples.py`. Source PDFs under `test_data/<product>/` are gitignored (too
large); the small result JSONs are committed.

`app.py` is not unit-tested — verify UI changes by running the app. `python -c "import app"`
at least catches import and syntax errors.

## Design specs

Non-trivial features get a design spec in `specs/YYYY-MM-DD-<topic>-design.md`, and larger
ones an accompanying `-plan.md`. Read the relevant spec before changing the behaviour it
describes:

- `specs/2026-05-30-ui-sanity-checks-design.md` — the four arithmetic checks and tolerances
- `specs/2026-08-13-prod-test-environment-toggle-design.md` — the (product, environment) target model

## Key Dependencies

- `PyMuPDF` (imported as `fitz`) — renders PDF pages to images for the preview panel
- `Pillow` — image handling
- `requests` — API communication
- `pytest` (dev only, `requirements-dev.txt`)

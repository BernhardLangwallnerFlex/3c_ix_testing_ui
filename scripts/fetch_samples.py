#!/usr/bin/env python3
"""Fire selected test PDFs through each product's live API and save result JSON.

Reads per-product API_URL/API_KEY from .env. Outputs to test_data/results/<product>/<name>.json
Used to validate the extraction shape against the sanity-checks spec and to seed fixtures.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data" / "results"


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


ENV = load_env(ROOT / ".env")

PRODUCTS = {
    "vetcostcheck": (ENV["VETCOSTCHECK_API_URL"], ENV["VETCOSTCHECK_API_KEY"]),
    "bps": (ENV["BPS_API_URL"], ENV["BPS_API_KEY"]),
    "sanierer": (ENV["SANIERER_API_URL"], ENV["SANIERER_API_KEY"]),
}

# Representative selection per product (multi-doc VCC file included on purpose).
SELECTION = {
    "vetcostcheck": [
        "testrechnung_01_bulldogge.pdf",
        "testrechnung_03_katze.pdf",
        "VCC_Viele_Dokumente.pdf",
    ],
    "bps": ["BPS_2.pdf", "BPS_4.pdf"],
    "sanierer": [
        "LO_Rechnung.pdf",
        "Verkaufsrechnung_42613847.pdf",
        "AR26076770.pdf",
    ],
}

POLL_INTERVAL = 3.0
JOB_TIMEOUT = 300


def process_one(product: str, base_url: str, api_key: str, pdf: Path) -> dict:
    h = {"X-Api-Key": api_key}
    base = base_url.rstrip("/")
    # upload
    with pdf.open("rb") as fh:
        r = requests.post(f"{base}/upload", files={"file": (pdf.name, fh)}, headers=h, timeout=300)
    r.raise_for_status()
    file_id = r.json()["file_id"]
    # process
    r = requests.post(f"{base}/process", json={"file_id": file_id}, headers=h, timeout=300)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    # poll
    start = time.time()
    while True:
        r = requests.get(f"{base}/job/{job_id}", headers=h, timeout=60)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "finished":
            return data.get("result")
        if status == "failed":
            raise RuntimeError(f"job failed: {data.get('error')}")
        if time.time() - start > JOB_TIMEOUT:
            raise TimeoutError(f"timeout after {JOB_TIMEOUT}s (last status={status})")
        time.sleep(POLL_INTERVAL)


def main():
    for product, files in SELECTION.items():
        base_url, api_key = PRODUCTS[product]
        outdir = OUT / product
        outdir.mkdir(parents=True, exist_ok=True)
        for name in files:
            pdf = ROOT / "test_data" / product / name
            tag = f"[{product}] {name}"
            if not pdf.exists():
                print(f"{tag}: MISSING FILE", flush=True)
                continue
            t0 = time.time()
            try:
                result = process_one(product, base_url, api_key, pdf)
                (outdir / f"{pdf.stem}.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )
                n = (result or {}).get("number_of_subdocuments")
                print(f"{tag}: OK in {time.time()-t0:.0f}s · subdocs={n}", flush=True)
            except Exception as e:
                print(f"{tag}: ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

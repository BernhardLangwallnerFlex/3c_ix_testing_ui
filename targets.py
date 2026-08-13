"""API targets: a target is a (product, environment) pair.

Pure module — no Streamlit, no network, no I/O beyond reading a mapping.
See specs/2026-08-13-prod-test-environment-toggle-design.md.
"""
import os
from typing import Dict, List, Mapping, Optional, Tuple

# Display name -> environment-variable prefix.
PRODUCT_PREFIX: Dict[str, str] = {
    "VetCostCheck": "VETCOSTCHECK",
    "BPS": "BPS",
    "Sanierer": "SANIERER",
}
PRODUCTS: List[str] = list(PRODUCT_PREFIX)

ENVIRONMENTS: Tuple[str, str] = ("Test", "Prod")

DEFAULT_PRODUCT = "VetCostCheck"
DEFAULT_ENVIRONMENT = "Test"

# VetCostCheck production predates the per-product variables, and the deployed
# Container App still sets only this legacy pair. Test targets deliberately have
# no fallback of any kind — see spec §2.
_LEGACY_PROD_VARS: Dict[str, Tuple[str, str]] = {
    "VetCostCheck": ("API_BASE_URL", "API_KEY"),
}


def _validate(product: str, env: str) -> None:
    if product not in PRODUCT_PREFIX:
        raise ValueError(f"Unknown product: {product!r}")
    if env not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env!r}")


def target_env_vars(product: str, env: str) -> Tuple[str, str]:
    """Names of the (url, key) environment variables backing this target."""
    _validate(product, env)
    prefix = PRODUCT_PREFIX[product]
    infix = "_TEST" if env == "Test" else ""
    return f"{prefix}{infix}_API_URL", f"{prefix}{infix}_API_KEY"


def resolve_target(
    product: str, env: str, environ: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Resolve a (product, environment) pair to {"base_url", "api_key"}.

    Missing variables yield "" — never None, and never a value belonging to a
    different environment.
    """
    _validate(product, env)
    environ = os.environ if environ is None else environ

    url_var, key_var = target_env_vars(product, env)
    base_url = environ.get(url_var, "")
    api_key = environ.get(key_var, "")

    if env == "Prod" and product in _LEGACY_PROD_VARS:
        legacy_url, legacy_key = _LEGACY_PROD_VARS[product]
        base_url = base_url or environ.get(legacy_url, "")
        api_key = api_key or environ.get(legacy_key, "")

    return {"base_url": base_url, "api_key": api_key}


def filter_runs(runs: List[dict], product: str, env: str) -> List[dict]:
    """Runs belonging to this target, in their existing order.

    A run must match on both fields: a prod run and a test run of the same file
    are otherwise indistinguishable in the Inspector.
    """
    return [r for r in runs if r.get("product") == product and r.get("env") == env]

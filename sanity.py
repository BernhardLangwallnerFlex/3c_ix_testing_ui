"""Arithmetic sanity checks for extraction results.

Pure, product-agnostic evaluator. See specs/2026-05-30-ui-sanity-checks-design.md.
`evaluate(result)` takes the `result` object from GET /job/{job_id} and returns a
normalized SanityReport the render layer maps directly to UI. No I/O, never throws.

Decision (deviation from spec §4 Check 1): line-item math is capped at `warn` — it
never reports `fail` on its own, because real Rabatt/Pauschale lines carry noisy
qty/unitPrice while still reconciling to the total (Σ lineTotalNet → net is the
trustworthy signal). All other checks behave exactly as the spec describes.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- tolerance band (§5): the only tunable numbers, in one place ----------
TOL = {
    "pass_abs": 0.02,    # €0.02
    "pass_rel": 0.001,   # 0.1%
    "warn_abs": 1.00,    # €1.00
    "warn_rel": 0.010,   # 1.0%
}

PASS, WARN, FAIL, SKIPPED = "pass", "warn", "fail", "skipped"
_SEVERITY = {SKIPPED: 0, PASS: 1, WARN: 2, FAIL: 3}

CHECK_LABELS = {
    "lineItems": "Line-item math",
    "itemsSumNet": "Items → net",
    "tax": "Tax check",
    "gross": "Gross check",
}


@dataclass
class CheckResult:
    id: str
    label: str
    verdict: str
    computed: Optional[float] = None
    reported: Optional[float] = None
    delta: Optional[float] = None
    note: Optional[str] = None


@dataclass
class SubdocReport:
    index: int
    number: Optional[str]
    currency: Optional[str]
    verdict: str
    checks: List[CheckResult]


@dataclass
class SanityReport:
    verdict: str
    subdoc_count: int
    subdocs: List[SubdocReport]


# --- helpers --------------------------------------------------------------
def _num(x: Any) -> Optional[float]:
    """Real number → float; null/missing/bool/non-numeric → None (so 0 stays a value)."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _r(x: float) -> float:
    """Round to cents and normalize -0.0 → 0.0 for clean display."""
    return round(x, 2) + 0.0


def classify(delta: float, ref: float) -> str:
    a, r = abs(delta), abs(ref)
    if a <= max(TOL["pass_abs"], TOL["pass_rel"] * r):
        return PASS
    if a <= max(TOL["warn_abs"], TOL["warn_rel"] * r):
        return WARN
    return FAIL


def _worst(verdicts: List[str]) -> str:
    return max(verdicts, key=lambda v: _SEVERITY[v])


def _best(verdicts: List[str]) -> str:
    return min(verdicts, key=lambda v: _SEVERITY[v])


def _skipped(cid: str, note: str) -> CheckResult:
    return CheckResult(cid, CHECK_LABELS[cid], SKIPPED, note=note)


def _rollup(verdicts: List[str]) -> str:
    relevant = [v for v in verdicts if v != SKIPPED]
    return _worst(relevant) if relevant else SKIPPED


# --- the four checks ------------------------------------------------------
def _check_line_items(items: List[dict]) -> CheckResult:
    cid = "lineItems"
    verdicts: List[str] = []
    passed = 0
    for it in items:
        q, u, l = _num(it.get("qty")), _num(it.get("unitPriceNet")), _num(it.get("lineTotalNet"))
        if q is None or u is None or l is None:
            continue
        candidates = [q * u]
        disc = _num(it.get("discount"))
        if disc is not None:
            candidates.append(q * u - disc)                # discount as absolute amount
            candidates.append(q * u * (1 - disc / 100))    # discount as percentage
        best = _best([classify(c - l, l) for c in candidates])
        verdicts.append(best)
        if best == PASS:
            passed += 1
    if not verdicts:
        return _skipped(cid, "no line with qty, unit price and line total")
    worst = _worst(verdicts)
    if worst == FAIL:           # decision: cap at warn, never fail
        worst = WARN
    return CheckResult(cid, CHECK_LABELS[cid], worst, note=f"{passed}/{len(verdicts)} ok")


def _check_items_sum(items: List[dict], net: Any, totals_discount: Any) -> CheckResult:
    cid = "itemsSumNet"
    net = _num(net)
    line_totals = [_num(it.get("lineTotalNet")) for it in items]
    numeric = [x for x in line_totals if x is not None]
    if net is None:
        return _skipped(cid, "totals.net missing")
    if not numeric:
        return _skipped(cid, "no numeric line totals")
    s = sum(numeric)
    verdict = classify(s - net, net)
    used = s
    td = _num(totals_discount)
    if verdict != PASS and td is not None:
        v2 = classify((s - td) - net, net)
        if _SEVERITY[v2] < _SEVERITY[verdict]:
            verdict, used = v2, s - td
    note = None
    if any(x is None for x in line_totals):  # incomplete sum is unreliable
        note = f"computed from {len(numeric)}/{len(line_totals)} lines"
        if _SEVERITY[verdict] < _SEVERITY[WARN]:
            verdict = WARN
    return CheckResult(cid, CHECK_LABELS[cid], verdict, _r(used), _r(net),
                       _r(used - net), note)


def _check_tax(items: List[dict], net: Any, rate: Any, amount: Any) -> CheckResult:
    cid = "tax"
    net, rate, amount = _num(net), _num(rate), _num(amount)
    if net is None:
        return _skipped(cid, "totals.net missing")
    if rate is None:
        return _skipped(cid, "totals.tax.rate missing")
    if amount is None:
        return _skipped(cid, "totals.tax.amount missing")
    computed = net * (rate / 100)
    verdict = classify(computed - amount, amount)
    note = None
    item_rates = {_num(it.get("taxRate")) for it in items} - {None}
    if item_rates and any(r != rate for r in item_rates):
        if verdict == FAIL:
            verdict = WARN
        note = "mixed rates"
    return CheckResult(cid, CHECK_LABELS[cid], verdict, _r(computed), _r(amount),
                       _r(computed - amount), note)


def _check_gross(net: Any, amount: Any, gross: Any, totals_discount: Any) -> CheckResult:
    cid = "gross"
    net, amount, gross = _num(net), _num(amount), _num(gross)
    if net is None:
        return _skipped(cid, "totals.net missing")
    if amount is None:
        return _skipped(cid, "totals.tax.amount missing")
    if gross is None:
        return _skipped(cid, "totals.gross missing")
    computed = net + amount
    verdict = classify(computed - gross, gross)
    used = computed
    td = _num(totals_discount)
    if verdict != PASS and td is not None:
        cand = net - td + amount
        v2 = classify(cand - gross, gross)
        if _SEVERITY[v2] < _SEVERITY[verdict]:
            verdict, used = v2, cand
    return CheckResult(cid, CHECK_LABELS[cid], verdict, _r(used), _r(gross),
                       _r(used - gross))


def evaluate(result: Optional[Dict[str, Any]]) -> SanityReport:
    subdocs_raw = (result or {}).get("subdocuments") or []
    reports: List[SubdocReport] = []
    for i, s in enumerate(subdocs_raw):
        s = s or {}
        items = s.get("items") or []
        totals = s.get("totals") or {}
        tax = totals.get("tax") or {}
        checks = [
            _check_line_items(items),
            _check_items_sum(items, totals.get("net"), totals.get("discount")),
            _check_tax(items, totals.get("net"), tax.get("rate"), tax.get("amount")),
            _check_gross(totals.get("net"), tax.get("amount"), totals.get("gross"),
                         totals.get("discount")),
        ]
        reports.append(SubdocReport(
            index=i,
            number=s.get("number"),
            currency=s.get("currency"),
            verdict=_rollup([c.verdict for c in checks]),
            checks=checks,
        ))
    return SanityReport(
        verdict=_rollup([r.verdict for r in reports]),
        subdoc_count=len(reports),
        subdocs=reports,
    )

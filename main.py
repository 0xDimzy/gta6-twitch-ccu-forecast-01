"""gta6-twitch-ccu-forecast-01 entry: scale-conditioned rolling-midnight
launch-day forecaster (Python 3.13, standard library only).

Method (fitted at run time from the case file itself -- nothing is hard-coded
to any launch, and no practice answer is memorised):

1.  Take the case's comparables (past big launches, hourly viewers from 14
    days before to 3 days after their release day).
2.  Keep only the launches that unlocked at midnight local time, rolling
    across zones (the way console launches, GTA VI included, unlock), and
    that are not flagged early-access-contaminated or weak in their unlock
    note.  Simultaneous-global and early-access launches have a different
    day shape and would flatten the forecast.
3.  For each kept launch and each forecast hour h = 2..24, form the log
    ratio r = ln(viewers_h / viewers_1) to that launch's own hour-1 print.
4.  Per hour, regress r on ln(hour-1 viewers) across the kept launches:
    bigger launches ramp less in relative terms (the largest observed
    hour-1 prints barely double by the evening peak, the smallest grow
    ~8x).  Predict the case's ratio from its own hour-1 print, with the
    print's log clamped to the range the kept launches span, so a launch
    bigger than anything observed takes the shape of the biggest observed
    launches instead of extrapolating a fitted slope past the data.
5.  Quantiles from the regression residuals: q_tau = p1 * exp(med + z_tau *
    sigma), sigma the residual spread, shrunk x0.8 (the raw residual spread
    is wider than the pinball-optimal spread on the practice cases) and
    pooled halfway toward the median spread across hours (each hour's
    regression rests on only ~8 launches, so its own residual spread is
    noisy; pooling stabilises it).

Fallbacks: if fewer than 8 usable launches survive the filter for an hour,
that hour falls back to the house persistence shape (spread widening like
a random walk, sigma 0.15 per root-hour).  Output is exactly 23 rows,
hours 2..24, five non-decreasing quantiles each, deterministic: two runs
on the same case write the same bytes.

Run: python3 -P -s /bot/main.py  (or MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
outside the grader's container).
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Random-walk spread of the persistence fallback, log units per root-hour.
FALLBACK_SIGMA = 0.15
# Residual spread is wider than pinball-optimal; shrink it.
SPREAD_SHRINK = 0.8
# Each hour's residual spread is pooled halfway toward the median spread
# across hours: with ~8 launches per regression, a single hour's spread
# estimate is noisy. 0.0 = fully pooled, 1.0 = per-hour only.
SPREAD_POOL_WEIGHT = 0.5
# Minimum usable launches for the per-hour regression.
MIN_LAUNCHES = 8


def round2(x):
    """x rounded to 2 decimals, half up (matches the house starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def is_rolling_midnight(comp):
    """True when the launch unlocked at midnight local time, rolling, and
    its unlock note does not flag it early-access-contaminated or weak."""
    note = (comp.get("unlock") or "")
    low = note.lower()
    if "tier: early-access-contaminated" in low or "tier: weak" in low:
        return False
    if "midnight local" not in low and "rolling" not in low:
        return False
    head = low.split("tier:")[0]
    if "single global" in head or "simultaneous" in head[:120]:
        return False
    return True


def release_day_hours(comp):
    """{case hour 1..24: viewers} for a comparable's release day."""
    try:
        day_start = comp["day_start"]
        ds = _parse_time(day_start)
    except (KeyError, ValueError):
        return {}
    out = {}
    for pt in comp.get("hourly") or []:
        if pt.get("viewers") is None:
            continue
        try:
            at = _parse_time(pt["from"])
        except ValueError:
            continue
        delta = (at - ds).total_seconds() / 3600.0
        hour = int(delta) + 1
        if 1 <= hour <= 24 and delta == int(delta):
            out[hour] = max(1, pt["viewers"])
    return out


def _parse_time(s):
    """ISO 8601 with a trailing Z, to an aware datetime."""
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def fit_hour(pairs):
    """Ordinary least squares of log-ratio on log hour-1 viewers.

    pairs: [(log_p1, log_ratio)].  Returns (mean_x, mean_y, slope, sigma).
    """
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    slope = 0.0
    if sxx > 0:
        slope = sum((p[0] - mx) * (p[1] - my) for p in pairs) / sxx
    resid = [p[1] - (my + slope * (p[0] - mx)) for p in pairs]
    sigma = statistics.pstdev(resid) if n > 1 else 0.3
    return mx, my, slope, max(0.08, sigma)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case["levels"], LEVELS))
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Per-hour training pairs from the kept comparables.
    pairs = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables") or []:
        if not is_rolling_midnight(comp):
            continue
        day = release_day_hours(comp)
        if 1 not in day:
            continue
        log_base = math.log(day[1])
        for h in range(2, 25):
            if h in day:
                pairs[h].append((log_base, math.log(day[h] / day[1])))

    hours = []
    # First pass: fit every hour, keeping (median log-ratio, residual
    # spread); hours with too few usable launches fall back to persistence.
    fitted = {}
    for row in case["forecast"]:
        h = row["hour"]
        pts = pairs[h]
        if len(pts) < MIN_LAUNCHES:
            fitted[h] = None
            continue
        xs = [p[0] for p in pts]
        lo, hi = min(xs), max(xs)
        # No extrapolation past the observed scale: a launch bigger
        # than anything in the set takes the biggest launches' shape.
        eff = min(max(log_p1, lo), hi)
        mx, my, slope, sigma = fit_hour(pts)
        fitted[h] = (my + slope * (eff - mx), sigma)
    # Pool the spreads: the median residual spread across fitted hours.
    spreads = [v[1] for v in fitted.values() if v is not None]
    pooled = statistics.median(spreads) if spreads else 0.3
    for row in case["forecast"]:
        h = row["hour"]
        fit = fitted[h]
        if fit is None:
            qs = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)))
                  for z in Z]
        else:
            med, sigma = fit
            sig = (SPREAD_POOL_WEIGHT * sigma
                   + (1 - SPREAD_POOL_WEIGHT) * pooled) * SPREAD_SHRINK
            qs = [round2(p1 * math.exp(med + z * sig)) for z in Z]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""REST API for the GOAT scoring engine.

Run:
    uvicorn goat.api:app --reload
Then open http://127.0.0.1:8000/docs for interactive docs.

Endpoints
    GET  /athletes                         who can be scored (optionally ?sport=)
    GET  /athletes/{name}                  one athlete: dimensions + every season
    GET  /rankings                         leaderboard with confidence ranges
    GET  /compare?a=LeBron James&b=Pele    head to head, incl. P(a beats b)
    GET  /presets                          named weight settings
    POST /refresh                          rebuild data from the sources

Weights: pass ?preset=comparable_only, and/or override single dimensions with
?weights=winning:2,longevity:0.5
"""

import copy
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from .config import CONFIG, DIMENSIONS, PRESETS
from .data import load, normalize_name
from .scoring import (athlete_dimensions, bootstrap, prob_beats, scale_and_total, season_table,
                      summarize)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / "data" / "nba", ROOT / "data" / "soccer"]
DEFAULT_BOOT = 200

app = FastAPI(title="GOAT Debate API",
              description="Cross-sport, cross-era greatness scores with confidence ranges.",
              version="0.1.0")


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _row(series):
    return {k: _clean(v) for k, v in series.items()}


@lru_cache(maxsize=1)
def _data():
    dirs = [d for d in DATA_DIRS if d.is_dir()]
    if not dirs:
        raise HTTPException(503, "No data built yet. POST /refresh or run the source builders.")
    return load(dirs)


def _cfg(preset: str, weights: Optional[str]) -> dict:
    if preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset {preset!r}. Choose from {sorted(PRESETS)}")
    cfg = copy.deepcopy(CONFIG)
    cfg["weights"] = dict(PRESETS[preset])
    for part in filter(None, (weights or "").split(",")):
        try:
            k, v = part.split(":")
            v = float(v)
        except ValueError:
            raise HTTPException(400, f"Bad weight {part!r}; use name:value, e.g. winning:2")
        if k not in DIMENSIONS:
            raise HTTPException(400, f"Unknown dimension {k!r}. Choose from {DIMENSIONS}")
        cfg["weights"][k] = v
    if sum(cfg["weights"].values()) <= 0:
        raise HTTPException(400, "Weights must add up to more than 0")
    return cfg


@lru_cache(maxsize=32)
def _scored(weights_key: tuple, boot: int):
    cfg = copy.deepcopy(CONFIG)
    cfg["weights"] = dict(weights_key)
    table = season_table(_data(), cfg)
    point = scale_and_total(athlete_dimensions(table, cfg), cfg).sort_values("total", ascending=False)
    boots = bootstrap(table, cfg, n_boot=boot)
    return table, summarize(point, boots), boots


def _run(preset, weights, boot):
    cfg = _cfg(preset, weights)
    return _scored(tuple(sorted(cfg["weights"].items())), boot), cfg


def _resolve(name: str, index) -> str:
    lookup = {normalize_name(a): a for a in index}
    key = normalize_name(name)
    if key in lookup:
        return lookup[key]
    matches = [a for k, a in lookup.items() if key in k]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise HTTPException(404, f"No athlete matching {name!r}")
    raise HTTPException(400, f"{name!r} is ambiguous: {matches[:10]}")


@app.get("/")
def root():
    return {"name": "GOAT Debate API", "docs": "/docs",
            "examples": ["/rankings?limit=10", "/compare?a=LeBron James&b=Pele",
                         "/compare?a=LeBron James&b=Pele&preset=comparable_only"]}


@app.get("/presets")
def presets():
    return PRESETS


@app.get("/athletes")
def athletes(sport: Optional[str] = None):
    s = _data()["seasons"]
    if sport:
        s = s[s["sport"] == sport]
    g = s.groupby(["athlete", "sport"]).agg(first=("season", "min"), last=("season", "max"),
                                             seasons=("season", "nunique")).reset_index()
    return [_row(r) for _, r in g.iterrows()]


@app.get("/rankings")
def rankings(sport: Optional[str] = None, preset: str = "default", weights: Optional[str] = None,
             limit: int = Query(25, ge=1, le=500), boot: int = Query(DEFAULT_BOOT, ge=20, le=2000)):
    (_, result, _), cfg = _run(preset, weights, boot)
    if sport:
        result = result[result["sport"] == sport]
    cols = ["sport", "total", "low_90", "high_90", "p_rank_1"] + [f"{d}_10" for d in DIMENSIONS]
    return {"weights": cfg["weights"], "bootstrap_runs": boot,
            "rankings": [{"rank": i + 1, "athlete": name, **_row(r[cols])}
                         for i, (name, r) in enumerate(result.head(limit).iterrows())]}


@app.get("/athletes/{name}")
def athlete(name: str, preset: str = "default", weights: Optional[str] = None,
            boot: int = Query(DEFAULT_BOOT, ge=20, le=2000)):
    (table, result, _), cfg = _run(preset, weights, boot)
    who = _resolve(name, result.index)
    r = result.loc[who]
    seasons = table[table["athlete"] == who].sort_values("season")
    honors = _data()["honors"]
    return {
        "athlete": who, "sport": r["sport"], "rank": int(r["rank"]), "weights": cfg["weights"],
        "total": _clean(r["total"]), "range_90": [_clean(r["low_90"]), _clean(r["high_90"])],
        "scores_0_to_10": {d: _clean(r[f"{d}_10"]) for d in DIMENSIONS},
        "raw": {k: _clean(r[k]) for k in DIMENSIONS + ["competition", "mean_z", "seasons_scored"]},
        "seasons": [_row(s[["season", "league", "z", "tail", "level", "winning", "recognition",
                             "big_stage"]]) for _, s in seasons.iterrows()],
        "honors": honors[honors["athlete"] == who][["season", "honor", "category"]].to_dict("records"),
    }


@app.get("/compare")
def compare(a: str, b: str, preset: str = "default", weights: Optional[str] = None,
            boot: int = Query(DEFAULT_BOOT, ge=20, le=2000)):
    (_, result, boots), cfg = _run(preset, weights, boot)
    wa, wb = _resolve(a, result.index), _resolve(b, result.index)
    ra, rb = result.loc[wa], result.loc[wb]
    p = prob_beats(boots, wa, wb)
    return {
        "weights": cfg["weights"],
        "a": {"athlete": wa, "sport": ra["sport"], "total": _clean(ra["total"]),
              "range_90": [_clean(ra["low_90"]), _clean(ra["high_90"])]},
        "b": {"athlete": wb, "sport": rb["sport"], "total": _clean(rb["total"]),
              "range_90": [_clean(rb["low_90"]), _clean(rb["high_90"])]},
        "by_dimension": {d: {wa: _clean(ra[f"{d}_10"]), wb: _clean(rb[f"{d}_10"])} for d in DIMENSIONS},
        "p_a_beats_b": round(p, 3),
        "verdict": f"{wa if p >= 0.5 else wb} rates higher in {max(p, 1 - p):.0%} of {boot} runs",
    }


@app.post("/refresh")
def refresh(source: str = Query("all", pattern="^(all|nba|soccer)$")):
    from .sources import nba, soccer
    if source in ("all", "nba"):
        nba.download()
        nba.build()
    if source in ("all", "soccer"):
        soccer.build()
    _data.cache_clear()
    _scored.cache_clear()
    return {"rebuilt": source}

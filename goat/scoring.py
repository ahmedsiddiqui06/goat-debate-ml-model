"""The scoring engine.

Pipeline:
  1. season_table: for every athlete-season,
       z      how far above their league's top-N scorers they were (robust z)
       tail   how rare that z is: -ln(P(T > z)), heavy-tailed Student-t
       level  how hard the league is to stand out in: -ln(top_N / talent_pool)
     plus the rarity of any honors won that season.
     A season's difficulty-adjusted value is level + tail, i.e. roughly
     "how rare is this season among everyone who was competing for it".
  2. athlete_dimensions: roll seasons up into six career dimensions.
  3. scale_and_total: put every dimension on a 0-10 scale across all athletes
     being compared (both sports together) and take a weighted total.
  4. bootstrap: resample each athlete's seasons AND redraw each league's
     talent pool within its low-high range, many times, to get confidence
     ranges and "P(A beats B)".
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

from .config import CONFIG, DIMENSIONS, HONOR_CATEGORIES


def robust_z(values: np.ndarray, x: float) -> float:
    """Distance of x from the median of `values`, in robust standard deviations."""
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    scale = np.median(np.abs(values - med)) * 1.4826  # MAD scaled to match a std
    if not scale > 0:
        scale = values.std()
    if not scale > 0:
        return np.nan
    return (x - med) / scale


def honor_rarity(winners, pool_size):
    """How rare an honor is, in 'nats': -ln(winners / pool_size).

    1 MVP out of 450 players  -> 6.1
    1 title out of 30 teams   -> 3.4
    """
    return -np.log(np.asarray(winners, float) / np.asarray(pool_size, float))


def tail_rarity(z, df: float = CONFIG["tail_df"]):
    """Rarity of being z std devs above the elite median: -ln(P(T > z)).

    Uses a heavy-tailed Student-t curve so extreme z-scores (often from short
    peer lists) grow in rarity gradually instead of exploding.
    """
    return -student_t.logsf(np.asarray(z, float), df)


def league_level(talent_pool, elite_n):
    """Difficulty of reaching a league's top-N: -ln(top_N / talent_pool).

    With top_N = 10:
    A school league of 200 players -> ln(200/10) = 3.0
    A league drawing on 5 million  -> ln(5e6/10) = 13.1
    """
    return np.log(np.maximum(np.asarray(talent_pool, float), elite_n) / elite_n)


def _elite_pools(pools: pd.DataFrame, cfg: dict) -> dict:
    """{(sport, league, season): (min games to qualify, top-N per-game scoring rates)}."""
    out = {}
    pools = pools.assign(rate=pools["scoring"] / pools["games"].where(pools["games"] > 0))
    for (sport, league, season), g in pools.groupby(["sport", "league", "season"]):
        min_games = cfg["min_games_fraction"] * g["games"].max()
        g = g[g["games"] >= min_games]
        out[(sport, league, season)] = (min_games, g["rate"].nlargest(cfg["elite_pool_size"]).to_numpy())
    return out


def _attach_levels(seasons: pd.DataFrame, leagues: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    years = pd.to_numeric(seasons["season"].str[:4], errors="coerce")
    lows, highs, missing = [], [], set()
    for (sport, league, _), yr in zip(seasons[["sport", "league", "season"]].itertuples(index=False), years):
        m = leagues[(leagues["sport"] == sport) & (leagues["league"] == league)
                    & (leagues["season_from"] <= yr) & (leagues["season_to"] >= yr)]
        if m.empty:
            missing.add(f"{sport} / {league} / {int(yr) if pd.notna(yr) else '?'}")
            lows.append(np.nan); highs.append(np.nan)
        else:
            row = m.iloc[0]
            lows.append(float(league_level(row["talent_pool_low"], cfg["elite_pool_size"])))
            highs.append(float(league_level(row["talent_pool_high"], cfg["elite_pool_size"])))
    if missing:
        raise ValueError("No talent-pool entry in leagues.csv for: " + "; ".join(sorted(missing)))
    return seasons.assign(level_low=lows, level_high=highs)


def season_table(data: dict, cfg: dict = CONFIG) -> pd.DataFrame:
    """One row per athlete-season: z, tail rarity, league level range, honor rarity."""
    seasons = data["seasons"].copy()
    elite = _elite_pools(data["pools"], cfg)

    z = []
    for r in seasons.itertuples(index=False):
        min_games, pool = elite.get((r.sport, r.league, r.season), (np.inf, np.array([])))
        if r.games < min_games or len(pool) < cfg["min_pool_size"]:
            z.append(np.nan)
        else:
            z.append(robust_z(pool, r.scoring / r.games))
    seasons["z"] = z
    seasons["tail"] = tail_rarity(seasons["z"], cfg["tail_df"])
    seasons = _attach_levels(seasons, data["leagues"], cfg)
    seasons = seasons[["athlete", "sport", "league", "season", "z", "tail", "level_low", "level_high"]]

    honors = data["honors"].copy()
    honors["rarity"] = honor_rarity(honors["winners"], honors["pool_size"])
    honors["dim"] = honors["category"].map(HONOR_CATEGORIES)
    per_season = (honors.pivot_table(index=["athlete", "sport", "season"], columns="dim",
                                     values="rarity", aggfunc="sum")
                  .reset_index())

    # A season can have stats, honors, or both (e.g. a World Cup year).
    table = seasons.merge(per_season, on=["athlete", "sport", "season"], how="outer")
    for dim in HONOR_CATEGORIES.values():
        table[dim] = table[dim].fillna(0.0) if dim in table else 0.0
    table["level"] = (table["level_low"] + table["level_high"]) / 2  # point estimate
    return table.reset_index(drop=True)


def athlete_dimensions(table: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Raw career value for each dimension (units differ by dimension)."""
    table = table.assign(adjusted=table["level"] + table["tail"])
    rows = []
    for (athlete, sport), g in table.groupby(["athlete", "sport"], sort=False):
        adj = g["adjusted"].dropna()
        zs = g["z"].dropna()
        rows.append({
            "athlete": athlete,
            "sport": sport,
            "dominance": adj.mean() if len(adj) else np.nan,
            "peak": adj.nlargest(cfg["peak_seasons"]).mean() if len(adj) else np.nan,
            "longevity": float((zs >= cfg["longevity_threshold"]).sum()),
            "winning": g["winning"].sum(),
            "recognition": g["recognition"].sum(),
            "big_stage": g["big_stage"].sum(),
            "competition": g["level"].mean(),
            "mean_z": zs.mean() if len(zs) else np.nan,
            "seasons_scored": len(zs),
        })
    return pd.DataFrame(rows).set_index("athlete")


def scale_and_total(dims: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Scale each dimension 0-10 by rank among all athletes, then weight."""
    out = dims.copy()
    n = len(out)
    for d in DIMENSIONS:
        vals = out[d].fillna(out[d].min() if out[d].notna().any() else 0.0)
        out[f"{d}_10"] = 5.0 if n < 2 else (vals.rank(method="average") - 1) / (n - 1) * 10
    w = cfg["weights"]
    total_w = sum(w[d] for d in DIMENSIONS)
    out["total"] = sum(out[f"{d}_10"] * w[d] for d in DIMENSIONS) / total_w
    return out


def score(data: dict, cfg: dict = CONFIG) -> pd.DataFrame:
    """Point estimate: one row per athlete, sorted best first."""
    scored = scale_and_total(athlete_dimensions(season_table(data, cfg), cfg), cfg)
    return scored.sort_values("total", ascending=False)


def bootstrap(table: pd.DataFrame, cfg: dict = CONFIG, n_boot: int = None, seed: int = None):
    """Resample seasons and redraw league difficulty, then re-score.

    Returns a (n_boot x n_athletes) DataFrame of total scores. Athletes with
    few seasons, one freak year, or a league whose difficulty is uncertain
    get wide ranges, which is the point.
    """
    n_boot = n_boot or cfg["bootstrap"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    leagues = list(table["league"].dropna().unique())
    league_idx = {lg: i for i, lg in enumerate(leagues)}

    # Pre-split each athlete's seasons into plain arrays (same logic as
    # athlete_dimensions, just vectorised so 500 runs take seconds).
    names, athletes = [], []
    for (athlete, _), g in table.groupby(["athlete", "sport"], sort=False):
        names.append(athlete)
        athletes.append({
            "z": g["z"].to_numpy(float), "tail": g["tail"].to_numpy(float),
            "lo": g["level_low"].to_numpy(float), "hi": g["level_high"].to_numpy(float),
            "lg": np.array([league_idx.get(x, -1) for x in g["league"]]),
            "honors": g[["winning", "recognition", "big_stage"]].to_numpy(float),
        })

    w = np.array([cfg["weights"][d] for d in DIMENSIONS])
    k = cfg["peak_seasons"]
    out = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        # One draw per league per run, uniform between its low and high level
        # (= log-uniform between talent_pool_low and talent_pool_high).
        u = np.append(rng.random(len(leagues)), np.nan)
        dims = np.full((len(names), len(DIMENSIONS)), np.nan)
        for i, a in enumerate(athletes):
            idx = rng.integers(0, len(a["z"]), len(a["z"]))
            z, lo, hi = a["z"][idx], a["lo"][idx], a["hi"][idx]
            adj = lo + u[a["lg"][idx]] * (hi - lo) + a["tail"][idx]
            adj, zs = adj[~np.isnan(adj)], z[~np.isnan(z)]
            if len(adj):
                dims[i, 0] = adj.mean()
                dims[i, 1] = np.sort(adj)[-k:].mean()
            dims[i, 2] = (zs >= cfg["longevity_threshold"]).sum()
            dims[i, 3:] = a["honors"][idx].sum(axis=0)
        out[b] = _scale_matrix(dims) @ w / w.sum()
    return pd.DataFrame(out, columns=names)


def _scale_matrix(dims: np.ndarray) -> np.ndarray:
    """Numpy twin of the 0-10 rank scaling in scale_and_total."""
    n = dims.shape[0]
    if n < 2:
        return np.full_like(dims, 5.0)
    scaled = np.empty_like(dims)
    for j in range(dims.shape[1]):
        col = dims[:, j]
        if np.isnan(col).all():
            col = np.zeros(n)
        else:
            col = np.where(np.isnan(col), np.nanmin(col), col)
        scaled[:, j] = (rankdata(col, method="average") - 1) / (n - 1) * 10
    return scaled


def summarize(point: pd.DataFrame, boots: pd.DataFrame) -> pd.DataFrame:
    """Add confidence columns to the point estimate."""
    ranks = boots.rank(axis=1, ascending=False, method="min")
    out = point.copy()
    out["low_90"] = boots.quantile(0.05).reindex(out.index)
    out["high_90"] = boots.quantile(0.95).reindex(out.index)
    out["p_rank_1"] = (ranks == 1).mean().reindex(out.index)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def prob_beats(boots: pd.DataFrame, a: str, b: str) -> float:
    """Share of bootstrap runs where athlete a scores higher than athlete b."""
    return float((boots[a] > boots[b]).mean())

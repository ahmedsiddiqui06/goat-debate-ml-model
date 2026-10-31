"""Loading and validating the shared ("common currency") data format.

Every sport is converted into the same three CSV files:

seasons.csv  one row per athlete per season
    athlete, sport, league, season, team, games, scoring
    (scoring = total points in basketball, total goals in soccer)

pools.csv    the peers each season is compared against
    sport, league, season, player, games, scoring
    (all players, or at least the league's top scorers, for that season)

honors.csv   titles, awards and big-stage honors
    athlete, sport, season, honor, category, winners, pool_size
    category is one of: title, award, big_stage
    winners / pool_size describes how rare the honor is, e.g. an MVP is
    1 winner out of ~450 players; a league title is 1 team out of 30.

leagues.csv  how hard each league is to stand out in (competition difficulty)
    sport, league, season_from, season_to, talent_pool_low, talent_pool_high, source
    talent_pool = how many people were effectively competing for this league's
    top spots: a school conference of 200 players -> 200; a league that draws
    on every serious player in the world -> millions. Give a low-high range;
    the confidence scores account for the uncertainty.
"""

from pathlib import Path
import unicodedata

import pandas as pd

from .config import HONOR_CATEGORIES

SCHEMA = {
    "seasons": ["athlete", "sport", "league", "season", "team", "games", "scoring"],
    "pools": ["sport", "league", "season", "player", "games", "scoring"],
    "honors": ["athlete", "sport", "season", "honor", "category", "winners", "pool_size"],
    "leagues": ["sport", "league", "season_from", "season_to", "talent_pool_low",
                "talent_pool_high", "source"],
}


def normalize_name(name: str) -> str:
    """Case- and accent-insensitive key, so 'pele' matches 'Pelé'."""
    text = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _read(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA[kind])
    df = pd.read_csv(path, dtype={"season": str})
    missing = set(SCHEMA[kind]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[SCHEMA[kind]]


def load(dirs) -> dict:
    """Load and concatenate seasons/pools/honors from one or more folders."""
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    frames = {k: [] for k in SCHEMA}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"Data folder not found: {d}")
        for kind in SCHEMA:
            frames[kind].append(_read(d / f"{kind}.csv", kind))
    data = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    validate(data)
    return data


def validate(data: dict) -> None:
    s, p, h, lg = data["seasons"], data["pools"], data["honors"], data["leagues"]
    for c in ["season_from", "season_to", "talent_pool_low", "talent_pool_high"]:
        vals = pd.to_numeric(lg[c], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"leagues.{c} has non-numeric or empty values")
        lg[c] = vals
    if (lg["talent_pool_low"] > lg["talent_pool_high"]).any() or (lg["talent_pool_low"] <= 0).any():
        raise ValueError("leagues: need 0 < talent_pool_low <= talent_pool_high")
    for name, df, cols in [("seasons", s, ["games", "scoring"]), ("pools", p, ["games", "scoring"]),
                           ("honors", h, ["winners", "pool_size"])]:
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.isna().any():
                raise ValueError(f"{name}.{c} has non-numeric or empty values")
            if (vals < 0).any():
                raise ValueError(f"{name}.{c} has negative values")
            df[c] = vals
    bad = set(h["category"]) - set(HONOR_CATEGORIES)
    if bad:
        raise ValueError(f"honors.category must be one of {sorted(HONOR_CATEGORIES)}, got {sorted(bad)}")
    if ((h["winners"] <= 0) | (h["winners"] > h["pool_size"])).any():
        raise ValueError("honors: winners must be between 1 and pool_size")
    for df in (s, p, h):
        df["season"] = df["season"].astype(str)

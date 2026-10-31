"""Real soccer data for historical players, converted into the shared format.

Older soccer has no game-by-game database, so this is built from researched
tables kept in data/raw/soccer/ (every row has a source):

  paulista_topscorers.csv  each Campeonato Paulista season's top-scorer
                           table, plus schedule length, teams and champion
  pele_honors.csv          titles won at the time (no retroactive awards)

    python -m goat.sources.soccer   # builds data/soccer/

Notes on method
- Old top-scorer tables list goals, not appearances. So every player in a
  season (Pelé included) is given the full schedule as 'games', which means
  the engine compares season goal totals. That is consistent within a season.
- Seasons whose table lists fewer than CONFIG['min_pool_size'] scorers are
  kept but not scored (too few peers to judge), e.g. 1972-73.
- Pelé's 1957 season is left out: the page's table and text disagree.
  1971 and 1974 are left out: his goal totals aren't listed.
- League titles come from the 'champion' column; shared titles count as
  2 winners.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "soccer"
OUT = ROOT / "data" / "soccer"
LEAGUE = "Campeonato Paulista"

ATHLETES = {"Pelé": "Santos"}  # athlete -> club, for league-title credit

DEFAULT_LEAGUES = """sport,league,season_from,season_to,talent_pool_low,talent_pool_high,source
soccer,Campeonato Paulista,1950,1979,300000,5000000,"ESTIMATE: Brazil's player base, with the Paulista holding a large share (not all) of Brazil's best players"
"""


def build() -> None:
    tops = pd.read_csv(RAW / "paulista_topscorers.csv", dtype={"season": str})
    honors_raw = pd.read_csv(RAW / "pele_honors.csv", dtype={"season": str})

    pools = pd.DataFrame({
        "sport": "soccer", "league": LEAGUE, "season": tops["season"], "player": tops["player"],
        "games": tops["max_matches"], "scoring": tops["goals"],
    })

    seasons, honors = [], []
    for athlete, club in ATHLETES.items():
        mine = tops[tops["player"] == athlete]
        for r in mine.itertuples():
            seasons.append([athlete, "soccer", LEAGUE, r.season, r.club, r.max_matches, r.goals])
        # league titles in seasons he played
        meta = tops.drop_duplicates("season").set_index("season")
        for season in mine["season"]:
            champs = meta.loc[season, "champion"].split("|")
            if club in champs:
                honors.append([athlete, "soccer", season, "Campeonato Paulista winner", "title",
                               len(champs), int(meta.loc[season, "teams"])])

    for r in honors_raw.itertuples():
        honors.append([r.athlete, "soccer", r.season, r.honor, r.category, r.winners, r.pool_size])

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seasons, columns=["athlete", "sport", "league", "season", "team", "games", "scoring"]) \
        .to_csv(OUT / "seasons.csv", index=False)
    pools.to_csv(OUT / "pools.csv", index=False)
    pd.DataFrame(honors, columns=["athlete", "sport", "season", "honor", "category", "winners",
                                  "pool_size"]).to_csv(OUT / "honors.csv", index=False)
    if not (OUT / "leagues.csv").exists():
        (OUT / "leagues.csv").write_text(DEFAULT_LEAGUES)
    print(f"Soccer: {len(seasons)} athlete-seasons, {len(pools)} pool rows, {len(honors)} honors -> {OUT}")


if __name__ == "__main__":
    build()

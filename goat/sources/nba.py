"""Real NBA data, converted into the shared format.

Source: Basketball-Reference season data, as published in the public dataset
github.com/sumitrodatta/bball-reference-datasets (every season since 1947),
plus the list of NBA champions and Finals MVPs in nba_finals.csv.

    python -m goat.sources.nba            # download + build data/nba/
    python -m goat.sources.nba --refresh  # re-download the raw files

Seasons are labelled by the year they ended (2016 = the 2015-16 season).
Only NBA/BAA seasons are used; ABA seasons are left out for now.
"""

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nba"
OUT = ROOT / "data" / "nba"
FINALS = Path(__file__).with_name("nba_finals.csv")

BASE = "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data/"
FILES = ["Player Totals.csv", "Player Award Shares.csv", "End of Season Teams.csv",
         "Team Summaries.csv", "All-Star Selections.csv"]
LEAGUES = {"NBA", "BAA"}
LEAGUE_NAME = "NBA"  # BAA (1947-49) is treated as the same league

# Approx. players on the two Finals rosters, for Finals MVP rarity.
FINALS_ROSTER_POOL = 26


def download(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = RAW / f
        if dest.exists() and not refresh:
            continue
        print(f"Downloading {f} ...")
        urllib.request.urlretrieve(BASE + urllib.parse.quote(f), dest)


def _read(name):
    return pd.read_csv(RAW / name)


def _one_row_per_player_season(totals: pd.DataFrame) -> pd.DataFrame:
    """Traded players have one row per team plus a combined '2TM'/'TOT' row; keep the combined one."""
    totals = totals.assign(combined=totals["team"].str.endswith("TM") | (totals["team"] == "TOT"))
    totals = totals.sort_values("combined", ascending=False)
    return totals.drop_duplicates(["season", "player_id"], keep="first")


def _greats(awards: pd.DataFrame, teams: pd.DataFrame) -> set:
    """Players worth ranking: any MVP or any 1st-team All-NBA/BAA selection."""
    mvp = awards[(awards["award"] == "nba mvp") & (awards["winner"])]["player_id"]
    first = teams[teams["type"].isin(["All-NBA", "All-BAA"]) & (teams["number_tm"] == "1st")]["player_id"]
    return set(mvp) | set(first)


def build(all_players: bool = False) -> None:
    totals = _read("Player Totals.csv")
    totals = totals[totals["lg"].isin(LEAGUES)]
    awards = _read("Player Award Shares.csv")
    teams_sel = _read("End of Season Teams.csv")
    team_sum = _read("Team Summaries.csv")
    team_sum = team_sum[team_sum["lg"].isin(LEAGUES) & (team_sum["team"] != "League Average")]
    allstar = _read("All-Star Selections.csv")
    allstar = allstar[allstar["lg"].isin(LEAGUES)]
    finals = pd.read_csv(FINALS)

    per_season = _one_row_per_player_season(totals)
    n_players = per_season.groupby("season")["player_id"].nunique()
    n_teams = team_sum.groupby("season")["team"].nunique()

    # pools: every NBA player, every season
    pools = pd.DataFrame({
        "sport": "basketball", "league": LEAGUE_NAME, "season": per_season["season"].astype(str),
        "player": per_season["player"], "games": per_season["g"], "scoring": per_season["pts"],
    })

    keep = set(per_season["player_id"]) if all_players else _greats(awards, teams_sel)
    names = per_season.drop_duplicates("player_id").set_index("player_id")["player"]
    mine = per_season[per_season["player_id"].isin(keep)]
    seasons = pd.DataFrame({
        "athlete": mine["player"], "sport": "basketball", "league": LEAGUE_NAME,
        "season": mine["season"].astype(str), "team": mine["team"], "games": mine["g"],
        "scoring": mine["pts"],
    })

    honors = []

    def add(pid, season, honor, category, winners, pool):
        if pid in keep and pool and winners <= pool:
            honors.append([names[pid], "basketball", str(season), honor, category, winners, int(pool)])

    # Championships: anyone who played for the champion that season.
    for f in finals.itertuples():
        season_teams = team_sum[team_sum["season"] == f.year]
        match = season_teams[season_teams["team"].str.endswith(f.winning_team)]
        if len(match) != 1:
            raise ValueError(f"Can't identify champion {f.winning_team!r} in {f.year}")
        abbr = match.iloc[0]["abbreviation"]
        for pid in totals[(totals["season"] == f.year) & (totals["team"] == abbr)]["player_id"].unique():
            add(pid, f.year, "NBA champion", "title", 1, n_teams.get(f.year))
        if isinstance(f.finals_mvp, str) and f.finals_mvp:
            pid = names[names == f.finals_mvp].index
            if len(pid):
                add(pid[0], f.year, "Finals MVP", "big_stage", 1, FINALS_ROSTER_POOL)

    # Season awards (winners only).
    for a in awards[awards["winner"] & awards["award"].isin(["nba mvp", "nba dpoy"])].itertuples():
        label = {"nba mvp": "MVP", "nba dpoy": "Defensive Player of the Year"}[a.award]
        add(a.player_id, a.season, label, "award", 1, n_players.get(a.season))

    # All-NBA / All-Defense: 1st team = top 5, 2nd = top 10, 3rd = top 15.
    tier = {"1st": 5, "2nd": 10, "3rd": 15}
    for t in teams_sel[teams_sel["type"].isin(["All-NBA", "All-BAA", "All-Defense"])].itertuples():
        kind = "All-Defense" if t.type == "All-Defense" else "All-NBA"
        add(t.player_id, t.season, f"{kind} {t.number_tm} team", "award",
            tier[t.number_tm], n_players.get(t.season))

    # All-Star selections.
    n_allstars = allstar.groupby("season")["player_id"].nunique()
    for s in allstar.itertuples():
        add(s.player_id, s.season, "All-Star", "award", n_allstars[s.season], n_players.get(s.season))

    OUT.mkdir(parents=True, exist_ok=True)
    seasons.to_csv(OUT / "seasons.csv", index=False)
    pools.to_csv(OUT / "pools.csv", index=False)
    pd.DataFrame(honors, columns=["athlete", "sport", "season", "honor", "category", "winners",
                                  "pool_size"]).to_csv(OUT / "honors.csv", index=False)
    if not (OUT / "leagues.csv").exists():
        (OUT / "leagues.csv").write_text(DEFAULT_LEAGUES)
    print(f"NBA: {seasons['athlete'].nunique()} athletes, {len(seasons)} athlete-seasons, "
          f"{len(pools)} pool rows, {len(honors)} honors -> {OUT}")


# Competition difficulty. These are judgment calls with wide ranges, not
# measured facts: roughly how many serious players were competing for NBA
# spots in each era. Edit data/nba/leagues.csv to change them.
DEFAULT_LEAGUES = """sport,league,season_from,season_to,talent_pool_low,talent_pool_high,source
basketball,NBA,1947,1954,20000,300000,"ESTIMATE: pre-integration, US college/amateur talent only"
basketball,NBA,1955,1979,100000,1000000,"ESTIMATE: integrated, almost entirely US-born talent"
basketball,NBA,1980,1999,300000,3000000,"ESTIMATE: larger US high school/college base, first international players"
basketball,NBA,2000,2030,1000000,10000000,"ESTIMATE: global recruitment"
"""


def main():
    ap = argparse.ArgumentParser(description="Build data/nba from Basketball-Reference data")
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--all-players", action="store_true",
                    help="score every NBA player, not just MVPs / 1st-team All-NBA")
    args = ap.parse_args()
    download(args.refresh)
    build(args.all_players)


if __name__ == "__main__":
    main()

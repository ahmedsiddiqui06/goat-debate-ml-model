import copy

import numpy as np
import pandas as pd
import pytest

from goat.config import CONFIG
from goat.data import normalize_name, validate
from goat.scoring import (athlete_dimensions, bootstrap, honor_rarity, league_level, robust_z,
                          scale_and_total, season_table, tail_rarity)


def make_data(talent=(1e6, 1e6), star_rate=40.0, n_seasons=5, extra_league=None):
    """A tiny league: 30 peers per season plus athletes 'Star' and 'Average'."""
    rng = np.random.default_rng(0)
    pools, seasons = [], []
    for y in range(2000, 2000 + n_seasons):
        for k in range(30):
            pools.append(["basketball", "L", str(y), f"P{k}", 80, round(80 * rng.uniform(5, 20))])
        for name, rate in [("Star", star_rate), ("Average", 15.0)]:
            pools.append(["basketball", "L", str(y), name, 80, round(80 * rate)])
            seasons.append([name, "basketball", "L", str(y), "T", 80, round(80 * rate)])
    leagues = [["basketball", "L", 1990, 2030, talent[0], talent[1], "test"]]
    if extra_league:
        leagues.append(extra_league)
    honors = [["Star", "basketball", "2001", "MVP", "award", 1, 450]]
    data = {
        "seasons": pd.DataFrame(seasons, columns=["athlete", "sport", "league", "season", "team",
                                                  "games", "scoring"]),
        "pools": pd.DataFrame(pools, columns=["sport", "league", "season", "player", "games",
                                              "scoring"]),
        "honors": pd.DataFrame(honors, columns=["athlete", "sport", "season", "honor", "category",
                                                "winners", "pool_size"]),
        "leagues": pd.DataFrame(leagues, columns=["sport", "league", "season_from", "season_to",
                                                  "talent_pool_low", "talent_pool_high", "source"]),
    }
    validate(data)
    return data


def test_robust_z():
    assert robust_z([1, 2, 3, 4, 5], 3) == 0
    assert robust_z([1, 2, 3, 4, 5], 10) > 3
    assert np.isnan(robust_z([5, 5, 5], 7))


def test_rarity_functions():
    assert honor_rarity(1, 450) == pytest.approx(np.log(450))
    assert tail_rarity(3.0) > tail_rarity(1.0) > tail_rarity(-1.0) > 0
    # heavy tail: an extreme z grows far less than under a normal curve
    assert tail_rarity(10.0) < 15
    assert league_level(200, 10) < league_level(5e6, 10)
    assert league_level(5, 10) == 0  # pool smaller than top-N -> no difficulty credit


def test_star_beats_average():
    out = scale_and_total(athlete_dimensions(season_table(make_data())))
    assert out.loc["Star", "total"] > out.loc["Average", "total"]
    assert out.loc["Star", "recognition"] == pytest.approx(np.log(450))


def test_harder_league_scores_higher():
    """Same stats in an easier league must score lower (the 5th vs 12th grader rule)."""
    hard = athlete_dimensions(season_table(make_data(talent=(1e6, 1e6))))
    easy = athlete_dimensions(season_table(make_data(talent=(300, 300))))
    assert hard.loc["Star", "dominance"] > easy.loc["Star", "dominance"]
    assert hard.loc["Star", "mean_z"] == pytest.approx(easy.loc["Star", "mean_z"])


def test_missing_league_entry_is_an_error():
    data = make_data()
    data["leagues"] = data["leagues"].iloc[0:0]
    with pytest.raises(ValueError, match="talent-pool"):
        season_table(data)


def test_bootstrap_matches_point_estimate_when_nothing_varies():
    """With identical seasons and a fixed talent pool, every bootstrap run equals the point estimate."""
    data = make_data()
    p = data["pools"]
    p.loc[p["player"] == "Star", "scoring"] = 3200
    s = data["seasons"]
    s.loc[s["athlete"] == "Star", "scoring"] = 3200
    data["honors"] = data["honors"].iloc[0:0]
    rng = np.random.default_rng(1)  # make every peer season identical too
    for y in p["season"].unique():
        peers = p["season"].eq(y) & p["player"].str.startswith("P")
        p.loc[peers, "scoring"] = (80 * np.linspace(5, 20, peers.sum())).round()
    table = season_table(data)
    point = scale_and_total(athlete_dimensions(table))["total"]
    boots = bootstrap(table, n_boot=20)
    for name in point.index:
        assert boots[name].to_numpy() == pytest.approx(point[name])


def test_bootstrap_shape_and_weights():
    cfg = copy.deepcopy(CONFIG)
    table = season_table(make_data(talent=(1e5, 1e7)), cfg)
    boots = bootstrap(table, cfg, n_boot=50)
    assert boots.shape == (50, 2)
    assert (boots["Star"] >= boots["Average"]).mean() > 0.9


def test_validation_rejects_bad_honors():
    data = make_data()
    data["honors"].loc[0, "winners"] = 1000  # more winners than the pool
    with pytest.raises(ValueError):
        validate(data)


def test_name_normalization():
    assert normalize_name("Pelé") == normalize_name("pele") == "pele"

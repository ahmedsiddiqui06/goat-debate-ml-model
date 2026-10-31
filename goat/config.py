"""Settings for the GOAT scoring engine.

Every number here is a modelling choice, not a fact. Change them and re-run
to see how sensitive the rankings are.
"""

CONFIG = {
    # Each season, an athlete is compared against the top-N scorers (by
    # per-game rate) in their league. Using the same N in every sport and era
    # is what makes the scores comparable.
    "elite_pool_size": 10,
    # Skip seasons where we know fewer peers than this (too noisy).
    "min_pool_size": 8,
    # Peak = average of the athlete's best N season dominance scores.
    "peak_seasons": 5,
    # A season counts toward longevity if dominance z >= this value
    # (0 = at least as good as the median of the league's top-N scorers).
    "longevity_threshold": 0.0,
    # A season (athlete or peer) counts only if they played at least this
    # share of the most games anyone played in that league-season. Works for
    # 82-game NBA seasons, 30-game 1960s leagues and short school seasons alike.
    "min_games_fraction": 0.5,
    # How much each dimension counts toward the total.
    "weights": {
        "dominance": 1.0,
        "peak": 1.0,
        "longevity": 1.0,
        "winning": 1.0,
        "recognition": 1.0,
        "big_stage": 1.0,
    },
    # Rarity of a z-score uses a Student-t curve with this many degrees of
    # freedom. Heavier tails than a normal curve, so one freak season against
    # a short list of peers can't produce an absurd score.
    "tail_df": 4,
    # Bootstrap resamples used for confidence scores.
    "bootstrap": 500,
    "seed": 42,
}

DIMENSIONS = ["dominance", "peak", "longevity", "winning", "recognition", "big_stage"]

# Named weight settings. "comparable_only" drops the honor categories that
# didn't exist for every era/sport (e.g. no contemporary MVP-type award or
# official World Cup best player for Pelé), so cross-era comparisons are
# judged only on things measured the same way for everyone.
PRESETS = {
    "default": dict(CONFIG["weights"]),
    "comparable_only": {"dominance": 1.0, "peak": 1.0, "longevity": 1.0, "winning": 1.0,
                        "recognition": 0.0, "big_stage": 0.0},
    "peak_greatness": {"dominance": 1.0, "peak": 2.0, "longevity": 0.5, "winning": 1.0,
                       "recognition": 0.5, "big_stage": 1.0},
}

# Honor categories map to dimensions.
HONOR_CATEGORIES = {"title": "winning", "award": "recognition", "big_stage": "big_stage"}

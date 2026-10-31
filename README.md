# goat-debate-ml-model

Compare the greatness of athletes **across sports and eras** — e.g. LeBron James vs. Pelé — by scoring how far each one stood above their own world, then comparing those scores.

## The idea

Raw stats can't be compared across sports (points vs. goals) or eras (30 ppg in 1962 ≠ 30 ppg in 2024). So every athlete is scored on the same six **dimensions**, each measured relative to their own league and season:

| Dimension | What it measures |
|---|---|
| **Dominance** | Average season: how far above their league's top-10 scorers they were, adjusted for how hard the league is |
| **Peak** | Same, averaged over their best 5 seasons |
| **Longevity** | Number of seasons at or above the median of their league's top-10 scorers |
| **Winning** | Titles, weighted by rarity (1 of 30 teams > 1 of 2 teams) |
| **Recognition** | Awards (MVP, All-NBA, …), weighted by rarity |
| **Big stage** | Big-game individual honors (e.g. Finals MVP) |

Key ingredients:

- **Outlier scores, not raw stats.** Each season becomes a robust z-score: how many "typical gaps" above the league's elite they were.
- **Rarity, not linear points.** A z of 4 is far rarer than a z of 2, so z-scores are converted to rarity (heavy-tailed Student-t, so freak values can't explode). Honors use the same idea: rarity = −ln(winners / pool size).
- **Competition difficulty.** Dominating a school conference isn't dominating the NBA. Each league has a *talent pool* (how many people it effectively selects from), giving a difficulty level = ln(talent pool / 10). The pool is a low–high range, and the uncertainty flows into the confidence scores.
- **Confidence scores.** Bootstrap: resample each athlete's seasons and redraw league difficulty 500 times. Output: a 90% range for each score, P(ranked #1), and P(A beats B).

## Quick start

```bash
pip install -r requirements.txt

# rebuild data (optional: built data is committed)
python -m goat.sources.nba        # downloads Basketball-Reference data, builds data/nba/
python -m goat.sources.soccer     # builds data/soccer/ from researched tables

# command line
python -m goat.compare --data data/nba data/soccer --athletes "LeBron James" "Pele"
python -m goat.compare --data data/nba data/soccer --athletes "LeBron James" "Pele" --preset comparable_only
python -m goat.compare --data data/nba data/soccer --weights winning=2 longevity=0.5

# API
uvicorn goat.api:app --reload     # then open http://127.0.0.1:8000/docs

# tests
pytest
```

### API endpoints

| Endpoint | Returns |
|---|---|
| `GET /compare?a=LeBron James&b=Pele` | Head to head by dimension, totals with 90% ranges, P(a beats b) |
| `GET /rankings?limit=25&sport=basketball` | Leaderboard with confidence |
| `GET /athletes/{name}` | One athlete: scores, raw values, every season, every honor |
| `GET /athletes` | Everyone who can be scored |
| `GET /presets` | Named weight settings |
| `POST /refresh?source=nba` | Rebuild data from sources |

All scoring endpoints accept `preset=` and `weights=winning:2,longevity:0.5`.

## Data

All inputs share one format (see `goat/data.py`): `seasons.csv`, `pools.csv` (peers each season), `honors.csv`, `leagues.csv` (talent pools). Add a sport or league by producing these four files.

**NBA** (`goat/sources/nba.py`): every NBA/BAA season since 1947 from [Basketball-Reference](https://www.basketball-reference.com) via the public dataset [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets), plus champions and Finals MVPs in `goat/sources/nba_finals.csv`. Scored athletes: every MVP or 1st-team All-NBA player (116).

**Soccer** (`goat/sources/soccer.py`): Pelé's Campeonato Paulista seasons, built from each season's top-scorer table on Wikipedia (source URL on every row of `data/raw/soccer/paulista_topscorers.csv`), and his titles in `data/raw/soccer/pele_honors.csv`. Only honors awarded *at the time* are counted. Retroactive ones (e.g. FIFA's 1958 Best Young Player, picked by internet survey decades later) are excluded.

## First result: LeBron vs. Pelé

| Weights | LeBron | Pelé | P(LeBron > Pelé) |
|---|---|---|---|
| `default` (all six dimensions) | 9.5 | 7.3 | 100% |
| `comparable_only` (no recognition / big-stage honors) | 9.3 | 9.9 | 1% |

Pelé is top of the scale on dominance, peak and winning. He scores 0 on recognition and big stage because the honors didn't exist for him (no Ballon d'Or for non-Europeans, no official World Cup best-player award before 1982). So the verdict depends on whether those categories count. The `comparable_only` preset is the fairer cross-era comparison, but it has its own bias (see below).

## Known limitations (v1)

- **Dominance = scoring only.** It's the one stat available in every sport and era. It underrates defenders, rebounders and playmakers (Bill Russell ranks lower than experts would put him).
- **Winning isn't fully comparable.** A 1960s Santos player could win state, national, continental, intercontinental and world titles in one year; an NBA player can win one title a year. This favors Pelé.
- **Talent pools are estimates.** The values in `data/*/leagues.csv` are labelled judgment calls with wide ranges. Change them and see how results move.
- **Old soccer tables list goals, not appearances**, so Paulista seasons compare season goal totals. Some lists are short: seasons with fewer than 8 known scorers are not scored, and Pelé's 1957, 1971–74 seasons aren't scored (inconsistent or missing data).
- **Taça Brasil 1961–65** entrant counts are approximated at 16 (exact numbers not found).
- **Only one soccer player so far.** Adding more (Maradona, Cruyff, Di Stéfano, Messi, Ronaldo) needs the same researched tables.

## Roadmap

1. More soccer legends and leagues (league-strength ranges for Europe vs. South America).
2. Richer, sport-specific stats for modern players (assists, rebounds, xG, plus/minus).
3. **ML layer:** learn the dimension weights from award voting with a learning-to-rank model, and train a "sparse → rich" model on modern players to estimate how well common-currency stats predict full-stat greatness for legends.
4. Front end: an interactive leaderboard with weight sliders.

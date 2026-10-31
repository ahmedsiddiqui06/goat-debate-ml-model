"""Command line: rank athletes and compare any two.

Examples
  python -m goat.compare --data data/sample
  python -m goat.compare --data data/nba data/soccer --athletes "LeBron James" "Pele"
  python -m goat.compare --data data/sample --weights winning=2 longevity=0.5
"""

import argparse
import copy
import sys

from .config import CONFIG, DIMENSIONS, PRESETS
from .data import load, normalize_name
from .scoring import bootstrap, prob_beats, score, season_table, summarize


def _resolve(names, index):
    lookup = {normalize_name(a): a for a in index}
    out = []
    for n in names:
        key = normalize_name(n)
        if key not in lookup:
            matches = [a for k, a in lookup.items() if key in k]
            if len(matches) != 1:
                sys.exit(f"Athlete not found or ambiguous: {n!r}. Known: {', '.join(index)}")
            out.append(matches[0])
        else:
            out.append(lookup[key])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="GOAT debate scoring engine")
    ap.add_argument("--data", nargs="+", required=True, help="data folder(s)")
    ap.add_argument("--athletes", nargs="*", help="athletes to compare head to head")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default", help="named weight setting")
    ap.add_argument("--weights", nargs="*", default=[], help="e.g. winning=2 longevity=0.5")
    ap.add_argument("--boot", type=int, default=CONFIG["bootstrap"], help="bootstrap runs")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)

    cfg = copy.deepcopy(CONFIG)
    cfg["weights"] = dict(PRESETS[args.preset])
    for kv in args.weights:
        k, v = kv.split("=")
        if k not in DIMENSIONS:
            sys.exit(f"Unknown dimension {k!r}. Choose from {DIMENSIONS}")
        cfg["weights"][k] = float(v)

    data = load(args.data)
    table = season_table(data, cfg)
    point = score(data, cfg)
    boots = bootstrap(table, cfg, n_boot=args.boot)
    result = summarize(point, boots)

    print(f"\nLEADERBOARD  ({len(result)} athletes, {args.boot} bootstrap runs, weights {cfg['weights']})\n")
    cols = ["rank", "sport", "total", "low_90", "high_90", "p_rank_1"] + [f"{d}_10" for d in DIMENSIONS]
    show = result[cols].head(args.top).copy()
    show["p_rank_1"] = (show["p_rank_1"] * 100).round(0).astype(int).astype(str) + "%"
    print(show.round(1).to_string())

    if args.athletes:
        picks = _resolve(args.athletes, list(result.index))
        print("\nHEAD TO HEAD (0-10 per dimension)\n")
        print(result.loc[picks, [f"{d}_10" for d in DIMENSIONS] + ["total"]].T.round(1).to_string())
        print("\nRAW VALUES")
        print("  dominance/peak: difficulty-adjusted season rarity (league level + how far above its top scorers)")
        print("  competition: average league difficulty   mean_z: std devs above league's top scorers")
        print("  longevity: elite seasons   winning/recognition/big_stage: summed honor rarity\n")
        print(result.loc[picks, DIMENSIONS + ["competition", "mean_z", "seasons_scored"]].T.round(2).to_string())
        if len(picks) >= 2:
            print()
            for i, a in enumerate(picks):
                for b in picks[i + 1:]:
                    print(f"P({a} > {b}) = {prob_beats(boots, a, b):.0%}")
    return result


if __name__ == "__main__":
    main()

"""
Compute per-pitcher and per-team rolling stats from raw play-by-play data.

No data leakage: stats for a given game only include plays from prior games.

Outputs (written to data/features/):
  pitcher_stats.json   — {pitcher: {game_pk: {era, k9, bb9, whip_proxy}}}
  team_stats.json      — {team: {game_pk: {ops_proxy, avg, hr_rate}}}

Usage:
  python -m src.features.compute_player_stats
"""

import json
import os
from collections import defaultdict

FEATURES_DIR = "data/features"

PLAYS_FILES = [
    "data/historical/plays_2023.json",
    "data/historical/plays_2024.json",
    "data/historical/plays_2025.json",
    "data/full_ingestion.json",
]

HIT_EVENTS    = {"Single", "Double", "Triple", "Home Run"}
WALK_EVENTS   = {"Walk", "Intent Walk", "Hit By Pitch"}
AB_EXCLUDES   = {"Walk", "Intent Walk", "Hit By Pitch", "Sac Fly", "Sac Bunt"}
ASSUMED_IP    = 5.5   # average innings per start

PITCHER_WINDOW = 8   # last N starts
TEAM_WINDOW    = 20  # last N games


def load_plays():
    plays = []
    for path in PLAYS_FILES:
        if not os.path.exists(path):
            print(f"  Skipping missing file: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        plays.extend(data)
        print(f"  Loaded {len(data)} plays from {path}")
    plays.sort(key=lambda p: (p["date"], p["game_pk"]))
    return plays


def group_by_game(plays):
    """Group plays into {game_pk: [plays]} preserving order."""
    games = defaultdict(list)
    for p in plays:
        games[p["game_pk"]].append(p)
    return games


def compute_pitcher_game_stats(game_plays, pitcher):
    """Compute stats for one pitcher in one game."""
    pitcher_plays = [p for p in game_plays if p.get("pitcher") == pitcher]
    if not pitcher_plays:
        return None

    strikeouts   = sum(1 for p in pitcher_plays if p["event"] == "Strikeout")
    walks        = sum(1 for p in pitcher_plays if p["event"] in WALK_EVENTS)
    hits_allowed = sum(1 for p in pitcher_plays if p["event"] in HIT_EVENTS)
    runs_allowed = sum(p.get("runs_scored", 0) for p in pitcher_plays)
    batters_faced = len(pitcher_plays)

    return {
        "strikeouts":    strikeouts,
        "walks":         walks,
        "hits_allowed":  hits_allowed,
        "runs_allowed":  runs_allowed,
        "batters_faced": batters_faced,
    }


def compute_team_game_stats(game_plays, team):
    """Compute offensive stats for one team in one game."""
    team_plays = [p for p in game_plays if p.get("team") == team]
    if not team_plays:
        return None

    singles  = sum(1 for p in team_plays if p["event"] == "Single")
    doubles  = sum(1 for p in team_plays if p["event"] == "Double")
    triples  = sum(1 for p in team_plays if p["event"] == "Triple")
    home_runs= sum(1 for p in team_plays if p["event"] == "Home Run")
    walks    = sum(1 for p in team_plays if p["event"] in WALK_EVENTS)
    hits     = singles + doubles + triples + home_runs
    ab       = sum(1 for p in team_plays if p["event"] not in AB_EXCLUDES)
    runs     = sum(p.get("runs_scored", 0) for p in team_plays)

    # TB = 1B + 2×2B + 3×3B + 4×HR
    tb = singles + 2*doubles + 3*triples + 4*home_runs
    pa = ab + walks

    avg      = hits / ab if ab > 0 else 0.0
    obp      = (hits + walks) / pa if pa > 0 else 0.0
    slg      = tb / ab if ab > 0 else 0.0
    ops      = obp + slg

    return {
        "hits": hits, "hr": home_runs, "walks": walks,
        "ab": ab, "runs": runs,
        "avg": round(avg, 4), "obp": round(obp, 4),
        "slg": round(slg, 4), "ops": round(ops, 4),
    }


def rolling_pitcher_stats(history):
    """Aggregate last PITCHER_WINDOW starts into summary stats."""
    window = history[-PITCHER_WINDOW:]
    if not window:
        return {"era": 4.50, "k9": 7.5, "bb9": 3.0, "whip": 1.30}

    total_k   = sum(g["strikeouts"]    for g in window)
    total_bb  = sum(g["walks"]         for g in window)
    total_h   = sum(g["hits_allowed"]  for g in window)
    total_r   = sum(g["runs_allowed"]  for g in window)
    starts    = len(window)
    total_ip  = starts * ASSUMED_IP

    era  = (total_r  / total_ip) * 9 if total_ip > 0 else 4.50
    k9   = (total_k  / total_ip) * 9 if total_ip > 0 else 7.5
    bb9  = (total_bb / total_ip) * 9 if total_ip > 0 else 3.0
    whip = (total_bb + total_h) / total_ip if total_ip > 0 else 1.30

    return {
        "era":  round(era,  4),
        "k9":   round(k9,   4),
        "bb9":  round(bb9,  4),
        "whip": round(whip, 4),
    }


def rolling_team_stats(history):
    """Aggregate last TEAM_WINDOW games into summary offensive stats."""
    window = history[-TEAM_WINDOW:]
    if not window:
        return {"avg": 0.250, "obp": 0.320, "slg": 0.400, "ops": 0.720, "runs_per_game": 4.5}

    n = len(window)
    return {
        "avg":           round(sum(g["avg"]  for g in window) / n, 4),
        "obp":           round(sum(g["obp"]  for g in window) / n, 4),
        "slg":           round(sum(g["slg"]  for g in window) / n, 4),
        "ops":           round(sum(g["ops"]  for g in window) / n, 4),
        "runs_per_game": round(sum(g["runs"] for g in window) / n, 4),
    }


def build_stats(plays):
    """
    For each game, compute what each pitcher's and team's rolling stats
    looked like BEFORE that game started.

    Returns:
      pitcher_lookup: {game_pk: {pitcher_name: {era, k9, bb9, whip}}}
      team_lookup:    {game_pk: {team_name: {avg, obp, slg, ops, runs_per_game}}}
    """
    games_by_pk = group_by_game(plays)

    # Sort games by date to build rolling state in order
    game_order = sorted(
        games_by_pk.items(),
        key=lambda x: x[1][0]["date"]
    )

    pitcher_history = defaultdict(list)   # pitcher -> [game stats in order]
    team_history    = defaultdict(list)   # team    -> [game stats in order]

    pitcher_lookup  = {}   # game_pk -> {pitcher: rolling stats}
    team_lookup     = {}   # game_pk -> {team: rolling stats}

    for game_pk, game_plays in game_order:
        date = game_plays[0]["date"]

        # Collect pitchers and teams in this game
        pitchers = set(p["pitcher"] for p in game_plays if p.get("pitcher"))
        teams    = set(p["team"]    for p in game_plays if p.get("team"))

        # Snapshot rolling stats BEFORE updating with this game
        pitcher_lookup[game_pk] = {
            pitcher: rolling_pitcher_stats(pitcher_history[pitcher])
            for pitcher in pitchers
        }
        team_lookup[game_pk] = {
            team: rolling_team_stats(team_history[team])
            for team in teams
        }

        # Now update history with this game's stats
        for pitcher in pitchers:
            stats = compute_pitcher_game_stats(game_plays, pitcher)
            if stats:
                pitcher_history[pitcher].append(stats)

        for team in teams:
            stats = compute_team_game_stats(game_plays, team)
            if stats:
                team_history[team].append(stats)

    return pitcher_lookup, team_lookup


def run():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("Loading plays...")
    plays = load_plays()
    print(f"Total plays: {len(plays)}")

    print("Computing rolling stats...")
    pitcher_lookup, team_lookup = build_stats(plays)

    pitcher_path = f"{FEATURES_DIR}/pitcher_stats.json"
    team_path    = f"{FEATURES_DIR}/team_stats.json"

    # Convert int keys to str for JSON serialization
    with open(pitcher_path, "w") as f:
        json.dump({str(k): v for k, v in pitcher_lookup.items()}, f)
    with open(team_path, "w") as f:
        json.dump({str(k): v for k, v in team_lookup.items()}, f)

    print(f"Saved pitcher stats → {pitcher_path}")
    print(f"Saved team stats    → {team_path}")
    print(f"Games covered: {len(pitcher_lookup)}")


if __name__ == "__main__":
    run()

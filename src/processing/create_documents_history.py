import json
import os

INPUT = "data/raw/mlb_historical.json"
OUTPUT = "data/processed/documents.json"


def _winner_loser_text(game):
    linescore = game.get("linescore") or {}
    teams = linescore.get("teams") or {}
    home_runs = teams.get("home", {}).get("runs")
    away_runs = teams.get("away", {}).get("runs")

    if home_runs is None or away_runs is None:
        return "Final score was not available in the source payload."

    home_team = game["teams"]["home"]["team"]["name"]
    away_team = game["teams"]["away"]["team"]["name"]

    if home_runs > away_runs:
        return f"{home_team} beat {away_team} {home_runs}-{away_runs}."
    if away_runs > home_runs:
        return f"{away_team} beat {home_team} {away_runs}-{home_runs}."
    return f"{home_team} and {away_team} finished tied at {home_runs}-{away_runs}."

def create_documents():

    with open(INPUT) as f:
        games = json.load(f)

    documents = []

    for g in games:
        game_date = (g.get("gameDate") or "")[:10]
        season = g.get("season")
        status = g.get("status", {}).get("detailedState", "Unknown")
        home_team = g["teams"]["home"]["team"]["name"]
        away_team = g["teams"]["away"]["team"]["name"]
        venue = (g.get("venue") or {}).get("name")

        parts = [
            f"MLB regular season game on {game_date}.",
            f"{away_team} at {home_team}.",
            f"Status: {status}.",
            _winner_loser_text(g),
        ]

        if venue:
            parts.append(f"Venue: {venue}.")

        doc = {
            "text": " ".join(parts),
            "metadata": {
                "gamePk": g.get("gamePk"),
                "date": game_date,
                "season": int(season) if season else None,
                "status": status,
                "home_team": home_team,
                "away_team": away_team,
                "venue": venue,
            }
        }

        documents.append(doc)

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(documents, f)

    print("Documents:", len(documents))


if __name__ == "__main__":
    create_documents()

import json

def create_documents():
    with open("data/raw/mlb_schedule.json") as f:
        data = json.load(f)

    documents = []

    for game_day in data["dates"]:
        date = game_day["date"]

        for game in game_day["games"]:
            home = game["teams"]["home"]["team"]["name"]
            away = game["teams"]["away"]["team"]["name"]
            status = game["status"]["detailedState"]

            text = f"Game on {date}: {away} vs {home}. Status: {status}"

            documents.append({
                "text": text,
                "metadata": {
                    "date": date,
                    "home_team": home,
                    "away_team": away
                }
            })

    with open("data/clean/documents.json", "w") as f:
        json.dump(documents, f, indent=2)

    print("Documents saved:", len(documents))
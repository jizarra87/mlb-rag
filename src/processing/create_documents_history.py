import json
import os

INPUT = "data/raw/mlb_historical.json"
OUTPUT = "data/processed/documents.json"

def create_documents():

    with open(INPUT) as f:
        games = json.load(f)

    documents = []

    for g in games:

        doc = {
            "text": json.dumps(g),   # todo el game
            "metadata": {
                "gamePk": g.get("gamePk"),
                "date": g.get("gameDate"),
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"]
            }
        }

        documents.append(doc)

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(documents, f)

    print("Documents:", len(documents))


if __name__ == "__main__":
    create_documents()
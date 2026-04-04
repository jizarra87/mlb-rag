import requests
import json
import os


def ingest_historical(start_year=2015, end_year=2025):

    all_games = []

    for year in range(start_year, end_year + 1):

        print(f"Downloading season {year}")

        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}"

        response = requests.get(url)

        data = response.json()

        for day in data["dates"]:
            for game in day["games"]:
                if game["gameType"] != 'R':
                    continue
                all_games.append(game)

    os.makedirs("data/raw", exist_ok=True)

    with open("data/raw/mlb_historical.json", "w") as f:
        json.dump(all_games, f, indent=2)

    print("Total games:", len(all_games))


if __name__ == "__main__":
    ingest_historical(2015, 2025)
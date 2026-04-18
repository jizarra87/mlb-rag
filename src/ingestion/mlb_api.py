import requests
import json
import os

def mlb_api():

    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"

    response = requests.get(url)

    data = response.json()

    print("Dates:", len(data["dates"]))

    for game_day in data["dates"]:
        for game in game_day["games"]:
            home = game["teams"]["home"]["team"]["name"]
            away = game["teams"]["away"]["team"]["name"]

            print(f"{away} vs {home}")

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/mlb_schedule.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Schedule saved")

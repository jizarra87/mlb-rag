import requests
import time
import json
import os

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_schedule(start_date, end_date):
    url = f"{BASE_URL}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    response = requests.get(url)
    data = response.json()

    game_pks = []

    for game_day in data["dates"]:
        for game in game_day["games"]:
            if game['gameType'] == 'R':
                game_pks.append(game["gamePk"])

    return game_pks


def get_game_feed(game_pk, retries=3):
    url = f"{BASE_URL}.1/game/{game_pk}/feed/live"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Retry {attempt+1} for game {game_pk}: {e}")
            time.sleep(1.5 * (attempt + 1))  # backoff

    return None
   # return requests.get(url).json()


def extract_plays(feed_data, game_pk):
    plays = []

    all_plays = feed_data["liveData"]["plays"]["allPlays"]
    venue = feed_data["gameData"]["venue"]["name"]
    home_team = feed_data["gameData"]["teams"]["home"]["name"]
    away_team = feed_data["gameData"]["teams"]["away"]["name"]
    game_date = feed_data["gameData"]["datetime"]["originalDate"]

    for play in all_plays:
        try:
            event = play["result"]["event"]

            # 🔥 filtro importante
            #if event not in ["Single", "Double", "Triple", "Home Run", "Strikeout", "Walk"]:
            #    continue

            batter = play["matchup"]["batter"]["fullName"]
            pitcher = play["matchup"]["pitcher"]["fullName"]
            description = play["result"]["description"]
            rbi = play["result"].get("rbi", 0)
            runs_scored = 0
            runners = play.get("runners", [])

            for runner in play.get("runners", []):
                if runner["movement"]["end"] == "score":
                    runs_scored += 1
            inning = play["about"]["inning"]
            half = play["about"]["halfInning"]
            if play["about"]["halfInning"] == "top":
                batting_team_name = away_team
            else:
                batting_team_name = home_team
            text = (
                f"Game at {venue} on {game_date}. "
                f"Batter: {batter}. "
                f"Pitcher: {pitcher}. "
                f"Event: {event}. "
                f"Description: {description}. "
                f"Inning: {inning} ({half})."
                f"team:{batting_team_name}."
            )

            plays.append({
                "text": text,
                "batter": batter,
                "pitcher": pitcher,
                "venue": venue,
                "date": game_date,
                "event": event,
                "game_pk": game_pk,
                "team":batting_team_name,
                "rbi": rbi,
                "runs_scored": runs_scored,
                "runners": runners
            })

        except Exception:
            continue

    return plays


def run_feed_ingestion():
    print("Fetching games...")

    game_pks = get_schedule("2026-03-20", "2026-04-18")

    print(f"Games found: {len(game_pks)}")

    output_file = "data/full_ingestion.json"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_docs = json.load(f)
        print(f"Loaded existing data: {len(all_docs)} plays")
    else:
        all_docs = []
    
    processed_games = set()

    if all_docs:
        processed_games = set(doc["game_pk"] for doc in all_docs)
        print(f"Games already processed: {len(processed_games)}")

    start_time = time.time()
    errors = 0
    SAVE_EVERY = 500
    
    for i, game_pk in enumerate(game_pks):
        if game_pk in processed_games:
            continue
        try:
            feed = get_game_feed(game_pk)

            if not feed:
                errors+=1
                continue
                
            plays = extract_plays(feed, game_pk)
            all_docs.extend(plays)
            # checkpoint + save
            if len(all_docs) % SAVE_EVERY < len(plays):
                elapsed = time.time() - start_time

                print(f"""
            Checkpoint reached
            Total plays: {len(all_docs)}
            Elapsed: {round(elapsed, 2)} sec
            Errors so far: {errors}
            """)

                with open(output_file, "w") as f:
                    json.dump(all_docs, f)

                print(f"Saved checkpoint: {len(all_docs)} plays")
            if len(all_docs) % SAVE_EVERY < len(plays):
                print(f"Checkpoint: {len(all_docs)} plays processed")
            if i%20 == 0 and i!=0:
                elapsed = time.time() - start_time
                print(f"""
                Processed {i} games
                Total plays: {len(all_docs)}
                Time elapsed: {round(elapsed, 2)} sec
                """)
                print(f"Errors so far: {errors}")
            time.sleep(0.2)
        except Exception as e:
            errors+=1
            print(f"Error game {game_pk}: {e}")

    print(f"Total plays: {len(all_docs)}")
    with open(output_file, "w") as f:
        json.dump(all_docs, f)

    print("Final save completed")
    return all_docs


if __name__ == "__main__":
    docs = run_feed_ingestion()
    print(docs[:3])
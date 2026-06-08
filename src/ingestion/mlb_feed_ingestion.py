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
            time.sleep(1.5 * (attempt + 1))

    return None


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
            batting_team_name = away_team if half == "top" else home_team

            text = (
                f"Game at {venue} on {game_date}. "
                f"Batter: {batter}. "
                f"Pitcher: {pitcher}. "
                f"Event: {event}. "
                f"Description: {description}. "
                f"Inning: {inning} ({half}). "
                f"Team: {batting_team_name}."
            )

            plays.append({
                "text": text,
                "batter": batter,
                "pitcher": pitcher,
                "venue": venue,
                "date": game_date,
                "event": event,
                "game_pk": game_pk,
                "home_team": home_team,
                "away_team": away_team,
                "team": batting_team_name,
                "inning": inning,
                "half_inning": half,
                "rbi": rbi,
                "runs_scored": runs_scored,
                "runners": runners,
            })

        except Exception:
            continue

    return plays


def extract_game_summary(feed_data, game_pk):
    """Return a single game-level record with outcome, score, and starting pitchers."""
    try:
        game_data = feed_data["gameData"]
        live_data = feed_data["liveData"]

        home_team = game_data["teams"]["home"]["name"]
        away_team = game_data["teams"]["away"]["name"]
        venue = game_data["venue"]["name"]
        game_date = game_data["datetime"]["originalDate"]

        # Final score from linescore
        linescore = live_data.get("linescore", {})
        home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)

        if home_score > away_score:
            winner, loser = home_team, away_team
        elif away_score > home_score:
            winner, loser = away_team, home_team
        else:
            winner, loser = None, None  # tie / incomplete

        # Starting pitchers: first pitcher seen in top (home pitching) and bottom (away pitching)
        all_plays = live_data["plays"]["allPlays"]
        home_starter, away_starter = None, None
        for play in all_plays:
            try:
                half = play["about"]["halfInning"]
                pitcher = play["matchup"]["pitcher"]["fullName"]
                if half == "top" and home_starter is None:
                    home_starter = pitcher
                if half == "bottom" and away_starter is None:
                    away_starter = pitcher
                if home_starter and away_starter:
                    break
            except Exception:
                continue

        return {
            "game_pk": game_pk,
            "date": game_date,
            "venue": venue,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "loser": loser,
            "home_starter": home_starter,
            "away_starter": away_starter,
        }

    except Exception as e:
        print(f"Error extracting summary for game {game_pk}: {e}")
        return None


def run_feed_ingestion(start_date=None, end_date=None):
    from datetime import date
    start_date = start_date or os.getenv("INGEST_START_DATE", "2026-03-20")
    end_date = end_date or os.getenv("INGEST_END_DATE", date.today().strftime("%Y-%m-%d"))

    print(f"Fetching games from {start_date} to {end_date}...")

    game_pks = get_schedule(start_date, end_date)
    print(f"Games found: {len(game_pks)}")

    plays_file = "data/full_ingestion.json"
    summaries_file = "data/game_summaries.json"

    # Load existing plays
    if os.path.exists(plays_file):
        with open(plays_file, "r") as f:
            all_docs = json.load(f)
        print(f"Loaded existing plays: {len(all_docs)}")
    else:
        all_docs = []

    # Load existing summaries
    if os.path.exists(summaries_file):
        with open(summaries_file, "r") as f:
            all_summaries = json.load(f)
        print(f"Loaded existing summaries: {len(all_summaries)}")
    else:
        all_summaries = []

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
                errors += 1
                continue

            plays = extract_plays(feed, game_pk)
            all_docs.extend(plays)

            summary = extract_game_summary(feed, game_pk)
            if summary:
                all_summaries.append(summary)

            if len(all_docs) % SAVE_EVERY < len(plays):
                elapsed = time.time() - start_time
                print(f"Checkpoint: {len(all_docs)} plays | {round(elapsed, 2)}s | errors: {errors}")
                with open(plays_file, "w") as f:
                    json.dump(all_docs, f)
                with open(summaries_file, "w") as f:
                    json.dump(all_summaries, f)

            if i % 20 == 0 and i != 0:
                elapsed = time.time() - start_time
                print(f"Processed {i} games | {len(all_docs)} plays | {round(elapsed, 2)}s | errors: {errors}")

            time.sleep(0.2)

        except Exception as e:
            errors += 1
            print(f"Error game {game_pk}: {e}")

    print(f"Total plays: {len(all_docs)} | Total game summaries: {len(all_summaries)}")

    with open(plays_file, "w") as f:
        json.dump(all_docs, f)
    with open(summaries_file, "w") as f:
        json.dump(all_summaries, f)

    print("Final save completed")
    return all_docs, all_summaries


if __name__ == "__main__":
    docs, summaries = run_feed_ingestion()
    print(f"Sample play: {docs[0]}")
    print(f"Sample summary: {summaries[0]}")

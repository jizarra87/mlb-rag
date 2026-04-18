import os
from dotenv import load_dotenv
from collections import Counter
import re
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import unicodedata
from datetime import datetime

load_dotenv(".env.dev")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

# Embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# LLM model
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Qdrant client
client = QdrantClient(QDRANT_HOST, port=6333)


# -----------------------------
# Retrieval Layer
# -----------------------------
def retrieve(question):

    vector = embed_model.get_text_embedding(question)

    results = client.search(
    collection_name="mlb_articles",
    query_vector=vector,
    limit=15
    )       

    #for r in results:
    #    print("----")
    #    print(r.payload.get("text", "")[:300])
    return results


# -----------------------------
# Generation Layer
# -----------------------------
def generate_answer(question):
    q = question.lower()
    

    if "last game" in q or "latest game" in q:
        player = extract_player_name_llm(question)
        if player:
            date, game_pk, events = get_latest_game_by_player(player)
            full_game_events = client.scroll(
                    collection_name="mlb_articles",
                    scroll_filter={
                        "must": [
                            {"key": "game_pk", "match": {"value": game_pk}}
                        ]
                    },
                    limit=5000
                )[0]

            if not events:
                return f"No data found for {player}"

            ptype = detect_player_type(events, player)
            if ptype == "batter":
                stats = stats = compute_game_stats(events, full_game_events, player)
            else:
                stats = compute_pitcher_stats(events, full_game_events, player)
            

          
            return f"""
            Last game of {player}

            Date: {date}
            Game PK: {game_pk}


            Game stats:
            {stats}
            """
    match = re.search(r"(.+?)\s+vs\s+(.+)", question, re.IGNORECASE)        
    if match:
        batter = match.group(1).strip().title()
        pitcher = match.group(2).strip().title()

        events = get_events_player_pitcher(batter, pitcher)

        if not events:
            return "No data found."

        event_list = [e.payload["event"] for e in events]

        hits = sum(1 for e in event_list if e in ["Single", "Double", "Triple", "Home Run"])
        hr = sum(1 for e in event_list if e in ["Home Run"])
        walks = event_list.count("Walk")
        ab = sum(1 for e in event_list if e not in ["Walk", "Sac Fly", "Sac Bunt"])
        avg = hits / ab if ab > 0 else 0

        return f"{batter} vs {pitcher} → AB: {ab}, H: {hits}, BB: {walks}, AVG: {round(avg,3)}"
    
    if "how many" in question.lower() or "stats" in question.lower():
        player = extract_player_name_llm(question)
        print(player)
        events = get_all_events_for_player(player)  
        
        event_list = [e.payload["event"] for e in events]

        hits = sum(1 for e in event_list if e in ["Single", "Double", "Triple", "Home Run"])
        hr = sum(1 for e in event_list if e in ["Home Run"])
        walks = event_list.count("Walk")
        ab = sum(1 for e in event_list if e not in ["Walk", "Sac Fly", "Sac Bunt"])
        avg = hits / ab if ab > 0 else 0
        rbis = sum(e.payload.get("rbi", 0) for e in events)
        runs = sum(e.payload.get("runs_scored", 0) for e in events)
        return f"AB: {ab}, H: {hits}, BB: {walks} , AVG: {round(avg,3)}, RBI: {rbis}, R: {runs}, Hr: {hr}"
    docs = retrieve(question)

    context = ""
    if not docs:
        print("No documents found.")
        return "No data available for this query."

    print(docs[0].payload)
    for d in docs:
        context += d.payload.get("text", "") + "\n"

    prompt = f"""
You are an MLB analyst.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly using the context.
"""

    response = llm.complete(prompt)

    return response.text


# -----------------------------
# Main
# -----------------------------

def get_all_events_for_player(player_name):
    results = client.scroll(
        collection_name="mlb_articles",
        scroll_filter={
            "must": [
                {
                    "key": "batter",
                    "match": {"value": player_name}
                }
            ]
        },
        limit=1000
    )
    
    return results[0]  # points

def get_events_by_team(team_name):
    results = client.scroll(
        collection_name="mlb_articles",
        scroll_filter={
            "must": [
                {
                    "key": "team",
                    "match": {"value": team_name}
                }
            ]
        },
        limit=1000
    )
    return results[0]

def get_events_player_team(player, team):
    results = client.scroll(
        collection_name="mlb_articles",
        scroll_filter={
            "must": [
                {"key": "batter", "match": {"value": player}},
                {"key": "team", "match": {"value": team}}
            ]
        },
        limit=1000
    )
    return results[0]

def get_events_player_pitcher(player, pitcher):
    results = client.scroll(
        collection_name="mlb_articles",
        scroll_filter={
            "must": [
                {"key": "batter", "match": {"value": player}},
                {"key": "pitcher", "match": {"value": pitcher}}
            ]
        },
        limit=1000
    )
    return results[0]

def get_latest_game_full():
    # traer muchos registros
    results = client.scroll(
        collection_name="mlb_articles",
        limit=5000
    )

    points = results[0]

    # encontrar última fecha
    latest_date = max(p.payload.get("date") for p in points)

    # filtrar solo ese juego (por date + game_pk)
    latest_games = [p for p in points if p.payload.get("date") == latest_date]

    # agrupar por game_pk (por si hay varios juegos ese día)
    game_pk = latest_games[0].payload.get("game_pk")

    game_events = [p for p in latest_games if p.payload.get("game_pk") == game_pk]

    return latest_date, game_events



def normalize_name(name):
    if not name:
        return ""

    # quitar acentos
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')

    # quitar sufijos
    for suffix in [" jr.", " jr", " iii", " ii"]:
        name = name.lower().replace(suffix, "")

    return name.lower().strip()

def name_match(player, batter):
    player_norm = normalize_name(player)
    batter_norm = normalize_name(batter)

    return player_norm in batter_norm or batter_norm in player_norm


def get_latest_game_by_player(player):
    points = []
    offset = None

    while True:
        res, offset = client.scroll(
            collection_name="mlb_articles",
            limit=1000,
            offset=offset
        )

        points.extend(res)

        if offset is None:
            break

    # 👉 filtrar SOLO eventos del jugador
    filtered = [
    p for p in points
    if name_match(player, p.payload.get("batter", ""))
        or name_match(player, p.payload.get("pitcher",""))
        ]

    if not filtered:
        return None, None, []

    # 👉 agrupar juegos SOLO del jugador
    games = {}

    for p in filtered:
        game_pk = p.payload.get("game_pk")
        date = p.payload.get("date")

        if game_pk not in games:
            games[game_pk] = datetime.strptime(date, "%Y-%m-%d")

    # 👉 último juego del jugador
    latest_game_pk = max(games, key=lambda k: games[k])
    latest_date = games[latest_game_pk]

    # 👉 SOLO eventos del jugador en ese juego
    player_events = [
        p for p in filtered
        if p.payload.get("game_pk") == latest_game_pk
    ]
    return latest_date, latest_game_pk, player_events

def compute_runs_for_player(player, game_events):
    runs = 0

    for e in game_events:
        runners = e.payload.get("runners", [])

        for r in runners:
            if r.get("details", {}).get("runner", {}).get("fullName") == player:
                if r.get("movement", {}).get("end") == "score":
                    runs += 1

    return runs

def compute_game_stats(player_events, full_game_events, player):
    event_list = [e.payload["event"] for e in player_events]

    singles = event_list.count("Single")
    doubles = event_list.count("Double")
    triples = event_list.count("Triple")
    home_runs = event_list.count("Home Run")

    hits = singles + doubles + triples + home_runs
    walks = event_list.count("Walk")

    ab = sum(
        1 for e in event_list
        if e not in ["Walk", "Sac Fly", "Sac Bunt", "Hit By Pitch"]
    )

    rbis = sum(e.payload.get("rbi", 0) for e in player_events)

    # 🔥 RUNS CORRECTO
    runs = 0
    for e in full_game_events:
        for r in e.payload.get("runners", []):
            try:
                runner_name = r["details"]["runner"]["fullName"]
                end = r["movement"]["end"]

                if runner_name == player and end == "score":
                    runs += 1
            except:
                continue

    avg = hits / ab if ab > 0 else 0

    return {
        "AB": ab,
        "H": hits,
        "BB": walks,
        "AVG": round(avg, 3),
        "HR": home_runs,
        "2B": doubles,
        "3B": triples,
        "RBI": rbis,
        "R": runs
    }

def extract_player_name_llm(question):
    prompt = f"""
Extract the MLB player name from this question.

Question: "{question}"

Rules:
- Return ONLY the player name
- No explanation
- If no player found, return: NONE
"""
    response = llm.complete(prompt)
    if response.text == "NONE":
        return None


    return response.text


def extract_player_name(question):
    import re

    # busca cualquier nombre tipo "Nombre Apellido"
    match = re.search(
        r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ\.]+)+)",
        question
    )

    if match:
        return match.group(1).strip()

    return None

def detect_player_type(events, player):
    batter_count = sum(
        1 for e in events
        if name_match(player, e.payload.get("batter", ""))
    )

    pitcher_count = sum(
        1 for e in events
        if name_match(player, e.payload.get("pitcher", ""))
    )

    return "pitcher" if pitcher_count > batter_count else "batter"

def compute_pitcher_stats(player_events, full_game_events, player):
    pitcher_events = [
        e for e in full_game_events
        if name_match(player, e.payload.get("pitcher", ""))
    ]

    strikeouts = sum(
        1 for e in pitcher_events
        if e.payload.get("event") == "Strikeout"
    )

    walks = sum(
        1 for e in pitcher_events
        if e.payload.get("event") == "Walk"
    )

    hits_allowed = sum(
        1 for e in pitcher_events
        if e.payload.get("event") in ["Single", "Double", "Triple", "Home Run"]
    )

    return {
        "BF_events": len(pitcher_events),
        "K": strikeouts,
        "BB": walks,
        "H_allowed": hits_allowed
    }

if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = generate_answer(question)
    print("\nAnswer:")
    print(answer)
    
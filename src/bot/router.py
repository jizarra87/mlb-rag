"""
LLM-based message router for the MLB bot.

Sends the user message to GPT-4o-mini with a system prompt that describes
all available actions. Returns a structured dict with intent + extracted
entities, replacing all regex-based routing in telegram_bot.py.

Intents:
  win_prediction  - who wins a game
  f5_total        - first-5-innings run total / over-under
  player_question - stats, history, matchup for a specific player
  general         - anything else (falls through to RAG)
"""

import json
import os

from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


SYSTEM_PROMPT = """You are a routing assistant for an MLB baseball analytics bot.

Given a user message, classify the intent and extract entities.

Available intents:
- win_prediction: user wants to know which team will win a game
- f5_total: user wants to know total runs in the first 5 innings, or an over/under line for F5
- player_question: user asks about a specific player's stats, history, or a player vs player matchup
- general: anything else (rules, history, general baseball questions)

For player_question, also set sub_intent:
- last_game: user asks what happened or how a player performed in their most recent game
  (keywords in any language: "ultimo juego", "last game", "last night", "anoche", "hoy", "today")
- season_stats: user asks for cumulative stats this season
  (keywords: "this season", "esta temporada", "how many", "cuantos", "batting average", "home runs")
- vs_matchup: user asks about one player's historical record against another player
  (pattern: "Player A vs Player B", "A contra B")
- vs_handedness: user asks how a batter performs against left-handed or right-handed pitchers
  (pattern: "Contreras vs lefties", "como batea contra zurdos/derechos", "vs LHP/RHP")
- roster_vs_pitcher: user asks how a whole team's roster performs against a specific pitcher
  (pattern: "Dodgers roster vs Skenes", "how does the Yankees lineup do against Cole")

Entity extraction rules:
- Normalize team names to full MLB names (e.g. "Yankees" -> "New York Yankees", "cerveceros" -> "Milwaukee Brewers")
- Normalize player names: always return the full first and last name, no initials or abbreviations
  (e.g. "J. Taillon" -> "Jameson Taillon", "G. Cole" -> "Gerrit Cole", "C. Sanchez" -> "Cristopher Sanchez")
  Remove accents (e.g. "Ronald Acuna" not "Ronald Acuña")
- For vs_matchup, put batter in "player" and pitcher in "pitcher"
- For roster_vs_pitcher, put the full team name in "home_team" and pitcher in "pitcher"
- home_team is the first team mentioned, away_team is the second
- Extract starting pitcher names if mentioned alongside a team
- Extract a numeric line (e.g. 4.5) for f5_total if present
- If both win_prediction and f5_total apply, prefer f5_total

Respond ONLY with valid JSON, no explanation. Schema:
{
  "intent": "win_prediction" | "f5_total" | "player_question" | "general",
  "sub_intent": "last_game" | "season_stats" | "vs_matchup" | "roster_vs_pitcher" | "vs_handedness" | null,
  "home_team": "<full team name or null>",
  "away_team": "<full team name or null>",
  "home_starter": "<pitcher full name or null>",
  "away_starter": "<pitcher full name or null>",
  "player": "<player full name or null>",
  "pitcher": "<pitcher full name for vs_matchup or null>",
  "line": <number or null>
}"""


def route(message: str) -> dict:
    """
    Route a user message to the correct handler.
    Returns a dict with 'intent' and optional entity fields.
    Falls back to general intent on any error.
    """
    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": message},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        result.setdefault("intent", "general")
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Router error: {e}")
        return {"intent": "general"}

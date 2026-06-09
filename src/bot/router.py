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

Entity extraction rules:
- Normalize team names to full MLB names (e.g. "Yankees" -> "New York Yankees", "Sox" -> "Boston Red Sox")
- Normalize player names: remove accents (e.g. "Ronald Acuna" not "Ronald Acuña"), use full name
- If a player changed teams, still return their name as-is (the system will handle team lookup)
- home_team is the first team mentioned, away_team is the second
- Extract starting pitcher names if mentioned alongside a team
- Extract a numeric line (e.g. 4.5) for f5_total if present
- If both win_prediction and f5_total apply, prefer f5_total

Respond ONLY with valid JSON, no explanation. Schema:
{
  "intent": "win_prediction" | "f5_total" | "player_question" | "general",
  "home_team": "<full team name or null>",
  "away_team": "<full team name or null>",
  "home_starter": "<pitcher full name or null>",
  "away_starter": "<pitcher full name or null>",
  "player": "<player full name or null>",
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

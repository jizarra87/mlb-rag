"""
MLB Telegram Bot

Routes incoming messages via LLM router to:
  - win_prediction  : win probability model
  - f5_total        : F5 run total model
  - player_question : RAG query engine
  - general         : RAG query engine

Requires TELEGRAM_BOT_TOKEN and OPENAI_API_KEY in environment.
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from src.bot.router import route
from src.models.predict import predict
from src.models.predict_f5 import predict_f5
from src.rag.query_engine import (
    generate_answer,
    get_latest_game_by_player,
    get_all_events_for_player,
    get_events_player_pitcher,
    compute_game_stats,
    compute_pitcher_stats,
    detect_player_type,
)

load_dotenv(".env.dev")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HELP_TEXT = (
    "⚾ *MLB Analytics Bot — Help*\n\n"

    "*Team order matters*\n"
    "Always mention the *home team first*, visitor second:\n"
    "`Home Team vs Visitor Team`\n\n"

    "*Win prediction*\n"
    "Ask which team will win a game:\n"
    "• `Who wins Yankees vs Red Sox?`\n"
    "• `Will the Dodgers beat the Astros tonight?`\n"
    "• `Yankees with Gerrit Cole vs Red Sox with Brayan Bello`\n\n"

    "*F5 run total*\n"
    "Predict runs in the first 5 innings:\n"
    "• `Total runs Yankees vs Red Sox`\n"
    "• `F5 over/under 4.5 Dodgers vs Astros`\n"
    "• `How many runs in the first 5, Blue Jays with Corbin vs Phillies with Sanchez?`\n\n"

    "*Player stats & history*\n"
    "Ask about a player's stats or recent games:\n"
    "• `How many home runs does Ronald Acuña have?`\n"
    "• `What happened in Judge's last game?`\n"
    "• `Luis Arraez vs Gerrit Cole`\n"
    "• `Dodgers roster stats vs Paul Skenes`\n"
    "• `Como batea el lineup de los Yankees contra Gerrit Cole?`\n\n"

    "*General baseball questions*\n"
    "• `What is the DH rule?`\n"
    "• `Who has the most Cy Young awards?`\n\n"

    "_Powered by RAG + ML prediction models._\n"
    "_Use /help to see this message again._"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚾ Welcome to the MLB Analytics Bot!\n\n"
        "Ask me anything about baseball — win predictions, run totals, player stats, and more.\n\n"
        "Type /help to see all supported question types with examples.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


def _handle_last_game(player: str) -> str:
    from src.rag.query_engine import _load_plays, name_match
    plays = _load_plays()

    matched = [
        p for p in plays
        if name_match(player, p.get("batter", ""))
        or name_match(player, p.get("pitcher", ""))
    ]
    if not matched:
        return f"No recent game data found for *{player}*."

    latest = max(matched, key=lambda p: p.get("date", ""))
    latest_pk = latest["game_pk"]
    latest_date = latest["date"]

    player_plays  = [p for p in matched        if p["game_pk"] == latest_pk]
    full_game     = [p for p in plays           if p["game_pk"] == latest_pk]

    # Determine batter vs pitcher by majority role
    as_batter  = [p for p in player_plays if name_match(player, p.get("batter",  ""))]
    as_pitcher = [p for p in player_plays if name_match(player, p.get("pitcher", ""))]
    is_batter  = len(as_batter) >= len(as_pitcher)

    if is_batter:
        events  = [p["event"] for p in as_batter]
        hits    = sum(1 for e in events if e in ["Single", "Double", "Triple", "Home Run"])
        hr      = events.count("Home Run")
        doubles = events.count("Double")
        triples = events.count("Triple")
        bb      = events.count("Walk")
        ab      = sum(1 for e in events if e not in ["Walk", "Sac Fly", "Sac Bunt", "Hit By Pitch"])
        avg     = round(hits / ab, 3) if ab > 0 else 0.000
        rbis    = sum(p.get("rbi", 0) for p in as_batter)
        # Runs scored: check runners movements in full game
        runs = sum(
            1 for p in full_game
            for r in (p.get("runners") or [])
            if isinstance(r, dict)
            and r.get("movement", {}).get("end") == "score"
            and r.get("details", {}).get("runner", {}).get("fullName", "") == latest.get("batter", "")
        )
        stats_str = f"AB: {ab}  H: {hits}  2B: {doubles}  3B: {triples}  HR: {hr}  BB: {bb}  RBI: {rbis}  R: {runs}  AVG: {avg}"
    else:
        ip_outs  = len(as_pitcher)
        er       = sum(p.get("runs_scored", 0) for p in as_pitcher)
        ks       = sum(1 for p in as_pitcher if p.get("event") == "Strikeout")
        bb_p     = sum(1 for p in as_pitcher if p.get("event") == "Walk")
        hits_all = sum(1 for p in as_pitcher if p.get("event") in ["Single", "Double", "Triple", "Home Run"])
        ip       = round(ip_outs / 3, 1)
        stats_str = f"IP: {ip}  H: {hits_all}  ER: {er}  K: {ks}  BB: {bb_p}"

    return f"*{player} — Last Game ({latest_date})*\n\n  {stats_str}"


def _handle_roster_vs_pitcher(team: str, pitcher: str) -> str:
    from src.rag.query_engine import _load_plays, name_match
    from collections import defaultdict
    plays = _load_plays()

    pitcher_plays = [p for p in plays if name_match(pitcher, p.get("pitcher", ""))]
    team_plays = [
        p for p in pitcher_plays
        if team.lower() in p.get("team", "").lower()
        or team.lower() in p.get("home_team", "").lower()
        or team.lower() in p.get("away_team", "").lower()
    ]

    if not team_plays:
        return f"No data found for *{team}* batters vs *{pitcher}*."

    stats = defaultdict(lambda: {"ab": 0, "h": 0, "hr": 0, "bb": 0, "rbi": 0})
    for p in team_plays:
        batter = p.get("batter", "Unknown")
        event  = p.get("event", "")
        if event not in ["Walk", "Sac Fly", "Sac Bunt", "Hit By Pitch"]:
            stats[batter]["ab"] += 1
        else:
            stats[batter]["bb"] += (1 if event == "Walk" else 0)
        if event in ["Single", "Double", "Triple", "Home Run"]:
            stats[batter]["h"] += 1
        if event == "Home Run":
            stats[batter]["hr"] += 1
        stats[batter]["rbi"] += p.get("rbi", 0)

    # Sort by AB descending, minimum 2 PA
    rows = [(b, s) for b, s in stats.items() if s["ab"] + s["bb"] >= 2]
    rows.sort(key=lambda x: x[1]["ab"], reverse=True)

    if not rows:
        return f"Not enough plate appearances for *{team}* batters vs *{pitcher}*."

    lines = [f"*{team} vs {pitcher}*\n_(all available seasons)_\n"]
    lines.append(f"{'Batter':<22} {'PA':>3} {'AB':>3} {'H':>3} {'HR':>3} {'BB':>3} {'RBI':>3} {'AVG':>5}")
    lines.append("-" * 50)
    for batter, s in rows:
        pa  = s["ab"] + s["bb"]
        avg = f"{s['h']/s['ab']:.3f}" if s["ab"] > 0 else ".000"
        lines.append(f"{batter:<22} {pa:>3} {s['ab']:>3} {s['h']:>3} {s['hr']:>3} {s['bb']:>3} {s['rbi']:>3} {avg:>5}")

    return "\n".join(lines)


def _handle_season_stats(player: str) -> str:
    from src.rag.query_engine import _load_plays, name_match
    plays = _load_plays()
    # Only 2026 season (current year data)
    matched = [
        p for p in plays
        if name_match(player, p.get("batter", "")) and p.get("date", "").startswith("2026")
    ]
    if not matched:
        return f"No 2026 season data found for *{player}*."

    events = [p["event"] for p in matched]
    hits    = sum(1 for e in events if e in ["Single", "Double", "Triple", "Home Run"])
    hr      = events.count("Home Run")
    doubles = events.count("Double")
    triples = events.count("Triple")
    bb      = events.count("Walk")
    ab      = sum(1 for e in events if e not in ["Walk", "Sac Fly", "Sac Bunt", "Hit By Pitch"])
    avg     = round(hits / ab, 3) if ab > 0 else 0.000
    rbis    = sum(p.get("rbi", 0) for p in matched)

    return (
        f"*{player} — 2026 Season Stats*\n\n"
        f"  AB: {ab}  H: {hits}  2B: {doubles}  3B: {triples}  HR: {hr}  BB: {bb}  RBI: {rbis}  AVG: {avg}"
    )


def _handle_vs_matchup(batter: str, pitcher: str) -> str:
    from src.rag.query_engine import _load_plays, name_match
    plays = _load_plays()
    matched = [
        p for p in plays
        if name_match(batter,  p.get("batter",  ""))
        and name_match(pitcher, p.get("pitcher", ""))
    ]
    if not matched:
        return f"No matchup data found for *{batter}* vs *{pitcher}*."

    events = [p["event"] for p in matched]
    hits = sum(1 for e in events if e in ["Single", "Double", "Triple", "Home Run"])
    hr   = events.count("Home Run")
    doubles = events.count("Double")
    triples = events.count("Triple")
    bb   = events.count("Walk")
    ab   = sum(1 for e in events if e not in ["Walk", "Sac Fly", "Sac Bunt", "Hit By Pitch"])
    avg  = round(hits / ab, 3) if ab > 0 else 0.000
    rbis = sum(p.get("rbi", 0) for p in matched)

    return (
        f"*{batter} vs {pitcher}*\n"
        f"_(career, all available seasons)_\n\n"
        f"  AB: {ab}  H: {hits}  2B: {doubles}  3B: {triples}  HR: {hr}  BB: {bb}  RBI: {rbis}  AVG: {avg}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user = update.effective_user.first_name
    logger.info(f"Message from {user}: {text}")

    await update.message.chat.send_action("typing")

    try:
        parsed = route(text)
        intent = parsed.get("intent", "general")
        logger.info(f"Routed to: {intent} | {parsed}")

        if intent == "win_prediction":
            home_team = parsed.get("home_team")
            away_team = parsed.get("away_team")

            if home_team and away_team:
                home_starter = parsed.get("home_starter")
                away_starter = parsed.get("away_starter")
                result = predict(
                    home_team=home_team,
                    away_team=away_team,
                    home_starter=home_starter,
                    away_starter=away_starter,
                )
                home_prob = result["home_prob"] * 100
                away_prob = result["away_prob"] * 100
                winner = home_team if result["home_prob"] > result["away_prob"] else away_team
                conf = max(home_prob, away_prob)

                starters_line = ""
                if home_starter and away_starter:
                    starters_line = f"🔥 Starters: {home_starter} vs {away_starter}\n"
                elif home_starter:
                    starters_line = f"🔥 Home starter: {home_starter}\n"
                elif away_starter:
                    starters_line = f"🔥 Away starter: {away_starter}\n"

                winner_label = "home" if result["home_prob"] > result["away_prob"] else "away"
                response = (
                    f"⚾ *Win Probability*\n\n"
                    f"🏠 {home_team} (home): *{home_prob:.1f}%*\n"
                    f"✈️ {away_team} (away): *{away_prob:.1f}%*\n"
                    f"{starters_line}\n"
                    f"📊 Predicted winner: *{winner}* ({winner_label}) — {conf:.1f}% confidence\n\n"
                    f"_Based on Pythagorean win expectation, pitcher K/9, WHIP, and team OPS._"
                )
            else:
                response = generate_answer(text)

        elif intent == "f5_total":
            home_team = parsed.get("home_team")
            away_team = parsed.get("away_team")

            if home_team and away_team:
                home_starter = parsed.get("home_starter")
                away_starter = parsed.get("away_starter")
                line = parsed.get("line")
                result = predict_f5(
                    home_team=home_team,
                    away_team=away_team,
                    home_starter=home_starter,
                    away_starter=away_starter,
                    line=line,
                )
                starters_line = ""
                if home_starter and away_starter:
                    starters_line = f"🔥 Starters: {home_starter} vs {away_starter}\n"
                elif home_starter:
                    starters_line = f"🔥 Home starter: {home_starter}\n"
                elif away_starter:
                    starters_line = f"🔥 Away starter: {away_starter}\n"

                if line is not None:
                    over_pct  = result["over_prob"]  * 100
                    under_pct = result["under_prob"] * 100
                    lean = "OVER" if result["over_prob"] > result["under_prob"] else "UNDER"
                    conf = max(over_pct, under_pct)
                    ou_line = (
                        f"📈 Over  {line}: *{over_pct:.1f}%*\n"
                        f"📉 Under {line}: *{under_pct:.1f}%*\n\n"
                        f"📊 Lean: *{lean}* ({conf:.1f}% confidence)\n"
                    )
                else:
                    ou_line = ""

                response = (
                    f"⚾ *F5 Run Total*\n\n"
                    f"🏠 {home_team} vs ✈️ {away_team}\n"
                    f"{starters_line}"
                    f"🎯 Expected F5 total: *{result['expected_total']:.1f} runs*\n\n"
                    f"{ou_line}"
                    f"_Based on starter ERA/WHIP/K9 and team OPS._"
                )
            else:
                response = generate_answer(text)

        elif intent == "player_question":
            sub_intent = parsed.get("sub_intent")
            player     = parsed.get("player")

            if sub_intent == "roster_vs_pitcher":
                team    = parsed.get("home_team") or parsed.get("away_team")
                pitcher = parsed.get("pitcher")
                if team and pitcher:
                    response = _handle_roster_vs_pitcher(team, pitcher)
                else:
                    response = generate_answer(text)
            elif sub_intent == "last_game" and player:
                response = _handle_last_game(player)
            elif sub_intent == "vs_matchup" and player:
                pitcher = parsed.get("pitcher") or player
                batter  = player
                response = _handle_vs_matchup(batter, pitcher)
            elif sub_intent == "season_stats" and player:
                response = _handle_season_stats(player)
            else:
                response = generate_answer(text)

        else:
            response = generate_answer(text)

    except Exception as e:
        logger.exception(f"Error handling message: {e}")
        response = "Sorry, something went wrong. Please try again."

    await update.message.reply_text(response, parse_mode="Markdown")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot started — polling for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()

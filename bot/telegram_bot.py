"""
MLB Telegram Bot

Routes incoming messages to either:
  - Win probability prediction (team vs team questions)
  - RAG query engine (general baseball questions)

Requires TELEGRAM_BOT_TOKEN in environment.
"""

import logging
import os
import re

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from src.models.predict import predict
from src.rag.query_engine import generate_answer

load_dotenv(".env.dev")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MLB_TEAMS = {
    "arizona diamondbacks", "atlanta braves", "baltimore orioles",
    "boston red sox", "chicago cubs", "chicago white sox",
    "cincinnati reds", "cleveland guardians", "colorado rockies",
    "detroit tigers", "houston astros", "kansas city royals",
    "los angeles angels", "los angeles dodgers", "miami marlins",
    "milwaukee brewers", "minnesota twins", "new york mets",
    "new york yankees", "oakland athletics", "philadelphia phillies",
    "pittsburgh pirates", "san diego padres", "san francisco giants",
    "seattle mariners", "st. louis cardinals", "tampa bay rays",
    "texas rangers", "toronto blue jays", "washington nationals",
    # common short forms
    "yankees", "red sox", "dodgers", "astros", "braves", "mets",
    "cubs", "cardinals", "giants", "padres", "phillies", "rangers",
    "angels", "mariners", "twins", "rays", "orioles", "guardians",
    "tigers", "royals", "brewers", "pirates", "reds", "rockies",
    "marlins", "nationals", "athletics", "diamondbacks",
}

PREDICT_KEYWORDS = [
    "who wins", "who will win", "winner", "predict", "chance",
    "probability", "favored", "favourite", "favorite", "odds",
]


def is_prediction_question(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in PREDICT_KEYWORDS):
        return True
    # "vs/beat/versus" only triggers prediction if at least one side is a known team
    if "vs" in t or "beat" in t or "versus" in t:
        if any(team in t for team in MLB_TEAMS):
            return True
    # "Team with Pitcher vs Team with Pitcher" pattern
    if "with" in t and "vs" in t:
        return any(team in t for team in MLB_TEAMS)
    return False


def extract_teams_from_message(text: str):
    """
    Extract two team names from patterns like:
      'Yankees vs Red Sox'
      'Who wins Dodgers vs Astros tonight?'
      'Will the Cubs beat the Cardinals?'
    Returns (team1, team2) or (None, None).
    """
    match = re.search(r"([A-Za-z ]+?)\s+vs\.?\s+([A-Za-z ]+?)(?:,|\?|$|tonight|today|game|with|starting|\s*$)", text, re.IGNORECASE)
    if match:
        t1 = match.group(1).strip().title()
        t2 = match.group(2).strip().title()
        for word in ["The ", "Will ", "Who Wins "]:
            t1 = t1.replace(word, "")
            t2 = t2.replace(word, "")
        return t1.strip(), t2.strip()

    match = re.search(r"([A-Za-z ]+?)\s+beat\s+([A-Za-z ]+?)(?:,|\?|$)", text, re.IGNORECASE)
    if match:
        t1 = match.group(1).strip().title()
        t2 = match.group(2).strip().title()
        for word in ["The ", "Will ", "Can "]:
            t1 = t1.replace(word, "")
            t2 = t2.replace(word, "")
        return t1.strip(), t2.strip()

    return None, None


def extract_starters_from_message(text: str):
    """
    Extract starting pitcher names from patterns like:
      'Yankees vs Red Sox, Cole vs Bello'
      'Yankees with Cole vs Red Sox with Bello'
      'Yankees vs Red Sox starting Cole and Bello'
    Returns (home_starter, away_starter) or (None, None).
    """
    # Pitcher name pattern: accepts "Gerrit Cole", "C.Cole", "C Cole", "Cole"
    NAME = r"([A-Z][a-zA-Z]*\.?\s+[A-Z][a-zA-Z\-']+)"

    # "Team with Pitcher vs Team with Pitcher"
    match = re.search(rf"with\s+{NAME}\s+vs.+with\s+{NAME}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().title(), match.group(2).strip().title()

    # After comma: "Cole vs Bello" or "Cole and Bello"
    match = re.search(rf",\s*{NAME}\s+(?:vs\.?|and)\s+{NAME}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().title(), match.group(2).strip().title()

    # "starting Pitcher1 and/vs Pitcher2"
    match = re.search(rf"starting\s+{NAME}\s+(?:vs\.?|and)\s+{NAME}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().title(), match.group(2).strip().title()

    return None, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚾ MLB Analytics Bot\n\n"
        "Ask me anything about baseball:\n\n"
        "*Predictions (team vs team):*\n"
        "• Who wins Yankees vs Red Sox?\n"
        "• Will the Dodgers beat the Astros?\n"
        "• Yankees vs Red Sox, Cole vs Bello\n"
        "• Yankees with Gerrit Cole vs Red Sox with Brayan Bello\n\n"
        "*Player questions:*\n"
        "• How many HRs does Aaron Judge have?\n"
        "• What happened in Judge's last game?\n"
        "• Luis Arraez vs Gerrit Cole\n\n"
        "Powered by RAG + ML prediction model.",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user = update.effective_user.first_name
    logger.info(f"Message from {user}: {text}")

    await update.message.chat.send_action("typing")

    try:
        if is_prediction_question(text):
            home_team, away_team = extract_teams_from_message(text)

            if home_team and away_team:
                home_starter, away_starter = extract_starters_from_message(text)
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

                response = (
                    f"⚾ *Win Probability*\n\n"
                    f"🏠 {home_team} (home): *{home_prob:.1f}%*\n"
                    f"✈️ {away_team} (away): *{away_prob:.1f}%*\n"
                    f"{starters_line}\n"
                    f"📊 Predicted winner: *{winner}* ({conf:.1f}% confidence)\n\n"
                    f"_Based on Pythagorean win expectation, pitcher K/9, WHIP, and team OPS._"
                )
            else:
                # Could not parse teams — fall through to RAG
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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot started — polling for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()

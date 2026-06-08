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

# Prediction trigger keywords
PREDICT_KEYWORDS = [
    "who wins", "who will win", "winner", "predict", "chance",
    "probability", "favored", "favourite", "favorite", "odds",
    "vs", "versus", "beat",
]


def is_prediction_question(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in PREDICT_KEYWORDS)


def extract_teams_from_message(text: str):
    """
    Try to extract two team names from patterns like:
      'Yankees vs Red Sox'
      'Who wins Dodgers vs Astros tonight?'
      'Will the Cubs beat the Cardinals?'
    Returns (team1, team2) or (None, None).
    """
    # Pattern: <something> vs <something>
    match = re.search(r"([A-Za-z ]+?)\s+vs\.?\s+([A-Za-z ]+?)(?:\?|$|tonight|today|game|\s*$)", text, re.IGNORECASE)
    if match:
        t1 = match.group(1).strip().title()
        t2 = match.group(2).strip().title()
        # Remove common filler words
        for word in ["The ", "Will ", "Who Wins "]:
            t1 = t1.replace(word, "")
            t2 = t2.replace(word, "")
        return t1.strip(), t2.strip()

    # Pattern: <team> beat <team>
    match = re.search(r"([A-Za-z ]+?)\s+beat\s+([A-Za-z ]+?)(?:\?|$)", text, re.IGNORECASE)
    if match:
        t1 = match.group(1).strip().title()
        t2 = match.group(2).strip().title()
        for word in ["The ", "Will ", "Can "]:
            t1 = t1.replace(word, "")
            t2 = t2.replace(word, "")
        return t1.strip(), t2.strip()

    return None, None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚾ MLB Analytics Bot\n\n"
        "Ask me anything about baseball:\n"
        "• Win predictions: \"Who wins Yankees vs Red Sox?\"\n"
        "• Player stats: \"How many HRs does Aaron Judge have?\"\n"
        "• Game recaps: \"What happened in Judge's last game?\"\n"
        "• Matchups: \"Aaron Judge vs Gerrit Cole\"\n\n"
        "Powered by RAG + ML prediction model."
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
                result = predict(home_team=home_team, away_team=away_team)
                home_prob = result["home_prob"] * 100
                away_prob = result["away_prob"] * 100
                winner = home_team if result["home_prob"] > result["away_prob"] else away_team
                conf = max(home_prob, away_prob)

                response = (
                    f"⚾ *Win Probability*\n\n"
                    f"🏠 {home_team} (home): *{home_prob:.1f}%*\n"
                    f"✈️ {away_team} (away): *{away_prob:.1f}%*\n\n"
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

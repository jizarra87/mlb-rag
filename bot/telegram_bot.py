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
from src.rag.query_engine import generate_answer

load_dotenv(".env.dev")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HELP_TEXT = (
    "⚾ *MLB Analytics Bot — Help*\n\n"

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
    "• `Luis Arraez vs Gerrit Cole`\n\n"

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

                response = (
                    f"⚾ *Win Probability*\n\n"
                    f"🏠 {home_team} (home): *{home_prob:.1f}%*\n"
                    f"✈️ {away_team} (away): *{away_prob:.1f}%*\n"
                    f"{starters_line}\n"
                    f"📊 Predicted winner: *{winner}* ({conf:.1f}% confidence)\n\n"
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

        else:
            # player_question and general both go to RAG
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

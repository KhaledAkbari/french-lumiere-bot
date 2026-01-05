
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TelegramVegaBotOpenAI.py
========================

Webhook-based Telegram bot for Render (FastAPI + python-telegram-bot + OpenAI).

Works ONLY inside the Telegram group named exactly:
    "French Lumière"

Bot reacts ONLY when an admin replies to a message with one of:
- /reptex  : reply to replied TEXT using OpenAI (French, concise)
- /repaud  : reply to replied VOICE/AUDIO -> transcribe (Whisper) -> OpenAI (concise)
- /cortex  : correct replied TEXT (grammar/spelling) + short note
- /coraud  : transcribe replied VOICE/AUDIO then correct + short note

Environment variables (set in Render):
- BOT_TOKEN               Telegram bot token
- OPENAI_API              OpenAI API key
- RENDER_EXTERNAL_URL      provided by Render Web Service
- WEBHOOK_SECRET_TOKEN     optional security header
- OPENAI_MODEL             optional, default "gpt-4o-mini"

Health:
- GET /healthz -> "healthy"
"""

import os
import io
import html
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, Message
from telegram.constants import ChatType, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# Audio conversion (needs ffmpeg installed on the OS)
from pydub import AudioSegment


# -----------------------------
# Minimal logging (no message content)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("FrenchLumiereBot")
logging.getLogger("httpx").setLevel(logging.WARNING)  # avoid verbose request logs


# -----------------------------
# Config
# -----------------------------
GROUP_NAME = "French Lumière"

BEHAVIOR_PROMPT = (
    "You are an assistant integrated into a Telegram bot. Your role is to respond "
    "to user messages in French, in a very short, polite, and cooperative manner. "
    "Use a professional and friendly tone. Keep the response concise—maximum 200 tokens. "
    "Reply only in French."
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MAX_OUTPUT_TOKENS = 200

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").strip()  # optional override
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()  # optional

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN. Add it in Render -> Environment Variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API. Add it in Render -> Environment Variables.")

BASE_URL = WEBHOOK_BASE_URL or RENDER_EXTERNAL_URL
if not BASE_URL:
    raise RuntimeError(
        "Missing public base URL. Render should set RENDER_EXTERNAL_URL automatically. "
        "If not, set WEBHOOK_BASE_URL=https://your-app.onrender.com"
    )

WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"


# -----------------------------
# OpenAI client
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=30)


# -----------------------------
# FastAPI app + Telegram Application
# -----------------------------
app = FastAPI(title="FrenchLumiereAdminReplyBot")
telegram_app: Optional[Application] = None


# -----------------------------
# Helpers
# -----------------------------
def is_target_group(update: Update) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    return (chat.title or "").strip() == GROUP_NAME


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ("administrator", "creator")


def mention_admin_html(user) -> str:
    """Mention admin by @username if available, else clickable mention link."""
    if not user:
        return "Admin"
    if user.username:
        return f"@{html.escape(user.username)}"
    safe_name = html.escape(user.first_name or "Admin")
    return f'<a href="tg://user?id={user.id}">{safe_name}</a>'


def sender_ref(user) -> str:
    """Reference sender by @username or first name (no HTML link needed)."""
    if not user:
        return "Utilisateur"
    if user.username:
        return f"@{user.username}"
    return user.first_name or "Utilisateur"


def replied_message(update: Update) -> Optional[Message]:
    msg = update.effective_message
    if not msg:
        return None
    return msg.reply_to_message


def audio_file_id(msg: Message) -> Optional[str]:
    """Get file_id from voice or audio."""
    if msg.voice:
        return msg.voice.file_id
    if msg.audio:
        return msg.audio.file_id
    return None


async def download_file_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()


def convert_to_wav(raw_bytes: bytes) -> bytes:
    """
    Convert Telegram audio (often ogg/opus) to wav using pydub + ffmpeg.
    """
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


def header_block(admin_user, sender_user, sender_id: int, msg_id: int) -> str:
    return (
        f"👮 Admin: {mention_admin_html(admin_user)}\n"
        f"👤 Original sender: {html.escape(sender_ref(sender_user))}\n"
        f"🧾 Sender ID: <code>{sender_id}</code> | Message ID: <code>{msg_id}</code>\n\n"
    )


async def run_blocking(func, *args):
    """Run blocking OpenAI calls in a thread so we don't block async loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def openai_chat(text: str) -> str:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BEHAVIOR_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.3,
    )
    out = (completion.choices[0].message.content or "").strip()
    return out or "Désolé, je n’ai pas pu générer de réponse. Pouvez-vous reformuler ?"


def openai_transcribe(wav_bytes: bytes) -> str:
    f = io.BytesIO(wav_bytes)
    f.name = "audio.wav"
    t = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="text",
    )
    if isinstance(t, str):
        return t.strip()
    return getattr(t, "text", "").strip()


async def safe_reply(update: Update, text: str):
    if update.effective_message:
        await update.effective_message.reply_text(text)


# -----------------------------
# Command handlers (admins only, reply required)
# -----------------------------
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await safe_reply(update, "❌ Only group admins can use this command.")
        return

    rep = replied_message(update)
    if not rep or not rep.text:
        await safe_reply(update, "⚠️ Reply to a TEXT message, then use /reptex.")
        return

    admin_user = update.effective_user
    sender_user = rep.from_user
    head = header_block(admin_user, sender_user, sender_user.id if sender_user else 0, rep.message_id)

    try:
        answer = await run_blocking(openai_chat, rep.text.strip())
        await update.effective_message.reply_text(
            head + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /reptex.")
    except (RateLimitError, APITimeoutError):
        await safe_reply(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await safe_reply(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /reptex (no user content logged).")
        await safe_reply(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await safe_reply(update, "❌ Only group admins can use this command.")
        return

    rep = replied_message(update)
    if not rep:
        await safe_reply(update, "⚠️ Reply to a VOICE/AUDIO message, then use /repaud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await safe_reply(update, "⚠️ The replied message must be VOICE or AUDIO for /repaud.")
        return

    admin_user = update.effective_user
    sender_user = rep.from_user
    head = header_block(admin_user, sender_user, sender_user.id if sender_user else 0, rep.message_id)

    try:
        raw = await download_file_bytes(context, fid)
        wav = convert_to_wav(raw)  # requires ffmpeg
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await safe_reply(update, "⚠️ Could not transcribe audio. Try again.")
            return

        answer = await run_blocking(openai_chat, transcript)
        await update.effective_message.reply_text(
            head + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /repaud.")
    except FileNotFoundError:
        await safe_reply(update, "⚠️ ffmpeg is missing. Use Dockerfile deployment (recommended).")
    except (RateLimitError, APITimeoutError):
        await safe_reply(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await safe_reply(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /repaud (no user content logged).")
        await safe_reply(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await safe_reply(update, "❌ Only group admins can use this command.")
        return

    rep = replied_message(update)
    if not rep or not rep.text:
        await safe_reply(update, "⚠️ Reply to a TEXT message, then use /cortex.")
        return

    admin_user = update.effective_user
    sender_user = rep.from_user
    head = header_block(admin_user, sender_user, sender_user.id if sender_user else 0, rep.message_id)

    task = (
        "Corrige la grammaire et l’orthographe du texte ci-dessous. "
        "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
        "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
        f"TEXTE:\n{rep.text.strip()}"
    )

    try:
        answer = await run_blocking(openai_chat, task)
        await update.effective_message.reply_text(
            head + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /cortex.")
    except (RateLimitError, APITimeoutError):
        await safe_reply(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await safe_reply(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /cortex (no user content logged).")
        await safe_reply(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await safe_reply(update, "❌ Only group admins can use this command.")
        return

    rep = replied_message(update)
    if not rep:
        await safe_reply(update, "⚠️ Reply to a VOICE/AUDIO message, then use /coraud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await safe_reply(update, "⚠️ The replied message must be VOICE or AUDIO for /coraud.")
        return

    admin_user = update.effective_user
    sender_user = rep.from_user
    head = header_block(admin_user, sender_user, sender_user.id if sender_user else 0, rep.message_id)

    try:
        raw = await download_file_bytes(context, fid)
        wav = convert_to_wav(raw)
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await safe_reply(update, "⚠️ Could not transcribe audio. Try again.")
            return

        task = (
            "Voici une transcription d’un message audio. "
            "Corrige la grammaire et l’orthographe. "
            "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
            "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
            f"TRANSCRIPTION:\n{transcript}"
        )

        answer = await run_blocking(openai_chat, task)
        await update.effective_message.reply_text(
            head + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /coraud.")
    except FileNotFoundError:
        await safe_reply(update, "⚠️ ffmpeg is missing. Use Dockerfile deployment (recommended).")
    except (RateLimitError, APITimeoutError):
        await safe_reply(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await safe_reply(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /coraud (no user content logged).")
        await safe_reply(update, "⚠️ Unexpected error. Please try again later.")


# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return Response(content="OK", media_type="text/plain")


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "healthy"


@app.post(WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    # Optional extra security
    if WEBHOOK_SECRET_TOKEN:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot not ready")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
    except Exception:
        logger.exception("Failed to process update (no content logged).")
        raise HTTPException(status_code=500, detail="Failed to process update")

    return {"ok": True}


# -----------------------------
# Startup / shutdown
# -----------------------------
@app.on_event("startup")
async def on_startup():
    global telegram_app

    logger.info("Starting French Lumière webhook bot...")
    logger.info("Webhook target: %s", WEBHOOK_URL)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("coraud", cmd_coraud))

    await telegram_app.initialize()
    await telegram_app.start()

    await telegram_app.bot.set_webhook(
        url=WEBHOOK_URL,
        secret_token=WEBHOOK_SECRET_TOKEN if WEBHOOK_SECRET_TOKEN else None,
        drop_pending_updates=True,
    )
    logger.info("Webhook set successfully.")


@app.on_event("shutdown")
async def on_shutdown():
    global telegram_app
    logger.info("Shutting down...")

    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
        telegram_app = None

    logger.info("Shutdown complete.")

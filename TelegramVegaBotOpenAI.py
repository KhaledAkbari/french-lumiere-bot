
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
French Lumière Admin Reply Bot (Render Webhook)
==============================================

This FastAPI webhook service is designed for Render Web Service hosting.
It connects a Telegram bot to OpenAI and works inside a Telegram group named:

    "French Lumière"

Behavior:
- The bot ONLY acts when an ADMIN replies to a message in the group using one of:
    /reptex  -> Reply to replied TEXT using OpenAI (French, concise)
    /repaud  -> Reply to replied VOICE/AUDIO:
               download -> convert (ogg/opus -> wav) -> Whisper -> OpenAI -> concise
    /cortex  -> Correct replied TEXT (grammar/spelling) + short note about corrections
    /coraud  -> Transcribe replied VOICE/AUDIO then correct + short note

Security:
- Verifies the command issuer is an admin of the group before doing anything.
- Optional: verifies Telegram secret header if WEBHOOK_SECRET_TOKEN is set.

Render endpoints:
- GET/HEAD /        -> OK
- GET      /healthz -> healthy (for Render health check)
- POST     /webhook -> Telegram webhook endpoint

Environment variables (set in Render):
- BOT_TOKEN              : Telegram bot token
- OPENAI_API             : OpenAI API key
- RENDER_EXTERNAL_URL     : set by Render automatically for Web Services
- WEBHOOK_SECRET_TOKEN    : recommended; verify X-Telegram-Bot-Api-Secret-Token
- OPENAI_MODEL            : optional, default "gpt-4o-mini"

Notes about audio:
- Telegram voice messages are usually OGG/OPUS.
- OpenAI transcription supports formats like wav/mp3/m4a/webm, not reliably ogg.
- This code converts to WAV using pydub (requires ffmpeg).
- On Render, easiest is to deploy with Dockerfile that installs ffmpeg.

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
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# Audio conversion
from pydub import AudioSegment


# -----------------------------
# Minimal logging (no user content)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("FrenchLumiereWebhookBot")
logging.getLogger("httpx").setLevel(logging.WARNING)  # avoid verbose request logs


# -----------------------------
# Configuration
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

# Env vars
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API", "").strip()
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").strip()  # optional override
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()  # optional

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN. Add it in Render -> Environment Variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API. Add it in Render -> Environment Variables.")

# Decide public base URL (Render provides RENDER_EXTERNAL_URL)
BASE_URL = WEBHOOK_BASE_URL or RENDER_EXTERNAL_URL
if not BASE_URL:
    raise RuntimeError(
        "Missing public base URL. On Render Web Service, RENDER_EXTERNAL_URL should exist.\n"
        "If not, set WEBHOOK_BASE_URL manually, e.g. https://your-app.onrender.com"
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
# Helper functions
# -----------------------------
def is_target_group(update: Update) -> bool:
    chat = update.effective_chat
    if not chat or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    return (chat.title or "").strip() == GROUP_NAME


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ("administrator", "creator")


def admin_mention_html(user) -> str:
    """Mention admin by username if available, else clickable mention."""
    if not user:
        return "Admin"
    if user.username:
        return f"@{html.escape(user.username)}"
    name = html.escape(user.first_name or "Admin")
    return f'<a href="tg://user?id={user.id}">{name}</a>'


def sender_reference(user) -> str:
    """Reference original sender by @username or first name."""
    if not user:
        return "Utilisateur"
    if user.username:
        return f"@{user.username}"
    return user.first_name or "Utilisateur"


def get_replied_message(update: Update) -> Optional[Message]:
    msg = update.effective_message
    if not msg:
        return None
    return msg.reply_to_message


def extract_audio_file_id(replied: Message) -> Optional[str]:
    """Return file_id for voice or audio message."""
    if replied.voice:
        return replied.voice.file_id
    if replied.audio:
        return replied.audio.file_id
    return None


async def download_telegram_file_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()


def convert_to_wav_bytes(input_bytes: bytes) -> bytes:
    """
    Convert Telegram OGG/OPUS or other audio formats to WAV using pydub.
    Requires ffmpeg in the runtime (we install it in Dockerfile).
    """
    audio = AudioSegment.from_file(io.BytesIO(input_bytes))
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


def format_header(admin_user, sender_user, sender_id: int, msg_id: int) -> str:
    admin_tag = admin_mention_html(admin_user)
    sender_tag = html.escape(sender_reference(sender_user))
    return (
        f"👮 Admin: {admin_tag}\n"
        f"👤 Original sender: {sender_tag}\n"
        f"🧾 Sender ID: <code>{sender_id}</code> | Message ID: <code>{msg_id}</code>\n\n"
    )


def openai_chat(text: str) -> str:
    """Chat reply in French, concise."""
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


def openai_transcribe_wav(wav_bytes: bytes) -> str:
    """Transcribe WAV using OpenAI Whisper."""
    f = io.BytesIO(wav_bytes)
    f.name = "audio.wav"
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="text",
    )
    if isinstance(transcript, str):
        return transcript.strip()
    return getattr(transcript, "text", "").strip()


async def call_openai_in_thread(func, *args):
    """Run blocking OpenAI SDK calls in a thread (non-blocking to async loop)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


async def handle_errors(update: Update, message: str):
    """Send a safe error message."""
    if update.effective_message:
        await update.effective_message.reply_text(message)


# -----------------------------
# Command handlers
# -----------------------------
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await handle_errors(update, "❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied or not replied.text:
        await handle_errors(update, "⚠️ Please reply to a TEXT message when using /reptex.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = format_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        answer = await call_openai_in_thread(openai_chat, replied.text.strip())
        await update.effective_message.reply_text(
            header + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /reptex.")
    except (RateLimitError, APITimeoutError):
        await handle_errors(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await handle_errors(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /reptex (no user content logged).")
        await handle_errors(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await handle_errors(update, "❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied:
        await handle_errors(update, "⚠️ Please reply to a VOICE/AUDIO message when using /repaud.")
        return

    file_id = extract_audio_file_id(replied)
    if not file_id:
        await handle_errors(update, "⚠️ The replied message must contain VOICE or AUDIO for /repaud.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = format_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        raw = await download_telegram_file_bytes(context, file_id)
        wav = convert_to_wav_bytes(raw)  # requires ffmpeg
        transcript = await call_openai_in_thread(openai_transcribe_wav, wav)

        if not transcript:
            await handle_errors(update, "⚠️ Could not transcribe the audio. Please try again.")
            return

        # Keep response concise even if audio long (max tokens enforced)
        answer = await call_openai_in_thread(openai_chat, transcript)

        await update.effective_message.reply_text(
            header + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /repaud.")
    except FileNotFoundError:
        await handle_errors(update, "⚠️ Audio conversion failed: ffmpeg is missing. Use the provided Dockerfile.")
    except (RateLimitError, APITimeoutError):
        await handle_errors(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await handle_errors(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /repaud (no user content logged).")
        await handle_errors(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await handle_errors(update, "❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied or not replied.text:
        await handle_errors(update, "⚠️ Please reply to a TEXT message when using /cortex.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = format_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    correction_task = (
        "Corrige la grammaire et l’orthographe du texte ci-dessous. "
        "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
        "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
        f"TEXTE:\n{replied.text.strip()}"
    )

    try:
        answer = await call_openai_in_thread(openai_chat, correction_task)
        await update.effective_message.reply_text(
            header + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /cortex.")
    except (RateLimitError, APITimeoutError):
        await handle_errors(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await handle_errors(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /cortex (no user content logged).")
        await handle_errors(update, "⚠️ Unexpected error. Please try again later.")


async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await handle_errors(update, "❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied:
        await handle_errors(update, "⚠️ Please reply to a VOICE/AUDIO message when using /coraud.")
        return

    file_id = extract_audio_file_id(replied)
    if not file_id:
        await handle_errors(update, "⚠️ The replied message must contain VOICE or AUDIO for /coraud.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = format_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        raw = await download_telegram_file_bytes(context, file_id)
        wav = convert_to_wav_bytes(raw)  # requires ffmpeg
        transcript = await call_openai_in_thread(openai_transcribe_wav, wav)

        if not transcript:
            await handle_errors(update, "⚠️ Could not transcribe the audio. Please try again.")
            return

        correction_task = (
            "Voici une transcription d’un message audio. "
            "Corrige la grammaire et l’orthographe. "
            "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
            "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
            f"TRANSCRIPTION:\n{transcript}"
        )

        answer = await call_openai_in_thread(openai_chat, correction_task)

        await update.effective_message.reply_text(
            header + html.escape(answer),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        logger.info("Executed /coraud.")
    except FileNotFoundError:
        await handle_errors(update, "⚠️ Audio conversion failed: ffmpeg is missing. Use the provided Dockerfile.")
    except (RateLimitError, APITimeoutError):
        await handle_errors(update, "⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await handle_errors(update, "⚠️ OpenAI API error. Please try again later.")
    except Exception:
        logger.exception("Error in /coraud (no user content logged).")
        await handle_errors(update, "⚠️ Unexpected error. Please try again later.")


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
    # Verify secret token header if enabled
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
        logger.exception("Failed to process update (no user content logged).")
        raise HTTPException(status_code=500, detail="Failed to process update")

    return {"ok": True}


# -----------------------------
# Startup / Shutdown
# -----------------------------
@app.on_event("startup")
async def on_startup():
    global telegram_app

    logger.info("Starting French Lumière webhook bot...")
    logger.info("Webhook target: %s", WEBHOOK_URL)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    # Register ONLY admin-triggered commands
    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("coraud", cmd_coraud))

    await telegram_app.initialize()
    await telegram_app.start()

    # Set webhook at startup
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

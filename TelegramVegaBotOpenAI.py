
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from pydub import AudioSegment  # requires ffmpeg in Docker


# -----------------------------
# Minimal logging (no sensitive content)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("FrenchLumiereBot")
logging.getLogger("httpx").setLevel(logging.WARNING)


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
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").strip()
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN manquant. Ajoute-le dans Render → Environment Variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API manquant. Ajoute-le dans Render → Environment Variables.")

BASE_URL = WEBHOOK_BASE_URL or RENDER_EXTERNAL_URL
if not BASE_URL:
    raise RuntimeError(
        "URL publique manquante. Render doit fournir RENDER_EXTERNAL_URL. "
        "Sinon, définis WEBHOOK_BASE_URL=https://ton-app.onrender.com"
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


def replied_message(update: Update) -> Optional[Message]:
    msg = update.effective_message
    if not msg:
        return None
    return msg.reply_to_message


def audio_file_id(msg: Message) -> Optional[str]:
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
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


async def run_blocking(func, *args):
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
    return out or "Désolé, je n’ai pas pu générer de réponse. Peux-tu reformuler ?"


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


def tts_to_ogg_opus_bytes(text_fr: str) -> bytes:
    """
    Génère un message vocal (voice) Telegram:
    1) TTS OpenAI -> MP3 bytes
    2) Convertit MP3 -> OGG/OPUS via pydub (ffmpeg requis)
    """
    # Petite sécurité: éviter des audios trop longs
    text_fr = (text_fr or "").strip()
    if len(text_fr) > 900:
        text_fr = text_fr[:900] + "…"

    # TTS -> mp3 en streaming
    mp3_bytes = b""
    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text_fr,
        format="mp3",
    ) as resp:
        for chunk in resp.iter_bytes():
            mp3_bytes += chunk

    # mp3 -> ogg/opus
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    out = io.BytesIO()
    audio.export(out, format="ogg", codec="libopus")
    return out.getvalue()


async def send_french_notice(update: Update, context: ContextTypes.DEFAULT_TYPE, text_fr: str) -> None:
    """Réponse en français au message de commande (utile pour erreurs)."""
    if update.effective_message:
        await update.effective_message.reply_text(text_fr)


async def reply_to_original_text(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str) -> None:
    """Répond au message original (pas au message de commande)."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text_fr,
        reply_to_message_id=replied.message_id,
        disable_web_page_preview=True,
    )


async def reply_to_original_voice(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str) -> None:
    """Répond au message original par un message vocal (voice)."""
    ogg_bytes = await run_blocking(tts_to_ogg_opus_bytes, text_fr)
    voice_file = io.BytesIO(ogg_bytes)
    voice_file.name = "reply.ogg"

    await context.bot.send_voice(
        chat_id=update.effective_chat.id,
        voice=voice_file,
        reply_to_message_id=replied.message_id,
    )


# -----------------------------
# Commands
# -----------------------------
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await send_french_notice(update, context, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = replied_message(update)
    if not rep or not rep.text:
        await send_french_notice(update, context, "⚠️ Réponds à un message texte, puis utilise /reptex.")
        return

    try:
        answer = await run_blocking(openai_chat, rep.text.strip())
        # Répond au message original
        await reply_to_original_text(update, context, rep, answer)
        logger.info("Executed /reptex.")
    except (RateLimitError, APITimeoutError):
        await send_french_notice(update, context, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_french_notice(update, context, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /reptex (no content logged).")
        await send_french_notice(update, context, "⚠️ Erreur inattendue. Réessaie plus tard.")


async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await send_french_notice(update, context, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = replied_message(update)
    if not rep or not rep.text:
        await send_french_notice(update, context, "⚠️ Réponds à un message texte, puis utilise /cortex.")
        return

    task = (
        "Corrige la grammaire et l’orthographe du texte ci-dessous. "
        "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
        "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
        f"TEXTE:\n{rep.text.strip()}"
    )

    try:
        answer = await run_blocking(openai_chat, task)
        await reply_to_original_text(update, context, rep, answer)
        logger.info("Executed /cortex.")
    except (RateLimitError, APITimeoutError):
        await send_french_notice(update, context, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_french_notice(update, context, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /cortex (no content logged).")
        await send_french_notice(update, context, "⚠️ Erreur inattendue. Réessaie plus tard.")


async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await send_french_notice(update, context, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = replied_message(update)
    if not rep:
        await send_french_notice(update, context, "⚠️ Réponds à un message vocal/audio, puis utilise /repaud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await send_french_notice(update, context, "⚠️ Le message visé doit contenir un vocal ou un audio.")
        return

    try:
        raw = await download_file_bytes(context, fid)
        wav = convert_to_wav(raw)
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await send_french_notice(update, context, "⚠️ Je n’ai pas réussi à transcrire l’audio. Réessaie.")
            return

        answer = await run_blocking(openai_chat, transcript)

        # IMPORTANT: réponse en AUDIO (voice), pas en texte
        await reply_to_original_voice(update, context, rep, answer)
        logger.info("Executed /repaud.")
    except (RateLimitError, APITimeoutError):
        await send_french_notice(update, context, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_french_notice(update, context, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /repaud (no content logged).")
        await send_french_notice(update, context, "⚠️ Erreur inattendue (audio). Réessaie plus tard.")


async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await send_french_notice(update, context, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = replied_message(update)
    if not rep:
        await send_french_notice(update, context, "⚠️ Réponds à un message vocal/audio, puis utilise /coraud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await send_french_notice(update, context, "⚠️ Le message visé doit contenir un vocal ou un audio.")
        return

    try:
        raw = await download_file_bytes(context, fid)
        wav = convert_to_wav(raw)
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await send_french_notice(update, context, "⚠️ Je n’ai pas réussi à transcrire l’audio. Réessaie.")
            return

        task = (
            "Voici une transcription d’un message audio. "
            "Corrige la grammaire et l’orthographe. "
            "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
            "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
            f"TRANSCRIPTION:\n{transcript}"
        )

        answer = await run_blocking(openai_chat, task)

        # IMPORTANT: réponse en AUDIO (voice)
        await reply_to_original_voice(update, context, rep, answer)
        logger.info("Executed /coraud.")
    except (RateLimitError, APITimeoutError):
        await send_french_notice(update, context, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_french_notice(update, context, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /coraud (no content logged).")
        await send_french_notice(update, context, "⚠️ Erreur inattendue (audio). Réessaie plus tard.")


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


@app.on_event("startup")
async def on_startup():
    global telegram_app
    logger.info("Starting French Lumière webhook bot...")
    logger.info("Webhook target: %s", WEBHOOK_URL)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
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

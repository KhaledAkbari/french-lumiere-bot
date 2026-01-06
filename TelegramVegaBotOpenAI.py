
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TelegramVegaBotOpenAI.py
========================

Bot Telegram (webhook) pour Render (FastAPI + python-telegram-bot + OpenAI).

Fonctionne UNIQUEMENT dans le groupe Telegram nommé exactement :
    "French Lumière"

Le bot réagit UNIQUEMENT quand un admin répond à un message avec :
- /reptex  : réponse IA (texte) au message texte ciblé
- /cortex  : correction (texte) du message texte ciblé + note courte
- /repaud  : transcription (Whisper) du vocal/audio ciblé -> réponse IA -> réponse en message vocal (audio)
- /coraud  : transcription (Whisper) du vocal/audio ciblé -> correction IA -> réponse en message vocal (audio)

Règles importantes :
- Vérifie que l’émetteur de la commande est admin.
- Répond au message original (pas au message de commande de l’admin).
- Toutes les réponses du bot sont en français.
- Les commandes audio renvoient un message vocal (voice) Telegram en OGG/OPUS.
- Aucun message “Traitement…” n’est envoyé : uniquement les indicateurs Telegram (typing/recording).

Variables d’environnement (Render) :
- BOT_TOKEN
- OPENAI_API
- RENDER_EXTERNAL_URL (automatique sur Render Web Service)
- WEBHOOK_SECRET_TOKEN (optionnel mais recommandé)
- OPENAI_MODEL (optionnel, défaut "gpt-4o-mini")
"""

import os
import io
import logging
import asyncio
import uuid
import re
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, Message
from telegram.constants import ChatType, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from pydub import AudioSegment  # nécessite ffmpeg dans Docker


# -----------------------------
# Logging minimal (pas de contenu utilisateur)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("FrenchLumiereBot")
logging.getLogger("httpx").setLevel(logging.WARNING)


class RedactSecretsFilter(logging.Filter):
    """
    Filtre de sécurité : masque les motifs pouvant ressembler à un token de bot Telegram
    si jamais ils apparaissent par erreur dans une exception/log.
    """
    _bot_token_pattern = re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            msg = self._bot_token_pattern.sub("bot<ROUGE>", msg)
            record.msg = msg
            record.args = ()
        except Exception:
            pass
        return True


logger.addFilter(RedactSecretsFilter())


# -----------------------------
# Configuration
# -----------------------------
GROUP_NAME = "French Lumière"

BEHAVIOR_PROMPT = (
    "Tu es un assistant intégré à un bot Telegram. "
    "Tu réponds uniquement en français, de manière polie, professionnelle et concise. "
    "Réponse courte (maximum 200 tokens). "
    "Aucune métadonnée, aucun en-tête, aucun identifiant."
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
# Client OpenAI
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=30)


# -----------------------------
# FastAPI + Telegram
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


def get_replied_message(update: Update) -> Optional[Message]:
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


def infer_audio_format_hint(msg: Message) -> Optional[str]:
    """
    Déduit un format probable pour aider ffmpeg/pydub à décoder l'audio.
    """
    if msg.voice:
        return "ogg"
    if msg.audio and msg.audio.mime_type:
        mt = msg.audio.mime_type.lower()
        if "ogg" in mt or "opus" in mt:
            return "ogg"
        if "mpeg" in mt or "mp3" in mt:
            return "mp3"
        if "mp4" in mt or "m4a" in mt or "aac" in mt:
            return "m4a"
    return None


async def download_file_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()


def convert_to_wav(raw_bytes: bytes, format_hint: Optional[str] = None) -> bytes:
    """
    Convertit l’audio Telegram (souvent ogg/opus) ou autre (mp3/m4a) en wav via pydub + ffmpeg.
    """
    buf = io.BytesIO(raw_bytes)
    # Donner un nom aide ffmpeg à deviner le format
    if format_hint == "ogg":
        buf.name = "audio.ogg"
    elif format_hint == "mp3":
        buf.name = "audio.mp3"
    elif format_hint == "m4a":
        buf.name = "audio.m4a"
    else:
        buf.name = "audio.bin"

    audio = AudioSegment.from_file(buf, format=format_hint) if format_hint else AudioSegment.from_file(buf)
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


async def run_blocking(func, *args):
    """
    Exécute une fonction sync dans un thread pour ne pas bloquer l'event loop.
    """
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
    Génère un message vocal (voice) Telegram :
    1) TTS OpenAI -> MP3 bytes
    2) Conversion MP3 -> OGG/OPUS via pydub (ffmpeg requis)
    """
    text_fr = (text_fr or "").strip()
    if not text_fr:
        text_fr = "Désolé, je n’ai pas pu générer de réponse."

    # Éviter un audio trop long (borne simple sur la longueur)
    if len(text_fr) > 900:
        text_fr = text_fr[:900] + "…"

    mp3_bytes = b""
    # IMPORTANT: response_format (pas "format")
    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text_fr,
        response_format="mp3",
    ) as resp:
        for chunk in resp.iter_bytes():
            mp3_bytes += chunk

    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    out = io.BytesIO()
    audio.export(out, format="ogg", codec="libopus")
    return out.getvalue()


async def chat_action_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: str):
    """
    Boucle qui envoie périodiquement un 'chat action' Telegram (typing/record_voice),
    afin que l’indicateur reste visible pendant les traitements longs.
    """
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def send_notice_fr(update: Update, text_fr: str) -> None:
    """
    Message d'information (erreur) au message de commande de l’admin.
    """
    if update.effective_message:
        await update.effective_message.reply_text(text_fr)


async def reply_to_original_text(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str) -> None:
    """
    Répond au message original (celui auquel l’admin a répondu).
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text_fr,
        reply_to_message_id=replied.message_id,
        disable_web_page_preview=True,
    )


async def reply_to_original_voice(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str) -> None:
    """
    Répond au message original avec un message vocal (voice) Telegram.
    """
    ogg_bytes = await run_blocking(tts_to_ogg_opus_bytes, text_fr)
    voice_file = io.BytesIO(ogg_bytes)
    voice_file.name = "reponse.ogg"
    voice_file.seek(0)

    await context.bot.send_voice(
        chat_id=update.effective_chat.id,
        voice=voice_file,
        reply_to_message_id=replied.message_id,
    )


# -----------------------------
# Commandes
# -----------------------------
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await send_notice_fr(update, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = get_replied_message(update)
    if not rep or not rep.text:
        await send_notice_fr(update, "⚠️ Réponds à un message texte, puis utilise /reptex.")
        return

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.TYPING))

        answer = await run_blocking(openai_chat, rep.text.strip())
        await reply_to_original_text(update, context, rep, answer)
        logger.info("Executed /reptex.")
    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /reptex (id=%s).", error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue. Réessaie plus tard. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await send_notice_fr(update, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = get_replied_message(update)
    if not rep or not rep.text:
        await send_notice_fr(update, "⚠️ Réponds à un message texte, puis utilise /cortex.")
        return

    task = (
        "Corrige la grammaire et l’orthographe du texte ci-dessous. "
        "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
        "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
        f"TEXTE:\n{rep.text.strip()}"
    )

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.TYPING))

        answer = await run_blocking(openai_chat, task)
        await reply_to_original_text(update, context, rep, answer)
        logger.info("Executed /cortex.")
    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Le service est surchargé ou a expiré. Réessaie bientôt.")
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /cortex (id=%s).", error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue. Réessaie plus tard. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await send_notice_fr(update, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = get_replied_message(update)
    if not rep:
        await send_notice_fr(update, "⚠️ Réponds à un message vocal/audio, puis utilise /repaud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await send_notice_fr(update, "⚠️ Le message visé doit contenir un vocal ou un audio.")
        return

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        # Indicateur “en cours” sans envoyer de texte
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.RECORD_VOICE))

        raw = await download_file_bytes(context, fid)

        # Conversion + transcription dans un thread (fiabilité)
        fmt = infer_audio_format_hint(rep)
        wav = await run_blocking(convert_to_wav, raw, fmt)
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await send_notice_fr(update, "⚠️ Je n’ai pas réussi à transcrire l’audio. Réessaie.")
            return

        # On peut passer à "typing" pendant la génération
        if action_task:
            action_task.cancel()
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.TYPING))

        answer = await run_blocking(openai_chat, transcript)
        await reply_to_original_voice(update, context, rep, answer)
        logger.info("Executed /repaud.")
    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Service surchargé ou expiré. Réessaie dans un instant.")
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /repaud (id=%s).", error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue (audio). Réessaie plus tard. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    if not await is_admin(update, context):
        await send_notice_fr(update, "❌ Seuls les administrateurs peuvent utiliser cette commande.")
        return

    rep = get_replied_message(update)
    if not rep:
        await send_notice_fr(update, "⚠️ Réponds à un message vocal/audio, puis utilise /coraud.")
        return

    fid = audio_file_id(rep)
    if not fid:
        await send_notice_fr(update, "⚠️ Le message visé doit contenir un vocal ou un audio.")
        return

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.RECORD_VOICE))

        raw = await download_file_bytes(context, fid)

        fmt = infer_audio_format_hint(rep)
        wav = await run_blocking(convert_to_wav, raw, fmt)
        transcript = await run_blocking(openai_transcribe, wav)

        if not transcript:
            await send_notice_fr(update, "⚠️ Je n’ai pas réussi à transcrire l’audio. Réessaie.")
            return

        task = (
            "Voici une transcription d’un message audio. "
            "Corrige la grammaire et l’orthographe. "
            "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
            "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
            f"TRANSCRIPTION:\n{transcript}"
        )

        if action_task:
            action_task.cancel()
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, ChatAction.TYPING))

        answer = await run_blocking(openai_chat, task)
        await reply_to_original_voice(update, context, rep, answer)
        logger.info("Executed /coraud.")
    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Service surchargé ou expiré. Réessaie dans un instant.")
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
    except Exception:
        logger.exception("Error in /coraud (id=%s).", error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue (audio). Réessaie plus tard. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


# -----------------------------
# Endpoints FastAPI
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


# -----------------------------
# Startup / Shutdown
# -----------------------------
@app.on_event("startup")
async def on_startup():
    global telegram_app
    logger.info("Starting French Lumière webhook bot...")

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

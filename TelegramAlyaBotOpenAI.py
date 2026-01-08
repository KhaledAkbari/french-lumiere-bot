
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TelegramAlyaBotOpenAI.py
========================

Backend Telegram Bot (webhook) for Render:
- FastAPI + Uvicorn
- python-telegram-bot (webhook mode)
- OpenAI (chat + Whisper transcription + TTS)
- pydub + ffmpeg (audio conversions)

GOAL
----
Alya is a friendly, warm French helper in specific Telegram groups.
She ONLY responds when an authorized user triggers her via commands.

WHERE IT WORKS
--------------
Only inside Telegram groups whose title matches exactly:
- "French Lumière"
- "Les Lumières du Français"

ACCESS RULE
-----------
Alya responds ONLY if the command sender is authorized:
- Admin/Creator OR
- WHITELIST_USER_IDS OR
- ALLOW_ALL_MEMBERS=True

COMMANDS (use as a reply to a message)
--------------------------------------
/reptex   : natural helpful reply (no grammar correction) -> text
/repaud   : natural helpful reply (no grammar correction) -> voice note (OGG/OPUS)
/cortex   : friendly correction + positive tone + feedback (incl. speech if audio) -> text
/coraud   : friendly correction + positive tone + feedback -> voice note
/reftex   : reformulate (more fluid) + 2–3 vocab suggestions -> text
/refaud   : reformulate (more fluid) + 2–3 vocab suggestions -> voice note
/sumtex   : summarize target message -> text
/sumaud   : summarize target message -> voice note
/exttex   : transcribe target audio/voice -> text
/aide     : show commands
/alya     : who is Alya? (identity)

AUDIO INPUT LIMITS (protect costs)
---------------------------------
Based on the TARGET (replied) voice/audio duration:
- If > 180s (3 minutes): ALWAYS refuse for everyone
- If > 60s: refuse unless caller is Admin/Creator
When refusing, Alya sends a “too long” message.

AUDIO OUTPUT LENGTH (voice notes)
---------------------------------
We keep /repaud, /coraud, /refaud, /sumaud generally within ~30–60 seconds
by limiting TTS input text length (character cap). Admins can receive longer.

PERSONA
-------
Alya means “light/elevated”. She should sound calm, supportive, never aggressive.
She should NOT correct grammar unless the user used /cortex or /coraud.

RENDER ENVIRONMENT VARIABLES
----------------------------
Required:
- BOT_TOKEN
- OPENAI_API

Recommended (security):
- WEBHOOK_SECRET_TOKEN

Optional:
- OPENAI_MODEL (default: gpt-4o-mini)
- TTS_MODEL    (default: tts-1)
- TTS_VOICE    (default: nova)
- TTS_SPEED    (default: 1.0)
- WEBHOOK_BASE_URL (only if you need to override Render URL)

Render provides automatically:
- PORT
- (usually) RENDER_EXTERNAL_URL

Health check endpoint:
- GET /healthz -> "healthy"
"""

import os
import io
import re
import uuid
import time
import random
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, Message
from telegram.constants import ChatType, ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.error import Forbidden

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from pydub import AudioSegment  # requires ffmpeg in Docker


# =========================================================
# ⚙️ QUICK SETTINGS (EDIT HERE)
# =========================================================

# ⏱ Auto-delete command messages after X seconds (fractional allowed)
COMMAND_DELETE_DELAY_S = 5.0  # e.g. 2.5

# ⏳ Per-user cooldown (anti-spam / cost control) — NOT shown in /aide
COOLDOWN_SECONDS = 5.0
COOLDOWN_APPLIES_TO = {
    "reptex", "repaud", "cortex", "coraud",
    "reftex", "refaud",
    "sumtex", "sumaud",
    "exttex",
}

# 🎙 Audio input guard (based on replied message duration)
MAX_AUDIO_SECONDS_ABSOLUTE = 180.0   # >3min always reject
MAX_AUDIO_SECONDS_NON_ADMIN = 60.0   # >60s reject unless admin/creator

# 🔊 Voice note output length control (by limiting TTS text input)
TTS_CHAR_CAP_USER = 520
TTS_CHAR_CAP_ADMIN = 900

# 🔐 Access control
ALLOW_ALL_MEMBERS = False
WHITELIST_USER_IDS = []  # e.g. [123456789]

# Allowed group titles (exact match)
GROUP_NAMES = {
    "French Lumière",
    "Les Lumières du Français",
}

BOT_NAME = "Alya"

# 🎧 TTS defaults (YOU REQUESTED: tts-1 + nova)
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "nova").strip()
# Alternatives:
# alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
try:
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0").strip())
except Exception:
    TTS_SPEED = 1.0

# Chat model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MAX_OUTPUT_TOKENS = 200

# Env vars
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API", "").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").strip()
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing. Set it in Render → Environment Variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API missing. Set it in Render → Environment Variables.")

BASE_URL = (WEBHOOK_BASE_URL or RENDER_EXTERNAL_URL).strip()
if not BASE_URL:
    raise RuntimeError(
        "Public URL missing. Render should provide RENDER_EXTERNAL_URL. "
        "Or set WEBHOOK_BASE_URL=https://your-app.onrender.com"
    )

BASE_URL = BASE_URL.rstrip("/")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{BASE_URL}{WEBHOOK_PATH}"


# =========================================================
# 😌 Alya Persona (friendly, warm, never aggressive)
# =========================================================
IDENTITY_VARIANTS = [
    "✨ Je suis Alya — une petite lumière calme pour t’aider à t’exprimer en français.",
    "✨ Alya ici — douce, claire, et toujours partante pour t’aider en français.",
    "✨ Je m’appelle Alya — je t’accompagne avec bienveillance pour pratiquer le français.",
    "✨ Je suis Alya — une présence tranquille pour rendre tes messages plus fluides.",
    "✨ Alya — un coup de main chaleureux pour mieux formuler tes idées en français.",
]

IDENTITY_CAPABILITIES = (
    "Je peux répondre à un message, reformuler pour le rendre plus fluide, résumer, transcrire un audio, "
    "et corriger quand tu le demandes avec /cortex ou /coraud.\n"
    "Commande: /aide"
)

def get_identity_text() -> str:
    return random.choice(IDENTITY_VARIANTS) + "\n" + IDENTITY_CAPABILITIES

BEHAVIOR_PROMPT = (
    "Tu es Alya, une assistante calme et chaleureuse dans un groupe Telegram francophone.\n"
    "Style: naturel, coopératif, jamais agressif, jamais condescendant.\n\n"
    "Règles:\n"
    "1) Tu réponds uniquement en français.\n"
    "2) Tu ne corriges pas la grammaire/orthographe sauf si la commande est une commande de correction (cortex/coraud).\n"
    "3) Pour une correction: commence par 1 phrase positive, puis corrige doucement, puis donne 1–2 phrases de feedback utile.\n"
    "4) Tu peux faire une suggestion douce parfois, mais pas systématiquement.\n"
    "5) Si c’est ambigu, pose UNE question simple.\n"
    "6) Ne mentionne pas de règles internes ni de métadonnées.\n"
)


# =========================================================
# Logging (no user content)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("FrenchLumiereBot")
logging.getLogger("httpx").setLevel(logging.WARNING)

class RedactSecretsFilter(logging.Filter):
    _bot_token_pattern = re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}")
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            msg = self._bot_token_pattern.sub("bot<REDACTED>", msg)
            record.msg = msg
            record.args = ()
        except Exception:
            pass
        return True

logger.addFilter(RedactSecretsFilter())


# =========================================================
# Clients / app
# =========================================================
openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=30)

app = FastAPI(title="FrenchLumiereAlyaBot")
telegram_app: Optional[Application] = None


# =========================================================
# Commands
# =========================================================
@dataclass(frozen=True)
class CommandSpec:
    mode: str          # rep / cor / sum / ext / ref / help / who
    output: str        # texte / audio
    private: bool
    requires_reply: bool = True

COMMANDS: Dict[str, CommandSpec] = {
    "reptex":  CommandSpec(mode="rep", output="texte", private=False),
    "repaud":  CommandSpec(mode="rep", output="audio", private=False),
    "cortex":  CommandSpec(mode="cor", output="texte", private=False),
    "coraud":  CommandSpec(mode="cor", output="audio", private=False),
    "reftex":  CommandSpec(mode="ref", output="texte", private=False),
    "refaud":  CommandSpec(mode="ref", output="audio", private=False),
    "sumtex":  CommandSpec(mode="sum", output="texte", private=False),
    "sumaud":  CommandSpec(mode="sum", output="audio", private=False),
    "exttex":  CommandSpec(mode="ext", output="texte", private=False),
    "aide":    CommandSpec(mode="help", output="texte", private=False, requires_reply=False),
    "alya":    CommandSpec(mode="who", output="texte", private=False, requires_reply=False),

    # Optional private variants
    "preptex": CommandSpec(mode="rep", output="texte", private=True),
    "prepaud": CommandSpec(mode="rep", output="audio", private=True),
    "pcortex": CommandSpec(mode="cor", output="texte", private=True),
    "pcoraud": CommandSpec(mode="cor", output="audio", private=True),
}

HELP_TEXT_FR = (
    "📌 *Commandes (à utiliser en réponse à un message)*\n\n"
    "• /reptex : répondre naturellement (texte + audio si présent) → *texte*\n"
    "• /repaud : répondre naturellement (texte + audio si présent) → *vocal*\n"
    "• /cortex : corriger avec bienveillance (texte + audio si présent) → *texte + feedback*\n"
    "• /coraud : corriger avec bienveillance (texte + audio si présent) → *vocal*\n"
    "• /reftex : reformuler en version plus fluide (texte + audio si présent) → *texte + 2–3 mots utiles*\n"
    "• /refaud : reformuler en version plus fluide (texte + audio si présent) → *vocal + 2–3 mots utiles*\n"
    "• /sumtex : résumer le message ciblé (texte/audio) → *texte*\n"
    "• /sumaud : résumer le message ciblé (texte/audio) → *vocal*\n"
    "• /exttex : audio → *transcription texte* (nécessite un audio)\n"
    "• /alya : qui suis-je ? (identité du bot)\n\n"
    "📩 *Versions privées (en DM à l’auteur original)*\n"
    "• /preptex, /prepaud, /pcortex, /pcoraud\n"
    "⚠️ L’utilisateur doit d’abord démarrer le bot en privé pour recevoir un DM.\n\n"
    "🧹 Les messages de commande sont supprimés automatiquement (si permissions).\n"
)


# =========================================================
# Cooldown memory
# =========================================================
_last_call_ts: Dict[Tuple[int, int], float] = {}

def cooldown_remaining(chat_id: int, user_id: int) -> float:
    now = time.monotonic()
    last = _last_call_ts.get((chat_id, user_id), 0.0)
    return max(0.0, (last + COOLDOWN_SECONDS) - now)

def mark_called(chat_id: int, user_id: int) -> None:
    _last_call_ts[(chat_id, user_id)] = time.monotonic()


# =========================================================
# Group / access helpers
# =========================================================
def is_target_group(update: Update) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    return (chat.title or "").strip() in GROUP_NAMES

async def is_admin_or_creator(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ("administrator", "creator")

async def user_can_use_commands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if not user:
        return False
    if ALLOW_ALL_MEMBERS:
        return True
    if user.id in WHITELIST_USER_IDS:
        return True
    return await is_admin_or_creator(update, context)

def get_replied_message(update: Update) -> Optional[Message]:
    msg = update.effective_message
    if not msg:
        return None
    return msg.reply_to_message

def extract_text_from_message(msg: Message) -> str:
    if msg.text:
        return msg.text.strip()
    if msg.caption:
        return msg.caption.strip()
    return ""

def audio_file_id(msg: Message) -> Optional[str]:
    if msg.voice:
        return msg.voice.file_id
    if msg.audio:
        return msg.audio.file_id
    return None

def audio_duration_seconds(msg: Message) -> Optional[float]:
    if msg.voice and msg.voice.duration is not None:
        return float(msg.voice.duration)
    if msg.audio and msg.audio.duration is not None:
        return float(msg.audio.duration)
    return None

def infer_audio_format_hint(msg: Message) -> Optional[str]:
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


# =========================================================
# Telegram + audio handling
# =========================================================
async def download_file_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()

def convert_to_wav(raw_bytes: bytes, format_hint: Optional[str] = None) -> bytes:
    buf = io.BytesIO(raw_bytes)
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


# =========================================================
# OpenAI calls
# =========================================================
def openai_chat(prompt: str) -> str:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BEHAVIOR_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.45,
    )
    out = (completion.choices[0].message.content or "").strip()
    return out or "Désolé, je n’ai pas pu répondre. Peux-tu reformuler ?"

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
    mp3_bytes = b""
    with openai_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text_fr,
        response_format="mp3",
        speed=TTS_SPEED,
    ) as resp:
        for chunk in resp.iter_bytes():
            mp3_bytes += chunk

    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    out = io.BytesIO()
    out.name = "reply.ogg"
    audio.export(out, format="ogg", codec="libopus")
    return out.getvalue()


# =========================================================
# UX helpers
# =========================================================
async def chat_action_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: str):
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        return
    except Exception:
        return

async def send_notice_fr(update: Update, text_fr: str, parse_mode: Optional[str] = None) -> None:
    if update.effective_message:
        try:
            await update.effective_message.reply_text(text_fr, parse_mode=parse_mode)
        except Exception:
            pass

async def delete_command_message_later(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, delay_s: float):
    await asyncio.sleep(float(delay_s))
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        return


# =========================================================
# Input building
# =========================================================
async def build_input_bundle(context: ContextTypes.DEFAULT_TYPE, msg: Message) -> Tuple[str, Optional[str]]:
    text = extract_text_from_message(msg)
    fid = audio_file_id(msg)
    if not fid:
        return text, None

    fmt = infer_audio_format_hint(msg)
    raw = await download_file_bytes(context, fid)
    wav = await run_blocking(convert_to_wav, raw, fmt)
    transcript = await run_blocking(openai_transcribe, wav)
    transcript = (transcript or "").strip()
    return text, transcript or None

def combine_inputs(text: str, transcript: Optional[str]) -> str:
    parts = []
    if text:
        parts.append(f"TEXTE:\n{text}")
    if transcript:
        parts.append(f"AUDIO (transcription):\n{transcript}")
    return "\n\n".join(parts).strip()


# =========================================================
# Identity triggers (authorized only)
# =========================================================
def _is_identity_question(text: str) -> bool:
    t = (text or "").strip().lower()
    triggers = (
        "qui es-tu", "qui es tu", "t'es qui", "tes qui",
        "c’est qui", "c'est qui", "qui est alya",
        "présente-toi", "presente-toi", "présente toi", "presente toi",
        "comment tu t'appelles", "comment tu t’appelles",
        "quel est ton nom", "ton nom",
    )
    return any(k in t for k in triggers)

async def handle_identity_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_target_group(update):
        return
    user = update.effective_user
    if user and user.is_bot:
        return
    msg = update.effective_message
    if not msg or not msg.text:
        return

    if not await user_can_use_commands(update, context):
        return

    if _is_identity_question(msg.text):
        try:
            await msg.reply_text(get_identity_text())
        except Exception:
            pass


# =========================================================
# Sending results
# =========================================================
async def send_text_result(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str, private_to_target: bool):
    target_user = replied.from_user
    if not target_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier l’auteur du message ciblé.")
        return

    if private_to_target:
        try:
            await context.bot.send_message(chat_id=target_user.id, text=text_fr, disable_web_page_preview=True)
        except Forbidden:
            await send_notice_fr(update, "⚠️ DM impossible. L’utilisateur doit d’abord écrire au bot en privé.")
        except Exception:
            await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du message privé.")
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text_fr,
            reply_to_message_id=replied.message_id,
            disable_web_page_preview=True,
        )

async def send_voice_result(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str, private_to_target: bool, caller_is_admin: bool):
    target_user = replied.from_user
    if not target_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier l’auteur du message ciblé.")
        return

    cap = TTS_CHAR_CAP_ADMIN if caller_is_admin else TTS_CHAR_CAP_USER
    text_fr = (text_fr or "").strip() or "Désolé, je n’ai pas pu générer de réponse."
    if len(text_fr) > cap:
        text_fr = text_fr[:cap].rstrip() + "…"

    ogg_bytes = await run_blocking(tts_to_ogg_opus_bytes, text_fr)
    voice_file = io.BytesIO(ogg_bytes)
    voice_file.name = "reponse.ogg"
    voice_file.seek(0)

    if private_to_target:
        try:
            await context.bot.send_voice(chat_id=target_user.id, voice=voice_file)
        except Forbidden:
            await send_notice_fr(update, "⚠️ DM impossible. L’utilisateur doit d’abord écrire au bot en privé.")
        except Exception:
            await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du vocal privé.")
    else:
        await context.bot.send_voice(
            chat_id=update.effective_chat.id,
            voice=voice_file,
            reply_to_message_id=replied.message_id,
        )


# =========================================================
# Prompts
# =========================================================
def prompt_rep(combined_input: str) -> str:
    add_suggestion = (random.random() < 0.30)
    suggestion_line = (
        "\n\nOptionnel: ajoute UNE petite suggestion douce (1 phrase) pour améliorer la clarté."
        if add_suggestion else ""
    )
    return (
        "Réponds naturellement au message ci-dessous, en français. "
        "Sois amical(e), coopératif(ve), simple et clair. "
        "Ne corrige pas la grammaire/orthographe (sauf si on te le demande explicitement)."
        f"{suggestion_line}\n\n"
        f"{combined_input}"
    )

def prompt_cor(combined_input: str) -> str:
    return (
        "Tu vas corriger le contenu ci-dessous en français. "
        "IMPORTANT: commence par 1 phrase positive (compliment sincère). "
        "Ensuite, donne la version corrigée (sans changer le style). "
        "Enfin, ajoute 1–2 phrases de feedback bienveillant (si c'est un audio, tu peux commenter prononciation/rythme). "
        "Reste doux/douce et encourageant(e).\n\n"
        f"{combined_input}"
    )

def prompt_sum_single(combined_input: str) -> str:
    return (
        "Résume le contenu ci-dessous en français en 2 à 4 phrases maximum. "
        "Sois clair et concis. N’invente rien.\n\n"
        f"{combined_input}"
    )

def prompt_ref(combined_input: str) -> str:
    return (
        "Reformule le message ci-dessous en français pour le rendre plus fluide et naturel, sans changer le sens. "
        "Après la reformulation, propose 2 à 3 mots/expressions utiles (vocabulaire) adaptés au contexte, avec une brève définition en français. "
        "Sois bref/breve.\n\n"
        f"{combined_input}"
    )


# =========================================================
# Main command handler
# =========================================================
async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE, cmd_name: str):
    if not is_target_group(update):
        return

    spec = COMMANDS.get(cmd_name)
    if not spec:
        return

    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return

    # Auto-delete the command message
    if update.effective_message:
        asyncio.create_task(
            delete_command_message_later(context, chat.id, update.effective_message.message_id, COMMAND_DELETE_DELAY_S)
        )

    # Only authorized users trigger responses
    if not await user_can_use_commands(update, context):
        return

    # help / identity
    if spec.mode == "help":
        await send_notice_fr(update, HELP_TEXT_FR, parse_mode=ParseMode.MARKDOWN)
        return
    if spec.mode == "who":
        await send_notice_fr(update, get_identity_text())
        return

    # cooldown
    if cmd_name in COOLDOWN_APPLIES_TO:
        rem = cooldown_remaining(chat.id, user.id)
        if rem > 0:
            await send_notice_fr(update, f"⏳ Un instant… réessaie dans {rem:.1f} s.")
            return
        mark_called(chat.id, user.id)

    replied = get_replied_message(update)
    if spec.requires_reply and not replied:
        await send_notice_fr(update, "⚠️ Réponds à un message, puis utilise la commande.")
        return
    if not replied or not replied.from_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier le message ciblé.")
        return

    caller_is_admin = await is_admin_or_creator(update, context)

    # audio input guard (based on the replied message)
    dur = audio_duration_seconds(replied)
    if dur is not None:
        if dur > MAX_AUDIO_SECONDS_ABSOLUTE:
            await send_notice_fr(update, f"🎙️ Audio trop long ({dur:.0f}s). Maximum {MAX_AUDIO_SECONDS_ABSOLUTE:.0f}s.")
            return
        if (not caller_is_admin) and dur > MAX_AUDIO_SECONDS_NON_ADMIN:
            await send_notice_fr(update, f"🎙️ Audio trop long ({dur:.0f}s). Limite {MAX_AUDIO_SECONDS_NON_ADMIN:.0f}s (sauf admins).")
            return

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        action = ChatAction.RECORD_VOICE if spec.output == "audio" else ChatAction.TYPING
        action_task = asyncio.create_task(chat_action_loop(context, chat.id, action))

        text, transcript = await build_input_bundle(context, replied)

        if spec.mode == "ext":
            if not transcript:
                await send_notice_fr(update, "⚠️ /exttex nécessite un message vocal ou audio.")
                return
            await send_text_result(update, context, replied, transcript.strip(), private_to_target=False)
            return

        combined = combine_inputs(text, transcript)
        if not combined:
            await send_notice_fr(update, "⚠️ Le message ciblé ne contient ni texte ni audio exploitable.")
            return

        if spec.mode == "sum":
            prompt = prompt_sum_single(combined)
        elif spec.mode == "rep":
            prompt = prompt_rep(combined)
        elif spec.mode == "cor":
            prompt = prompt_cor(combined)
        elif spec.mode == "ref":
            prompt = prompt_ref(combined)
        else:
            await send_notice_fr(update, "⚠️ Mode de commande inconnu.")
            return

        result = await run_blocking(openai_chat, prompt)

        if spec.output == "audio":
            await send_voice_result(update, context, replied, result, private_to_target=spec.private, caller_is_admin=caller_is_admin)
        else:
            await send_text_result(update, context, replied, result, private_to_target=spec.private)

    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Service surchargé ou expiré. Réessaie dans un instant.")
        logger.warning("OpenAI timeout/ratelimit in /%s (id=%s).", cmd_name, error_id)
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
        logger.warning("OpenAI APIError in /%s (id=%s).", cmd_name, error_id)
    except Exception:
        logger.exception("Unexpected error in /%s (id=%s).", cmd_name, error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


# =========================================================
# Command wrappers
# =========================================================
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "reptex")
async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "repaud")
async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "cortex")
async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "coraud")
async def cmd_reftex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "reftex")
async def cmd_refaud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "refaud")
async def cmd_sumtex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "sumtex")
async def cmd_sumaud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "sumaud")
async def cmd_exttex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "exttex")
async def cmd_aide(update: Update, context: ContextTypes.DEFAULT_TYPE):    await handle_command(update, context, "aide")
async def cmd_alya(update: Update, context: ContextTypes.DEFAULT_TYPE):    await handle_command(update, context, "alya")


# =========================================================
# FastAPI endpoints
# =========================================================
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


# =========================================================
# Startup / shutdown
# =========================================================
@app.on_event("startup")
async def on_startup():
    global telegram_app
    logger.info("Starting %s webhook bot...", BOT_NAME)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_identity_questions))

    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("coraud", cmd_coraud))
    telegram_app.add_handler(CommandHandler("reftex", cmd_reftex))
    telegram_app.add_handler(CommandHandler("refaud", cmd_refaud))
    telegram_app.add_handler(CommandHandler("sumtex", cmd_sumtex))
    telegram_app.add_handler(CommandHandler("sumaud", cmd_sumaud))
    telegram_app.add_handler(CommandHandler("exttex", cmd_exttex))
    telegram_app.add_handler(CommandHandler("aide", cmd_aide))
    telegram_app.add_handler(CommandHandler("alya", cmd_alya))

    await telegram_app.initialize()
    await telegram_app.start()

    await telegram_app.bot.set_webhook(
        url=WEBHOOK_URL,
        secret_token=WEBHOOK_SECRET_TOKEN if WEBHOOK_SECRET_TOKEN else None,
        drop_pending_updates=True,
    )

    logger.info("Webhook set: %s", WEBHOOK_URL)

@app.on_event("shutdown")
async def on_shutdown():
    global telegram_app
    logger.info("Shutting down...")

    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
        telegram_app = None

    logger.info("Shutdown complete.")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TelegramVegaBotOpenAI.py
========================

Bot Telegram (webhook) pour Render (FastAPI + python-telegram-bot + OpenAI).

Fonctionne UNIQUEMENT dans les groupes Telegram nommés exactement :
    - "French Lumière"
    - "Les Lumières du Français"

Commandes (en réponse à un message) :
- /reptex   : analyse (texte + audio si présent) -> réponse en texte
- /repaud   : analyse (texte + audio si présent) -> réponse en message vocal
- /cortex   : corrige (texte + audio si présent) -> réponse en texte
- /coraud   : corrige (texte + audio si présent) -> réponse en message vocal

Commandes privées (envoie en DM au destinataire original) :
- /preptex  : version privée de /reptex
- /prepaud  : version privée de /repaud
- /pcortex  : version privée de /cortex
- /pcoraud  : version privée de /coraud

Résumé (UNIQUEMENT le message ciblé) :
- /sumtex   : résume le message ciblé (texte/audio) -> texte
- /sumaud   : résume le message ciblé (texte/audio) -> message vocal

Extraction :
- /exttex   : extrait le texte d'un audio (audio -> transcription) -> texte

Aide :
- /aide     : explique les commandes

Identité :
- /alya     : qui suis-je ? (identité du bot)

Règles :
- Le bot réagit uniquement dans les groupes autorisés.
- Toutes les réponses du bot sont en français.
- Les commandes audio renvoient un message vocal (voice) Telegram en OGG/OPUS.
- Les messages de commande (ceux des admins/utilisateurs autorisés) sont supprimés après 15 secondes (si permissions).
- Les commandes privées envoient un DM à l'auteur original (si l'utilisateur a démarré le bot en privé).
- Alya répond aussi automatiquement (sans commande) si quelqu’un lui demande qui elle est.

Variables d’environnement (Render) :
- BOT_TOKEN
- OPENAI_API
- RENDER_EXTERNAL_URL (automatique sur Render Web Service)
- WEBHOOK_SECRET_TOKEN (optionnel mais recommandé)
- OPENAI_MODEL (optionnel, défaut "gpt-4o-mini")
- (optionnel) TTS_MODEL (défaut "tts-1")
- (optionnel) TTS_VOICE (défaut "alloy")
- (optionnel) TTS_SPEED (défaut "1.0")
"""

import os
import io
import re
import uuid
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, Message
from telegram.constants import ChatType, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.error import Forbidden

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from pydub import AudioSegment  # nécessite ffmpeg dans Docker


# =========================================================
# 🔐 CONTRÔLE D’ACCÈS (facile à configurer)
# =========================================================
# Mode A (défaut): seuls les admins/créateurs peuvent utiliser les commandes
# Mode B: ALLOW_ALL_MEMBERS=True -> tout le monde peut utiliser les commandes
# Mode C: WHITELIST_USER_IDS -> utilisateurs autorisés même s'ils ne sont pas admins
ALLOW_ALL_MEMBERS = False
WHITELIST_USER_IDS = [7455750778, 6864593197]  # ex: [7455750778, 6864593197]


# =========================================================
# 🔊 VOIX TTS (facile à configurer)
# =========================================================
# Modèles possibles: "tts-1", "tts-1-hd" (latence vs qualité)
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1").strip()

# Voix supportées pour tts-1 / tts-1-hd :
# alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
TTS_VOICE = os.getenv("TTS_VOICE", "nova").strip()

# Vitesse de lecture (0.25 à 4.0), défaut 1.0
try:
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0").strip())
except Exception:
    TTS_SPEED = 1.0

TTS_1_VOICES = {"alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"}
if TTS_MODEL in ("tts-1", "tts-1-hd") and TTS_VOICE not in TTS_1_VOICES:
    raise RuntimeError(
        f"TTS_VOICE invalide pour {TTS_MODEL}. Choisis parmi: {sorted(TTS_1_VOICES)}"
    )
if TTS_SPEED < 0.25 or TTS_SPEED > 4.0:
    raise RuntimeError("TTS_SPEED doit être entre 0.25 et 4.0")


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
# Configuration (groupes + identité)
# -----------------------------
# Groupes autorisés (titre exact Telegram, accents inclus)
GROUP_NAMES = {
    "French Lumière",
    "Les Lumières du Français",
}

# Identité du bot
BOT_NAME = "Alya"
BOT_IDENTITY_FR = (
    "✨ Je m’appelle Alya.\n"
    "Mon nom évoque une lumière élevée : je suis là pour aider à améliorer ton français.\n\n"
    "Je peux analyser et corriger des messages (texte ou audio), résumer un message ciblé, "
    "et transcrire un audio.\n"
    "Dans ce groupe, j’interviens surtout via les commandes (ex: /reptex, /repaud, /cortex, /coraud, /sumtex, /sumaud, /exttex)."
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MAX_OUTPUT_TOKENS = 220

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

# Prompt système global (en français, concis) — avec identité Alya
BEHAVIOR_PROMPT = (
    f"Tu es {BOT_NAME}, une assistante intégrée à un bot Telegram. "
    "Ton nom évoque une lumière élevée. "
    "Si on te demande qui tu es, réponds brièvement que tu t’appelles Alya et que tu aides à améliorer le français. "
    "Tu réponds uniquement en français, de manière polie, professionnelle et concise. "
    "Aucune métadonnée, aucun en-tête, aucun identifiant. "
    "Évite les longues réponses. Si besoin, demande une clarification en une phrase."
)

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
# Spécifications de commandes (modulaire)
# -----------------------------
@dataclass(frozen=True)
class CommandSpec:
    mode: str         # "rep" | "cor" | "sum" | "ext" | "help" | "who"
    output: str       # "texte" | "audio"
    private: bool     # True -> DM à la cible, False -> réponse dans le groupe
    requires_reply: bool = True


COMMANDS: Dict[str, CommandSpec] = {
    "reptex":  CommandSpec(mode="rep",  output="texte", private=False),
    "repaud":  CommandSpec(mode="rep",  output="audio", private=False),
    "cortex":  CommandSpec(mode="cor",  output="texte", private=False),
    "coraud":  CommandSpec(mode="cor",  output="audio", private=False),

    "preptex": CommandSpec(mode="rep",  output="texte", private=True),
    "prepaud": CommandSpec(mode="rep",  output="audio", private=True),
    "pcortex": CommandSpec(mode="cor",  output="texte", private=True),
    "pcoraud": CommandSpec(mode="cor",  output="audio", private=True),

    "sumtex":  CommandSpec(mode="sum",  output="texte", private=False),
    "sumaud":  CommandSpec(mode="sum",  output="audio", private=False),

    "exttex":  CommandSpec(mode="ext",  output="texte", private=False),
    "aide":    CommandSpec(mode="help", output="texte", private=False, requires_reply=False),
    "alya":    CommandSpec(mode="who",  output="texte", private=False, requires_reply=False),
}


HELP_TEXT_FR = (
    "📌 *Commandes (à utiliser en réponse à un message)*\n\n"
    "• /reptex : analyse (texte + audio si présent) → réponse en texte\n"
    "• /repaud : analyse (texte + audio si présent) → réponse en vocal\n"
    "• /cortex : correction (texte + audio si présent) → texte + note\n"
    "• /coraud : correction (texte + audio si présent) → vocal\n"
    "• /sumtex : résumé du message ciblé (texte/audio) → texte\n"
    "• /sumaud : résumé du message ciblé (texte/audio) → vocal\n"
    "• /exttex : audio → transcription texte (nécessite un audio)\n"
    "• /alya : qui suis-je ? (identité du bot)\n\n"
    "📩 *Versions privées (en DM à l’auteur original)*\n"
    "• /preptex, /prepaud, /pcortex, /pcoraud\n"
    "⚠️ L’utilisateur doit d’abord démarrer le bot en privé pour recevoir un DM.\n\n"
    "🧹 Les messages de commande sont supprimés après 15 secondes (si permissions).\n"
)


# -----------------------------
# Helpers (groupe / accès)
# -----------------------------
def is_target_group(update: Update) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    return (chat.title or "").strip() in GROUP_NAMES


async def user_can_use_commands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Politique d'accès:
    - Si ALLOW_ALL_MEMBERS=True : tout le monde
    - Sinon, si user.id dans WHITELIST_USER_IDS : autorisé
    - Sinon : admin/créateur uniquement
    """
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False

    if ALLOW_ALL_MEMBERS:
        return True

    if user.id in WHITELIST_USER_IDS:
        return True

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
    Convertit l’audio (ogg/opus, mp3, m4a, etc.) en wav via pydub + ffmpeg.
    """
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
    """
    Exécute une fonction sync dans un thread pour ne pas bloquer l'event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def openai_chat(prompt: str) -> str:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BEHAVIOR_PROMPT},
            {"role": "user", "content": prompt},
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

    # borne simple pour éviter des vocaux trop longs
    if len(text_fr) > 900:
        text_fr = text_fr[:900] + "…"

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
    Message d'information (erreur) au message de commande.
    """
    if update.effective_message:
        try:
            await update.effective_message.reply_text(text_fr)
        except Exception:
            pass


async def delete_command_message_later(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, delay_s: int = 15):
    """
    Supprime le message de commande après un délai (si permissions).
    NB: Le bot doit avoir le droit de supprimer des messages dans le groupe.
    """
    await asyncio.sleep(delay_s)
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        return


def extract_text_from_message(msg: Message) -> str:
    """
    Extrait un texte possible : texte direct ou caption.
    """
    if msg.text:
        return msg.text.strip()
    if msg.caption:
        return msg.caption.strip()
    return ""


async def build_input_bundle(context: ContextTypes.DEFAULT_TYPE, msg: Message) -> Tuple[str, Optional[str]]:
    """
    Retourne (texte, transcription_audio) depuis le message ciblé.
    - texte : msg.text ou msg.caption
    - transcription_audio : transcription si voice/audio présent
    """
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
    """
    Combine texte et transcription si disponibles.
    """
    parts = []
    if text:
        parts.append(f"TEXTE:\n{text}")
    if transcript:
        parts.append(f"AUDIO (transcription):\n{transcript}")
    return "\n\n".join(parts).strip()


def _is_identity_question(text: str) -> bool:
    """
    Détecte si quelqu’un demande l’identité d’Alya (sans commande).
    Déclencheurs volontairement simples et robustes.
    """
    t = (text or "").strip().lower()

    triggers = (
        "qui es-tu", "qui es tu", "t'es qui", "tes qui",
        "c’est qui", "c'est qui", "qui est alya", "qui es-tu alya", "qui es tu alya",
        "présente-toi", "presente-toi", "présente toi", "presente toi",
        "tu es qui", "tu es qui ?",
        "c'est toi alya", "c’est toi alya",
        "comment tu t'appelles", "comment tu t’appelles",
        "ton nom", "quel est ton nom",
    )

    return any(k in t for k in triggers)


async def handle_identity_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Réponse automatique si quelqu’un demande : "qui es-tu ?" / "c’est qui Alya ?" etc.
    Ne nécessite pas d’être admin.
    """
    if not is_target_group(update):
        return

    user = update.effective_user
    if user and user.is_bot:
        return

    msg = update.effective_message
    if not msg or not msg.text:
        return

    if _is_identity_question(msg.text):
        try:
            await msg.reply_text(BOT_IDENTITY_FR)
        except Exception:
            pass


async def send_text_result(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str, private_to_target: bool):
    """
    Envoie une réponse texte soit :
    - dans le groupe en réponse au message original
    - en privé à l'auteur original (si possible)
    """
    target_user = replied.from_user
    if not target_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier l’auteur du message ciblé.")
        return

    if private_to_target:
        try:
            await context.bot.send_message(chat_id=target_user.id, text=text_fr, disable_web_page_preview=True)
        except Forbidden:
            await send_notice_fr(
                update,
                "⚠️ Impossible d’envoyer un message privé à cet utilisateur. "
                "Il doit d’abord démarrer une conversation avec le bot (en privé)."
            )
        except Exception:
            await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du message privé.")
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text_fr,
            reply_to_message_id=replied.message_id,
            disable_web_page_preview=True,
        )


async def send_voice_result(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str, private_to_target: bool):
    """
    Envoie une réponse en message vocal (voice) soit :
    - dans le groupe en réponse au message original
    - en privé à l'auteur original (si possible)
    """
    target_user = replied.from_user
    if not target_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier l’auteur du message ciblé.")
        return

    ogg_bytes = await run_blocking(tts_to_ogg_opus_bytes, text_fr)
    voice_file = io.BytesIO(ogg_bytes)
    voice_file.name = "reponse.ogg"
    voice_file.seek(0)

    if private_to_target:
        try:
            await context.bot.send_voice(chat_id=target_user.id, voice=voice_file)
        except Forbidden:
            await send_notice_fr(
                update,
                "⚠️ Impossible d’envoyer un message privé à cet utilisateur. "
                "Il doit d’abord démarrer une conversation avec le bot (en privé)."
            )
        except Exception:
            await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du message vocal privé.")
    else:
        await context.bot.send_voice(
            chat_id=update.effective_chat.id,
            voice=voice_file,
            reply_to_message_id=replied.message_id,
        )


# -----------------------------
# Prompts par mode
# -----------------------------
def prompt_rep(combined_input: str) -> str:
    return (
        "Tu analyses un message en français. "
        "Réponds de façon utile et naturelle, comme une réponse à l'auteur. "
        "Si c'est un rap, commente brièvement le flow, les rimes, la clarté, et propose 1 amélioration. "
        "Répond uniquement en français.\n\n"
        f"{combined_input}"
    )


def prompt_cor(combined_input: str) -> str:
    return (
        "Tu corriges la grammaire et l’orthographe du contenu ci-dessous. "
        "Retourne d’abord la version corrigée. Ensuite, sur une nouvelle ligne, "
        "ajoute une courte note (très brève) expliquant ce qui a été corrigé.\n\n"
        f"{combined_input}"
    )


def prompt_sum_single(combined_input: str) -> str:
    return (
        "Tu résumes le contenu ci-dessous en français en 2 à 4 phrases maximum. "
        "Sois clair et concis. Ne rajoute pas d'informations inventées.\n\n"
        f"{combined_input}"
    )


# -----------------------------
# Handler générique de commande
# -----------------------------
async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE, cmd_name: str):
    # Restriction groupe
    if not is_target_group(update):
        return

    spec = COMMANDS.get(cmd_name)
    if not spec:
        return

    # Suppression du message de commande après 15s
    if update.effective_message:
        asyncio.create_task(
            delete_command_message_later(context, update.effective_chat.id, update.effective_message.message_id, 15)
        )

    # /aide ne nécessite pas de reply
    if spec.mode == "help":
        await send_notice_fr(update, HELP_TEXT_FR)
        return

    # /alya ne nécessite pas d'être admin, ni de reply
    if spec.mode == "who":
        await send_notice_fr(update, BOT_IDENTITY_FR)
        return

    # Contrôle d'accès (admin/créateur par défaut, ou toggle/whitelist)
    if not await user_can_use_commands(update, context):
        await send_notice_fr(update, "❌ Tu n’as pas l’autorisation d’utiliser cette commande.")
        return

    replied = get_replied_message(update)
    if spec.requires_reply and not replied:
        await send_notice_fr(update, "⚠️ Réponds à un message, puis utilise la commande.")
        return

    # Il faut une cible identifiable
    if not replied or not replied.from_user:
        await send_notice_fr(update, "⚠️ Impossible d’identifier le message ciblé.")
        return

    error_id = uuid.uuid4().hex[:8]
    action_task = None

    try:
        # Indicateur “en cours”
        action = ChatAction.RECORD_VOICE if spec.output == "audio" else ChatAction.TYPING
        action_task = asyncio.create_task(chat_action_loop(context, update.effective_chat.id, action))

        # Construire (texte + transcription audio si présent)
        text, transcript = await build_input_bundle(context, replied)

        # Mode extraction : nécessite audio
        if spec.mode == "ext":
            if not transcript:
                await send_notice_fr(update, "⚠️ /exttex nécessite un message vocal ou audio.")
                return
            await send_text_result(update, context, replied, transcript.strip(), private_to_target=False)
            logger.info("Executed /exttex.")
            return

        combined = combine_inputs(text, transcript)
        if not combined:
            await send_notice_fr(update, "⚠️ Le message ciblé ne contient ni texte ni audio exploitable.")
            return

        # Mode résumé : UNIQUEMENT le message ciblé
        if spec.mode == "sum":
            prompt = prompt_sum_single(combined)
            result = await run_blocking(openai_chat, prompt)
            if spec.output == "audio":
                await send_voice_result(update, context, replied, result, private_to_target=spec.private)
            else:
                await send_text_result(update, context, replied, result, private_to_target=spec.private)
            logger.info("Executed /%s.", cmd_name)
            return

        # Modes rep/cor
        if spec.mode == "rep":
            prompt = prompt_rep(combined)
        elif spec.mode == "cor":
            prompt = prompt_cor(combined)
        else:
            await send_notice_fr(update, "⚠️ Mode de commande inconnu.")
            return

        result = await run_blocking(openai_chat, prompt)

        if spec.output == "audio":
            await send_voice_result(update, context, replied, result, private_to_target=spec.private)
        else:
            await send_text_result(update, context, replied, result, private_to_target=spec.private)

        logger.info("Executed /%s.", cmd_name)

    except (RateLimitError, APITimeoutError):
        await send_notice_fr(update, "⚠️ Service surchargé ou expiré. Réessaie dans un instant.")
        logger.warning("OpenAI timeout/ratelimit in /%s (id=%s).", cmd_name, error_id)
    except APIError:
        await send_notice_fr(update, "⚠️ Erreur OpenAI. Réessaie plus tard.")
        logger.warning("OpenAI APIError in /%s (id=%s).", cmd_name, error_id)
    except Exception:
        logger.exception("Erreur inattendue dans /%s (id=%s).", cmd_name, error_id)
        await send_notice_fr(update, f"⚠️ Erreur inattendue. Réessaie plus tard. (code {error_id})")
    finally:
        if action_task:
            action_task.cancel()


# -----------------------------
# Wrappers par commande (PTB exige une fonction par handler)
# -----------------------------
async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "reptex")
async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "repaud")
async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "cortex")
async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "coraud")

async def cmd_preptex(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "preptex")
async def cmd_prepaud(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "prepaud")
async def cmd_pcortex(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "pcortex")
async def cmd_pcoraud(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "pcoraud")

async def cmd_sumtex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "sumtex")
async def cmd_sumaud(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "sumaud")

async def cmd_exttex(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "exttex")
async def cmd_aide(update: Update, context: ContextTypes.DEFAULT_TYPE):    await handle_command(update, context, "aide")
async def cmd_alya(update: Update, context: ContextTypes.DEFAULT_TYPE):    await handle_command(update, context, "alya")


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
    logger.info("Starting %s webhook bot...", BOT_NAME)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    # Réponses automatiques (sans commande) : identité Alya
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_identity_questions))

    # Command handlers
    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("coraud", cmd_coraud))

    telegram_app.add_handler(CommandHandler("preptex", cmd_preptex))
    telegram_app.add_handler(CommandHandler("prepaud", cmd_prepaud))
    telegram_app.add_handler(CommandHandler("pcortex", cmd_pcortex))
    telegram_app.add_handler(CommandHandler("pcoraud", cmd_pcoraud))

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

   y.")


@app.on_event("shutdown")
async def on_shutdown():
    global telegram_app
    logger.info("Shutting down...")

    if telegram_app:
        await telegram_app.stop()
        await telegram_app.shutdown()
        telegram_app = None

    logger.info("Shutdown complete.")

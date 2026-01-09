
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TelegramAlyaBotOpenAI.py
========================

Backend Telegram Bot (webhook) for Render:
- FastAPI + Uvicorn
- python-telegram-bot (webhook mode)
- OpenAI: chat (responses), Whisper (transcription), TTS (voice notes)
- pydub + ffmpeg (audio conversions)

GOAL
----
Alya is a friendly, warm French helper inside specific Telegram groups.
She responds ONLY when an authorized user triggers her via commands.

WHERE IT WORKS (exact group title match)
--------------------------------------
- "French Lumière"
- "Les Lumières du Français"

ACCESS RULE
-----------
Alya responds only if the command sender is authorized:
- Admin/Creator OR
- WHITELIST_USER_IDS OR
- ALLOW_ALL_MEMBERS=True

PRIVACY MODE NOTE
-----------------
For robust /contex thread reconstruction, BotFather privacy mode should be DISABLED
(/setprivacy -> Disable) and the bot should be removed/re-added to the group.

COMMANDS (use as a reply to a message)
--------------------------------------
/reptex   : natural helpful reply (no grammar correction) -> text
/repaud   : natural helpful reply (no grammar correction) -> voice note (OGG/OPUS)
/cortex   : mild correction + positive tone + feedback -> text
/coraud   : mild correction + positive tone + feedback -> voice note
/reftex   : reformulate (more fluid) + concise vocab (exactly 2 items) -> text
/refaud   : reformulate (more fluid) + concise vocab (exactly 2 items) -> voice note
/sumtex   : summarize target message -> text
/sumaud   : summarize target message -> voice note
/exttex   : transcribe target audio/voice -> text
/aide     : show commands (does NOT show /con* nor hidden commands)
/alya     : identity

CONTEXT-AWARE (THREAD) COMMANDS (HIDDEN)
----------------------------------------
/contex and /conaud answer using ONLY the current reply-chain (thread) context.
- Context is NOT stored to disk. It is ephemeral in-memory only.
- Thread is collected ONLY when /con* is used.
- First try: build chain from payload reply_to_message objects.
- If payload is missing nested reply objects (common), fallback: reconstruct chain from
  an in-memory message index cache built from group traffic (requires privacy OFF).
- Thread includes ALL participants in the chain (no filtering), with speaker prefixes.
- Thread input budget is capped to ~THREAD_MAX_INPUT_TOKENS (approx via char budget).

DEBUG (HIDDEN)
--------------
/pdebcon : reply to a message, Alya DM's you a debug report showing:
- payload-based chain
- cache-based reconstructed chain
(Not shown in /aide.)

ADMIN-ONLY PRIVATE COMMANDS (HIDDEN)
-----------------------------------
/apexttex : extract text from audio and send privately to the admin who triggered it
/apreptex : analyze rap text/audio and send privately to the admin who triggered it

ADMIN-ONLY CACHE TOOLS (HIDDEN)
-------------------------------
/cacheinfo   : DM cache size, TTL, sample items, metrics
/diagprivacy : DM quick privacy effectiveness check (recent cached non-command messages)

BEHAVIOR / COST
---------------
- Output is capped to MAX_OUTPUT_TOKENS (default 150).
- Alya does NOT correct grammar unless /cortex or /coraud is used.
- For corrections: 1 positive sentence, then mild corrections, then constructive feedback.
- Always respond in French.

AUDIO LIMITS (protect costs)
---------------------------
Based on the replied (target) audio duration:
- If > 180s: always refuse
- If > 60s: refuse unless caller is admin/creator

TTS
---
Defaults: tts-1 + nova.

RENDER ENVIRONMENT VARIABLES
----------------------------
Required:
- BOT_TOKEN
- OPENAI_API
Recommended:
- WEBHOOK_SECRET_TOKEN
Optional:
- OPENAI_MODEL, TTS_MODEL, TTS_VOICE, TTS_SPEED, WEBHOOK_BASE_URL
- THREAD_MAX_CONTEXT_CHARS, THREAD_MAX_HOPS, THREAD_CACHE_TTL_S, THREAD_CACHE_MAX_MSG
- THREAD_SAFE_FALLBACK=1 (default 1)
Health:
- GET /healthz -> healthy
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
from typing import Optional, Dict, Tuple, List, Any

from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse

from telegram import Update, Message
from telegram.constants import ChatType, ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.error import Forbidden

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from pydub import AudioSegment


# =========================================================
# ⚙️ QUICK SETTINGS (EDIT HERE)
# =========================================================

# ⏱ Auto-delete command messages after X seconds
COMMAND_DELETE_DELAY_S = 0.1

# ⏳ Per-user cooldown (anti-spam / cost control) — NOT shown in /aide
COOLDOWN_SECONDS = 5.0
COOLDOWN_APPLIES_TO = {
    "reptex", "repaud", "cortex", "coraud",
    "reftex", "refaud", "sumtex", "sumaud", "exttex",
    "contex", "conaud", "pdebcon",
    "apexttex", "apreptex",
}

# Thread context budget (approx)
THREAD_MAX_INPUT_TOKENS = 300
# Heuristic: ~4 chars/token → 300 tokens ≈ 1200 chars
THREAD_MAX_CONTEXT_CHARS = int(os.getenv("THREAD_MAX_CONTEXT_CHARS", "1200"))
THREAD_MAX_HOPS = int(os.getenv("THREAD_MAX_HOPS", "12"))
THREAD_MAX_CHARS_BOT = int(os.getenv("THREAD_MAX_CHARS_BOT", "260"))
THREAD_MAX_CHARS_USER = int(os.getenv("THREAD_MAX_CHARS_USER", "450"))

# Ephemeral cache (in-memory only; clears on restart)
THREAD_CACHE_TTL_S = int(os.getenv("THREAD_CACHE_TTL_S", str(24 * 60 * 60)))  # default 6 hours
THREAD_CACHE_MAX_MSG = int(os.getenv("THREAD_CACHE_MAX_MSG", "8000"))
THREAD_CACHE_MAX_TURNS = int(os.getenv("THREAD_CACHE_MAX_TURNS", "12"))

# Minimal safe fallback when thread is missing (1=on, 0=off)
THREAD_SAFE_FALLBACK = (os.getenv("THREAD_SAFE_FALLBACK", "1").strip() == "1")

# 🎙 Audio input guard
MAX_AUDIO_SECONDS_ABSOLUTE = 180.0
MAX_AUDIO_SECONDS_NON_ADMIN = 60.0

# 🔊 Output voice control (TTS input cap)
TTS_CHAR_CAP_USER = 520
TTS_CHAR_CAP_ADMIN = 900

# 🔐 Access control
ALLOW_ALL_MEMBERS = False
WHITELIST_USER_IDS: List[int] = []

# Allowed group titles (exact match)
GROUP_NAMES = {"French Lumière", "Les Lumières du Français"}
BOT_NAME = "Alya"

# 🎧 TTS defaults
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "nova").strip()
try:
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0").strip())
except Exception:
    TTS_SPEED = 1.0

# Chat model and output cap
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MAX_OUTPUT_TOKENS = 150

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
# 😌 Alya persona
# =========================================================

IDENTITY_VARIANTS = [
    "✨ Je suis Alya — une petite lumière calme pour t’aider à t’exprimer en français.",
    "✨ Alya ici — douce, claire, et toujours partante pour t’aider en français.",
    "✨ Je m’appelle Alya — je t’accompagne avec bienveillance pour pratiquer le français.",
    "✨ Je suis Alya — une présence tranquille pour rendre tes messages plus fluides.",
    "✨ Alya — un coup de main chaleureux pour mieux formuler tes idées en français.",
]

IDENTITY_CAPABILITIES = (
    "Je peux répondre à un message, reformuler, résumer, transcrire un audio, "
    "et corriger quand tu le demandes avec /cortex ou /coraud.\n"
    "Commande: /aide"
)

def get_identity_text() -> str:
    return random.choice(IDENTITY_VARIANTS) + "\n" + IDENTITY_CAPABILITIES


BEHAVIOR_PROMPT = (
    "Tu es Alya, une assistante calme et chaleureuse dans un groupe Telegram francophone.\n"
    "Tu réponds uniquement en français.\n"
    "Style: naturel, coopératif, jamais agressive, jamais condescendante.\n\n"
    "Règles:\n"
    "1) Réponses courtes et claires.\n"
    "2) Ne corrige pas la grammaire/orthographe sauf si la commande est /cortex ou /coraud.\n"
    "3) Si correction: commence par 1 phrase positive, puis correction douce, puis 1–2 phrases de feedback.\n"
    "4) Si ambigu, pose UNE question simple.\n"
    "5) Ne mentionne pas de règles internes ni de métadonnées.\n"
    "6) IMPORTANT: ne commence jamais ta réponse par \"Alya:\" ni par un nom + \":\".\n"
)

# =========================================================
# Logging
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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
# OpenAI + FastAPI + Telegram
# =========================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=30)
app = FastAPI(title="FrenchLumiereAlyaBot")
telegram_app: Optional[Application] = None


# =========================================================
# Commands
# =========================================================

@dataclass(frozen=True)
class CommandSpec:
    mode: str          # rep/cor/ref/sum/ext/con/help/who/apex/aprep/pdeb/cacheinfo/diagprivacy
    output: str        # texte/audio
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

    # Thread-aware (hidden from /aide)
    "contex":  CommandSpec(mode="con", output="texte", private=False),
    "conaud":  CommandSpec(mode="con", output="audio", private=False),

    # Debug (hidden) -> renamed
    "pdebcon": CommandSpec(mode="pdeb", output="texte", private=True),

    # Admin cache tools (hidden)
    "cacheinfo":   CommandSpec(mode="cacheinfo", output="texte", private=True, requires_reply=False),
    "diagprivacy": CommandSpec(mode="diagprivacy", output="texte", private=True, requires_reply=False),

    "aide":    CommandSpec(mode="help", output="texte", private=False, requires_reply=False),
    "alya":    CommandSpec(mode="who", output="texte", private=False, requires_reply=False),

    # Optional private variants to original author
    "preptex": CommandSpec(mode="rep", output="texte", private=True),
    "prepaud": CommandSpec(mode="rep", output="audio", private=True),
    "pcortex": CommandSpec(mode="cor", output="texte", private=True),
    "pcoraud": CommandSpec(mode="cor", output="audio", private=True),

    # Admin-only private (hidden)
    "apexttex": CommandSpec(mode="apex", output="texte", private=True),
    "apreptex": CommandSpec(mode="aprep", output="texte", private=True),
}

# /aide excludes /con*, /pdebcon, /cacheinfo, /diagprivacy and admin-only commands
HELP_TEXT_FR = (
    "📌 *Commandes (à utiliser en réponse à un message)*\n\n"
    "• /reptex : répondre naturellement (texte + audio si présent) → *texte*\n"
    "• /repaud : répondre naturellement (texte + audio si présent) → *vocal*\n"
    "• /cortex : corriger avec bienveillance (texte + audio si présent) → *texte + feedback*\n"
    "• /coraud : corriger avec bienveillance (texte + audio si présent) → *vocal*\n"
    "• /reftex : reformuler plus fluide → *texte + 2 mots*\n"
    "• /refaud : reformuler plus fluide → *vocal + 2 mots*\n"
    "• /sumtex : résumer (texte/audio) → *texte*\n"
    "• /sumaud : résumer (texte/audio) → *vocal*\n"
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
    return msg.reply_to_message if msg else None

def extract_text_from_message(msg: Message) -> str:
    return (msg.text or msg.caption or "").strip()

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
# Output sanitization (FIXED for Python 3.12)
# =========================================================

def sanitize_bot_output(text: str) -> str:
    """
    Empêche les réponses d’Alya de commencer par 'Alya:' ou 'Nom:'.
    Évite 'Alya: Alya: ...' et garde la sortie propre.

    (Fix Python 3.12: use flags= instead of inline (?is) after ^)
    """
    t = (text or "").strip()

    # Remove 1 or more "Alya:" prefixes at the beginning (case-insensitive)
    t = re.sub(r"^\s*alya\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*alya\s*:\s*", "", t, flags=re.IGNORECASE)  # in case it's repeated

    # Remove generic "Name:" prefix at the very start (rare but possible)
    t = re.sub(
        r"^\s*[A-Za-zÀ-ÖØ-öø-ÿ0-9_\- ]{1,30}\s*:\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    return t.strip()


# =========================================================
# Thread cache (ephemeral in-memory)
# =========================================================

_THREAD_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}

# Simple metrics to validate thread capture behavior
METRICS: Dict[str, int] = {
    "con_calls": 0,
    "payload_chain_nonzero": 0,
    "payload_chain_zero": 0,
    "cache_hits": 0,
    "cache_entries": 0,
}

def _metrics_refresh() -> None:
    METRICS["cache_entries"] = len(_THREAD_CACHE)

def _cache_key(chat_id: int, message_id: int) -> Tuple[int, int]:
    return (chat_id, message_id)

def _cache_cleanup() -> None:
    cutoff = time.time() - THREAD_CACHE_TTL_S
    to_del = [k for k, v in _THREAD_CACHE.items() if v.get("ts", 0) < cutoff]
    for k in to_del:
        _THREAD_CACHE.pop(k, None)

    if len(_THREAD_CACHE) > THREAD_CACHE_MAX_MSG:
        items = sorted(_THREAD_CACHE.items(), key=lambda kv: kv[1].get("ts", 0))
        for k, _ in items[: len(_THREAD_CACHE) - THREAD_CACHE_MAX_MSG]:
            _THREAD_CACHE.pop(k, None)

    _metrics_refresh()

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[:n].rstrip() + "…") if len(s) > n else s

def cache_message_update(update: Update) -> None:
    """Cache incoming group messages (requires privacy OFF for full coverage)."""
    chat = update.effective_chat
    msg = update.effective_message
    user = update.effective_user

    if not chat or not msg or not user:
        return
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return
    if (chat.title or "").strip() not in GROUP_NAMES:
        return

    text = (msg.text or msg.caption or "").strip()
    reply_to_id = msg.reply_to_message.message_id if msg.reply_to_message else None

    # Better display name (first + last when available)
    sender_name = ((user.first_name or "") + " " + (user.last_name or "")).strip() or (user.username or "Membre")

    _cache_cleanup()
    _THREAD_CACHE[_cache_key(chat.id, msg.message_id)] = {
        "ts": time.time(),
        "sender_id": user.id,
        "sender_name": sender_name,
        "text": text[:450],
        "reply_to_id": reply_to_id,
        "is_bot": bool(user.is_bot),
    }

    # Light instrumentation (helps confirm privacy mode effectiveness)
    logger.info("cache:add chat=%s msg=%s reply_to=%s text_len=%s",
                chat.id, msg.message_id, reply_to_id, len(text))
    _metrics_refresh()

def cache_outgoing(chat_id: int, bot_id: int, message_id: int, reply_to_id: Optional[int], text: str) -> None:
    """Cache Alya outgoing messages explicitly (bots often don't receive their own updates)."""
    _cache_cleanup()
    _THREAD_CACHE[_cache_key(chat_id, message_id)] = {
        "ts": time.time(),
        "sender_id": bot_id,
        "sender_name": "Alya",
        "text": (text or "")[:450],
        "reply_to_id": reply_to_id,
        "is_bot": True,
    }
    _metrics_refresh()

def reconstruct_chain_from_cache(chat_id: int, message_id: int, bot_id: int) -> List[Dict[str, str]]:
    """Reconstruct reply chain using cached reply_to_id pointers."""
    _cache_cleanup()
    turns: List[Dict[str, str]] = []
    total_chars = 0
    cur_id: Optional[int] = message_id

    for _ in range(THREAD_CACHE_MAX_TURNS):
        if not cur_id:
            break
        node = _THREAD_CACHE.get(_cache_key(chat_id, cur_id))
        if not node:
            break

        sid = node.get("sender_id")
        name = "Alya" if sid == bot_id else (node.get("sender_name") or "Membre")
        txt = (node.get("text") or "").strip()

        if txt:
            cap = THREAD_MAX_CHARS_BOT if sid == bot_id else THREAD_MAX_CHARS_USER
            txt = _truncate(txt, cap)

            # Safety: avoid "Alya: Alya: ..."
            if txt.lower().startswith(f"{name.lower()}:"):
                content = txt.strip()
            else:
                content = f"{name}: {txt}".strip()

            role = "assistant" if sid == bot_id else "user"
            turns.append({"role": role, "content": content})

            total_chars += len(content)
            if total_chars >= THREAD_MAX_CONTEXT_CHARS:
                break

        cur_id = node.get("reply_to_id")
        if not cur_id:
            break

    turns.reverse()
    if turns:
        METRICS["cache_hits"] += 1
    return turns

async def cache_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Async wrapper so PTB can await safely."""
    cache_message_update(update)


# =========================================================
# Thread-aware context: payload-based chain (best effort)
# =========================================================

def _speaker_prefix_from_user(u, bot_user_id: int) -> str:
    if not u:
        return ""
    if u.id == bot_user_id:
        return "Alya: "
    name = ((u.first_name or "") + " " + (u.last_name or "")).strip() or (u.username or "Membre")
    return f"{name}: "

def collect_reply_chain_context_payload(start_msg: Optional[Message], bot_user_id: int) -> List[Dict[str, str]]:
    """Collect reply chain using nested reply_to_message objects from payload (may be missing)."""
    turns: List[Dict[str, str]] = []
    total = 0
    cur = start_msg

    for _ in range(THREAD_MAX_HOPS):
        if not cur:
            break
        u = cur.from_user
        if not u:
            break

        txt = (cur.text or cur.caption or "").strip()
        if txt:
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            cap = THREAD_MAX_CHARS_BOT if u.id == bot_user_id else THREAD_MAX_CHARS_USER
            txt = _truncate(txt, cap)

            prefix = _speaker_prefix_from_user(u, bot_user_id)

            # Safety: avoid "Alya: Alya: ..."
            if txt.lower().startswith(prefix.strip().lower()):
                content = txt.strip()
            else:
                content = (prefix + txt).strip()

            role = "assistant" if u.id == bot_user_id else "user"
            turns.append({"role": role, "content": content})
            total += len(content)
            if total >= THREAD_MAX_CONTEXT_CHARS:
                break

        cur = cur.reply_to_message

    turns.reverse()
    if turns:
        METRICS["payload_chain_nonzero"] += 1
    else:
        METRICS["payload_chain_zero"] += 1
    return turns

def format_thread_debug(chat_id: int, replied: Message, combined: str, payload_turns: List[Dict[str, str]], cache_turns: List[Dict[str, str]]) -> str:
    node = _THREAD_CACHE.get(_cache_key(chat_id, replied.message_id))
    node_reply = node.get("reply_to_id") if node else None

    def fmt(label: str, turns: List[Dict[str, str]]) -> List[str]:
        out: List[str] = []
        out.append(f"== {label} ==")
        out.append(f"turns: {len(turns)}")
        out.append(f"chars: {sum(len(t.get('content','')) for t in turns)}")
        for i, t in enumerate(turns, 1):
            role = t.get("role", "?")
            content = t.get("content", "").replace("\n", " ")
            if len(content) > 170:
                content = content[:170] + "…"
            out.append(f"{i:02d}) {role}: {content}")
        out.append("")
        return out

    lines: List[str] = []
    lines.append("🧪 DEBUG /con* — contexte capturé")
    lines.append(f"- chat_id: {chat_id}")
    lines.append(f"- message_id cible: {replied.message_id}")
    lines.append(f"- reply_to_message_id (payload): {replied.reply_to_message.message_id if replied.reply_to_message else None}")
    lines.append(f"- cache.reply_to_id (cible): {node_reply}")
    lines.append(f"- budget chars (≈ {THREAD_MAX_INPUT_TOKENS} tokens): {THREAD_MAX_CONTEXT_CHARS}")
    lines.append(f"- cache entries (approx): {len(_THREAD_CACHE)}")
    lines.append("")
    lines.extend(fmt("PAYLOAD (reply chain depuis l'update)", payload_turns))
    lines.extend(fmt("CACHE (reconstruction)", cache_turns))

    c = combined.replace("\n", " ")
    if len(c) > 220:
        c = c[:220] + "…"
    lines.append(f"MSG COURANT (combined): {c}")
    return "\n".join(lines)


# =========================================================
# OpenAI + audio helpers
# =========================================================

async def run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))

def openai_chat(messages: List[Dict[str, str]]) -> str:
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.45,
    )
    out = (completion.choices[0].message.content or "").strip()
    return out or "Désolé, je n’ai pas pu répondre. Peux-tu reformuler ?"

async def download_file_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()

def convert_to_wav(raw_bytes: bytes, format_hint: Optional[str] = None) -> bytes:
    buf = io.BytesIO(raw_bytes)
    buf.name = f"audio.{format_hint or 'bin'}"
    audio = AudioSegment.from_file(buf, format=format_hint) if format_hint else AudioSegment.from_file(buf)
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()

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

async def send_text_private_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, text_fr: str):
    try:
        await context.bot.send_message(chat_id=user_id, text=text_fr, disable_web_page_preview=True)
    except Forbidden:
        await send_notice_fr(update, "⚠️ DM impossible. Démarre le bot en privé d’abord.")
    except Exception:
        await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du DM.")

async def send_text_result_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str):
    # sanitize to prevent "Alya:" prefixes in output
    text_fr = sanitize_bot_output(text_fr)

    sent = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text_fr,
        reply_to_message_id=replied.message_id,
        disable_web_page_preview=True,
    )
    cache_outgoing(update.effective_chat.id, context.bot.id, sent.message_id, replied.message_id, text_fr)

async def send_voice_result_to_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, replied: Message, text_fr: str, caller_is_admin: bool):
    cap = TTS_CHAR_CAP_ADMIN if caller_is_admin else TTS_CHAR_CAP_USER
    text_fr = (text_fr or "").strip() or "Désolé, je n’ai pas pu générer de réponse."

    # sanitize to prevent "Alya:" prefixes in output
    text_fr = sanitize_bot_output(text_fr)

    if len(text_fr) > cap:
        text_fr = text_fr[:cap].rstrip() + "…"

    ogg_bytes = await run_blocking(tts_to_ogg_opus_bytes, text_fr)
    voice_file = io.BytesIO(ogg_bytes)
    voice_file.name = "reponse.ogg"
    voice_file.seek(0)

    sent = await context.bot.send_voice(
        chat_id=update.effective_chat.id,
        voice=voice_file,
        reply_to_message_id=replied.message_id,
    )
    cache_outgoing(update.effective_chat.id, context.bot.id, sent.message_id, replied.message_id, f"(vocal) {text_fr}")


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
# Prompt builders
# =========================================================

def messages_for_single(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": BEHAVIOR_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def prompt_rep(combined_input: str) -> str:
    add_suggestion = (random.random() < 0.30)
    suggestion_line = ("\n\nOptionnel: une suggestion douce (1 phrase)." if add_suggestion else "")
    return (
        "Réponds naturellement au message ci-dessous, en français. "
        "Sois amical(e), coopératif(ve), simple et clair. "
        "Ne corrige pas la grammaire/orthographe. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'."
        f"{suggestion_line}\n\n{combined_input}"
    )

def prompt_cor(combined_input: str) -> str:
    return (
        "Corrige le contenu ci-dessous en français. "
        "Format: 1 phrase positive, puis correction douce, puis 1–2 phrases de feedback. "
        "Reste bref/breve. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'.\n\n"
        f"{combined_input}"
    )

def prompt_ref(combined_input: str) -> str:
    return (
        "Reformule le message ci-dessous en français pour le rendre plus fluide, sans changer le sens. "
        "Puis donne exactement 2 mots/expressions utiles avec une mini définition (très courte). "
        "Sois très concise. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'.\n\n"
        f"{combined_input}"
    )

def prompt_sum(combined_input: str) -> str:
    return (
        "Résume le contenu ci-dessous en français en 2 à 3 phrases maximum. "
        "Sois clair et concis. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'.\n\n"
        f"{combined_input}"
    )

def prompt_con() -> str:
    return (
        "En te basant UNIQUEMENT sur le fil (reply chain) ci-dessus, réponds au dernier message de façon utile et concise. "
        "Toujours en français. Si une info manque, pose UNE question courte. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'."
    )

def prompt_rap_analyze(combined_input: str) -> str:
    return (
        "Analyse ce texte/rap (ou transcription). "
        "Réponds brièvement en français: 1) point fort, 2) amélioration concrète, 3) suggestion courte. "
        "IMPORTANT: ne commence jamais par 'Alya:' ou un nom + ':'.\n\n"
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

    # Auto-delete command message
    if update.effective_message:
        asyncio.create_task(delete_command_message_later(context, chat.id, update.effective_message.message_id, COMMAND_DELETE_DELAY_S))

    # Only authorized users trigger responses
    if not await user_can_use_commands(update, context):
        return

    # Help / identity
    if spec.mode == "help":
        await send_notice_fr(update, HELP_TEXT_FR, parse_mode=ParseMode.MARKDOWN)
        return
    if spec.mode == "who":
        await send_notice_fr(update, get_identity_text())
        return

    # Cooldown (skip admin cache tools)
    if cmd_name in COOLDOWN_APPLIES_TO:
        rem = cooldown_remaining(chat.id, user.id)
        if rem > 0:
            await send_notice_fr(update, f"⏳ Un instant… réessaie dans {rem:.1f} s.")
            return
        mark_called(chat.id, user.id)

    caller_is_admin = await is_admin_or_creator(update, context)

    # Admin-only cache tools (DM only; no public ack)
    if spec.mode in ("cacheinfo", "diagprivacy"):
        if not caller_is_admin:
            await send_notice_fr(update, "❌ Cette commande est réservée aux admins/créateurs.")
            return

        recent_cutoff = time.time() - 5 * 60  # last 5 minutes
        recent = [(k, v) for k, v in _THREAD_CACHE.items() if v.get("ts", 0) >= recent_cutoff]
        recent_noncmd = len(recent)
        _metrics_refresh()

        if spec.mode == "cacheinfo":
            sample = sorted(_THREAD_CACHE.items(), key=lambda kv: kv[1].get("ts", 0), reverse=True)[:6]
            lines = []
            lines.append("📊 Cache info")
            lines.append(f"- entries: {METRICS['cache_entries']}")
            lines.append(f"- TTL (s): {THREAD_CACHE_TTL_S}")
            lines.append(f"- max msgs: {THREAD_CACHE_MAX_MSG}")
            lines.append(f"- recent (5 min): {recent_noncmd}")
            lines.append(f"- payload_chain_nonzero: {METRICS['payload_chain_nonzero']}")
            lines.append(f"- payload_chain_zero: {METRICS['payload_chain_zero']}")
            lines.append(f"- cache_hits: {METRICS['cache_hits']}")
            lines.append("")
            for (ck, node) in sample:
                lines.append(
                    f"{time.strftime('%H:%M:%S', time.localtime(node.get('ts',0)))} "
                    f"chat={ck[0]} msg={ck[1]} from={node.get('sender_name')} "
                    f"reply_to={node.get('reply_to_id')} text=\"{_truncate(node.get('text',''), 100)}\""
                )
            await send_text_private_to_user(update, context, user.id, "\n".join(lines))
            return

        if spec.mode == "diagprivacy":
            msg = (
                "🔎 Diagnostic confidentialité\n"
                f"• Messages non-commandes observés (5 min): {recent_noncmd}\n"
                "• Interprétation: "
                + ("✅ Privacy OFF (groupe livré au cache)" if recent_noncmd > 0 else "⚠️ Probable privacy ON (ou peu d'activité)")
            )
            await send_text_private_to_user(update, context, user.id, msg)
            return

    # From here: commands that may require a replied message
    replied = get_replied_message(update)

    if spec.requires_reply:
        if not replied:
            await send_notice_fr(update, "⚠️ Réponds à un message, puis utilise la commande.")
            return
        if not replied.from_user:
            await send_notice_fr(update, "⚠️ Impossible d’identifier le message ciblé.")
            return

    # Audio input guard (only when a replied message exists)
    if replied:
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

        # Build input (only for reply-based commands)
        combined = ""
        transcript = None
        if replied:
            text, transcript = await build_input_bundle(context, replied)
            combined = combine_inputs(text, transcript)

        # Admin-only hidden
        if spec.mode in ("apex", "aprep"):
            if not caller_is_admin:
                await send_notice_fr(update, "❌ Cette commande est réservée aux admins/créateurs.")
                return
            if spec.mode == "apex":
                if not transcript:
                    await send_notice_fr(update, "⚠️ /apexttex nécessite un message vocal ou audio.")
                    return
                await send_text_private_to_user(update, context, user.id, transcript.strip())
                return
            if spec.mode == "aprep":
                result = await run_blocking(openai_chat, messages_for_single(prompt_rap_analyze(combined)))
                await send_text_private_to_user(update, context, user.id, sanitize_bot_output(result))
                return

        # Debug hidden (DM only; no public ack) -> /pdebcon
        if spec.mode == "pdeb":
            if not caller_is_admin:
                await send_notice_fr(update, "❌ Cette commande est réservée aux admins/créateurs.")
                return
            bot_id = context.bot.id
            chat_id = chat.id
            parent = replied.reply_to_message if replied else None
            payload_turns = collect_reply_chain_context_payload(parent, bot_id) if parent else []
            cache_turns = reconstruct_chain_from_cache(chat_id, replied.message_id, bot_id) if replied else []
            dbg = format_thread_debug(chat_id, replied, combined, payload_turns, cache_turns)
            await send_text_private_to_user(update, context, user.id, dbg)
            return

        # Extraction
        if spec.mode == "ext":
            if not transcript:
                await send_notice_fr(update, "⚠️ /exttex nécessite un message vocal ou audio.")
                return
            await send_text_result_to_chat(update, context, replied, transcript.strip())
            return

        if spec.mode != "con" and spec.mode not in ("help", "who") and replied and not combined:
            await send_notice_fr(update, "⚠️ Le message ciblé ne contient ni texte ni audio exploitable.")
            return

        # Thread-aware
        if spec.mode == "con":
            METRICS["con_calls"] += 1
            bot_id = context.bot.id
            chat_id = chat.id

            # Best effort: payload chain from parent
            parent = replied.reply_to_message if replied else None
            chain_turns = collect_reply_chain_context_payload(parent, bot_id) if parent else []

            # Fallback: reconstruct from cache starting at replied message
            if not chain_turns and replied:
                chain_turns = reconstruct_chain_from_cache(chat_id, replied.message_id, bot_id)

            # Ensure the last turn reflects the current combined content
            current_line = (_speaker_prefix_from_user(replied.from_user, bot_id) + combined).strip()
            current_line = _truncate(current_line, THREAD_MAX_CHARS_USER)

            if chain_turns:
                chain_turns[-1] = {"role": "user", "content": current_line}
            else:
                chain_turns = [{"role": "user", "content": current_line}]

            # Minimal safe fallback (no drift) when we have no prior context
            only_current = (len(chain_turns) == 1)
            if THREAD_SAFE_FALLBACK and only_current:
                fallback_text = (
                    "Je ne vois pas le fil de conversation.\n"
                    "Peux-tu préciser en une phrase le contexte du message auquel tu réponds ?"
                )
                if spec.output == "audio":
                    await send_voice_result_to_chat(update, context, replied, fallback_text, caller_is_admin)
                else:
                    await send_text_result_to_chat(update, context, replied, fallback_text)
                return

            msgs = [{"role": "system", "content": BEHAVIOR_PROMPT}] + chain_turns + [
                {"role": "user", "content": prompt_con()}
            ]
            result = await run_blocking(openai_chat, msgs)

            if spec.output == "audio":
                await send_voice_result_to_chat(update, context, replied, result, caller_is_admin)
            else:
                await send_text_result_to_chat(update, context, replied, result)
            return

        # Normal single-message commands
        if spec.mode == "rep":
            user_prompt = prompt_rep(combined)
        elif spec.mode == "cor":
            user_prompt = prompt_cor(combined)
        elif spec.mode == "ref":
            user_prompt = prompt_ref(combined)
        elif spec.mode == "sum":
            user_prompt = prompt_sum(combined)
        else:
            await send_notice_fr(update, "⚠️ Mode de commande inconnu.")
            return

        result = await run_blocking(openai_chat, messages_for_single(user_prompt))

        # Private variants to original author
        if spec.private:
            member_id = replied.from_user.id
            try:
                await context.bot.send_message(chat_id=member_id, text=sanitize_bot_output(result), disable_web_page_preview=True)
            except Forbidden:
                await send_notice_fr(update, "⚠️ DM impossible. L’utilisateur doit d’abord démarrer le bot en privé.")
            except Exception:
                await send_notice_fr(update, "⚠️ Erreur lors de l’envoi du DM.")
        else:
            if spec.output == "audio":
                await send_voice_result_to_chat(update, context, replied, result, caller_is_admin)
            else:
                await send_text_result_to_chat(update, context, replied, result)

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

async def cmd_reptex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "reptex")
async def cmd_repaud(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "repaud")
async def cmd_cortex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "cortex")
async def cmd_coraud(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "coraud")
async def cmd_reftex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "reftex")
async def cmd_refaud(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "refaud")
async def cmd_sumtex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "sumtex")
async def cmd_sumaud(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "sumaud")
async def cmd_exttex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "exttex")
async def cmd_contex(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "contex")
async def cmd_conaud(update: Update, context: ContextTypes.DEFAULT_TYPE):   await handle_command(update, context, "conaud")
async def cmd_pdebcon(update: Update, context: ContextTypes.DEFAULT_TYPE):  await handle_command(update, context, "pdebcon")
async def cmd_cacheinfo(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "cacheinfo")
async def cmd_diagprivacy(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "diagprivacy")
async def cmd_aide(update: Update, context: ContextTypes.DEFAULT_TYPE):     await handle_command(update, context, "aide")
async def cmd_alya(update: Update, context: ContextTypes.DEFAULT_TYPE):     await handle_command(update, context, "alya")
async def cmd_apexttex(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "apexttex")
async def cmd_apreptex(update: Update, context: ContextTypes.DEFAULT_TYPE): await handle_command(update, context, "apreptex")


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
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)):
    if WEBHOOK_SECRET_TOKEN and x_telegram_bot_api_secret_token != WEBHOOK_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot not ready")

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


# =========================================================
# Startup / shutdown
# =========================================================

@app.on_event("startup")
async def on_startup():
    global telegram_app
    logger.info("Starting %s webhook bot...", BOT_NAME)

    telegram_app = Application.builder().token(BOT_TOKEN).build()

    # Passive cache for ALL messages (requires privacy OFF for full group coverage)
    telegram_app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, cache_all_messages), group=0)

    # Identity trigger (text)
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_identity_questions), group=1)

    # Commands
    telegram_app.add_handler(CommandHandler("reptex", cmd_reptex))
    telegram_app.add_handler(CommandHandler("repaud", cmd_repaud))
    telegram_app.add_handler(CommandHandler("cortex", cmd_cortex))
    telegram_app.add_handler(CommandHandler("coraud", cmd_coraud))
    telegram_app.add_handler(CommandHandler("reftex", cmd_reftex))
    telegram_app.add_handler(CommandHandler("refaud", cmd_refaud))
    telegram_app.add_handler(CommandHandler("sumtex", cmd_sumtex))
    telegram_app.add_handler(CommandHandler("sumaud", cmd_sumaud))
    telegram_app.add_handler(CommandHandler("exttex", cmd_exttex))

    # Thread-aware (hidden from /aide)
    telegram_app.add_handler(CommandHandler("contex", cmd_contex))
    telegram_app.add_handler(CommandHandler("conaud", cmd_conaud))

    # Debug (hidden) -> /pdebcon
    telegram_app.add_handler(CommandHandler("pdebcon", cmd_pdebcon))

    # Admin cache tools (hidden)
    telegram_app.add_handler(CommandHandler("cacheinfo", cmd_cacheinfo))
    telegram_app.add_handler(CommandHandler("diagprivacy", cmd_diagprivacy))

    telegram_app.add_handler(CommandHandler("aide", cmd_aide))
    telegram_app.add_handler(CommandHandler("alya", cmd_alya))

    # Admin-only hidden
    telegram_app.add_handler(CommandHandler("apexttex", cmd_apexttex))
    telegram_app.add_handler(CommandHandler("apreptex", cmd_apreptex))

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

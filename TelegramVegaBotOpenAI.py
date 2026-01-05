
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
French Lumière Admin-Reply Bot (Telegram + OpenAI)
==================================================

What it does:
-------------
This bot works inside a Telegram group named exactly: "French Lumière".

It only responds when an ADMIN replies to a specific message using one of:
  /reptex  -> Bot replies (in French) to the replied TEXT message using OpenAI.
             Must mention the admin and include original sender user ID + message ID.
  /repaud  -> Bot replies to the replied AUDIO/VOICE message:
             - downloads audio
             - transcribes using OpenAI Whisper (speech-to-text)
             - sends transcript to OpenAI Chat
             - reply concise (<= 200 tokens)
             Must mention the admin and include original sender user ID + message ID.
  /cortex  -> Correct replied TEXT (grammar/spelling) + short note explaining corrections.
  /coraud  -> Transcribe replied AUDIO/VOICE, then correct grammar/spelling + short note.

Security & Rules:
-----------------
- The user issuing the command must be an ADMIN of the group.
- The command must be sent as a reply to a message.
- Bot only works in the group named "French Lumière" (exact match).
- Minimal logging: does NOT log message text or transcripts.

Important Telegram setting:
---------------------------
To let the bot read all group messages, disable bot privacy:
BotFather -> /setprivacy -> Disable

Dependencies:
-------------
pip install python-telegram-bot==21.10 openai==1.59.7 pydub==0.25.1
(Recommended) install FFmpeg for audio conversion.

OpenAI models used:
-------------------
- Chat: default "gpt-4o-mini" (change via OPENAI_MODEL)
- Transcription: "whisper-1"

"""

import os
import io
import logging
from typing import Optional, Tuple

from telegram import Update, Message
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# Optional but recommended for audio conversion (Telegram voice often .ogg/opus)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False


# -----------------------------
# CONFIG (placeholders)
# -----------------------------
GROUP_NAME = "French Lumière"

# Option 1 (recommended): set env vars
#   export TELEGRAM_BOT_TOKEN="..."
#   export OPENAI_API_KEY="..."
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN", "PASTE_TELEGRAM_BOT_TOKEN_HERE").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API", "PASTE_OPENAI_API_KEY_HERE").strip()

# OpenAI Chat model (you can change it)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# OpenAI behavior prompt (required by your spec)
BEHAVIOR_PROMPT = (
    "You are an assistant integrated into a Telegram bot. Your role is to respond "
    "to user messages in French, in a very short, polite, and cooperative manner. "
    "Use a professional and friendly tone. Keep the response concise—maximum 200 tokens. "
    "Reply only in French."
)

# Token limit for ALL bot outputs (as requested)
MAX_OUTPUT_TOKENS = 200

# Minimal logging (do NOT log user content)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("FrenchLumiereBot")

# Silence verbose httpx logs that could include sensitive URLs in some setups
logging.getLogger("httpx").setLevel(logging.WARNING)


# -----------------------------
# OPENAI CLIENT
# -----------------------------
if not OPENAI_API_KEY or OPENAI_API_KEY == "PASTE_OPENAI_API_KEY_HERE":
    raise RuntimeError("Missing OPENAI API key. Set OPENAI_API env var or edit the placeholder.")

openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=30)


# -----------------------------
# HELPERS
# -----------------------------
def is_target_group(update: Update) -> bool:
    """Return True only if the message is in the target group name."""
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False
    # Title must match exactly
    return (chat.title or "").strip() == GROUP_NAME


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if the user who issued the command is an admin in this chat."""
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False

    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ("administrator", "creator")


def mention_html(user) -> str:
    """
    Mention a user safely in HTML:
    - If username exists, show @username (not a real mention link, but readable)
    - Otherwise create a clickable mention link via tg://user?id=
    """
    if not user:
        return "Utilisateur"
    if user.username:
        return f"@{user.username}"
    # HTML mention link
    name = (user.first_name or "Utilisateur").replace("<", "").replace(">", "")
    return f'<a href="tg://user?id={user.id}">{name}</a>'


def sender_display(user) -> str:
    """Return @username or first name (no HTML link) for referencing sender in text."""
    if not user:
        return "Utilisateur"
    if user.username:
        return f"@{user.username}"
    return user.first_name or "Utilisateur"


def get_replied_message(update: Update) -> Optional[Message]:
    """Return the message being replied to, or None."""
    msg = update.effective_message
    if not msg:
        return None
    return msg.reply_to_message


def extract_audio_file_id(replied: Message) -> Optional[str]:
    """Get file_id from replied audio/voice."""
    if replied.voice:
        return replied.voice.file_id
    if replied.audio:
        return replied.audio.file_id
    # You can add video_note, document audio, etc. if needed
    return None


async def download_telegram_file_as_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    """Download Telegram file to memory as bytes."""
    tg_file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    return buf.getvalue()


def convert_audio_to_wav_bytes(audio_bytes: bytes) -> Tuple[bytes, str]:
    """
    Convert audio bytes to WAV using pydub+ffmpeg.
    Returns (wav_bytes, filename).
    If pydub/ffmpeg isn't available, returns original bytes and a generic filename.

    Note: OpenAI transcription supports wav/mp3/webm/etc; Telegram voice is often ogg/opus.
    """
    if not PYDUB_AVAILABLE:
        return audio_bytes, "audio.ogg"

    try:
        # pydub can auto-detect formats if ffmpeg is installed
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        out = io.BytesIO()
        audio.export(out, format="wav")
        return out.getvalue(), "audio.wav"
    except Exception:
        # Fallback: return original bytes if conversion fails
        return audio_bytes, "audio.ogg"


def build_header(admin_user, sender_user, sender_id: int, msg_id: int) -> str:
    """Build the required header with mentions and IDs."""
    admin_mention = mention_html(admin_user)
    sender_ref = sender_display(sender_user)  # username or first name
    return (
        f"👮 Admin: {admin_mention}\n"
        f"👤 Original sender: {sender_ref}\n"
        f"🧾 Sender ID: <code>{sender_id}</code> | Message ID: <code>{msg_id}</code>\n\n"
    )


# -----------------------------
# OPENAI CALLS
# -----------------------------
def openai_chat_reply(user_content: str) -> str:
    """
    Call OpenAI Chat Completions with the fixed behavior prompt.
    Keep output concise (<= MAX_OUTPUT_TOKENS).
    """
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BEHAVIOR_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.3,
    )
    text = (completion.choices[0].message.content or "").strip()
    return text or "Désolé, je n’ai pas pu générer de réponse. Pouvez-vous reformuler ?"


def openai_transcribe(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio using OpenAI transcription endpoint (Whisper).
    Uses model: whisper-1
    """
    # OpenAI expects a file-like object with a name attribute
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = filename  # important: give it a filename with extension

    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=file_obj,
        response_format="text",
    )

    # transcript may be a string or object depending on SDK; handle both
    if isinstance(transcript, str):
        return transcript.strip()
    return getattr(transcript, "text", "").strip()


# -----------------------------
# COMMAND HANDLERS
# -----------------------------
async def reptex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to a replied TEXT message using OpenAI."""
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await update.effective_message.reply_text("❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied or not replied.text:
        await update.effective_message.reply_text("⚠️ Please reply to a TEXT message when using /reptex.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = build_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        prompt = replied.text.strip()
        answer = openai_chat_reply(prompt)
        await update.effective_message.reply_text(
            header + answer,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info("Executed /reptex successfully.")
    except (RateLimitError, APITimeoutError):
        await update.effective_message.reply_text("⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await update.effective_message.reply_text("⚠️ OpenAI API error. Please try again later.")
    except Exception:
        await update.effective_message.reply_text("⚠️ Unexpected error occurred. Please try again later.")
        logger.exception("Error in /reptex (no user content logged).")


async def repaud(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to a replied AUDIO/VOICE message: transcribe -> OpenAI -> reply concise."""
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await update.effective_message.reply_text("❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied:
        await update.effective_message.reply_text("⚠️ Please reply to an audio/voice message when using /repaud.")
        return

    file_id = extract_audio_file_id(replied)
    if not file_id:
        await update.effective_message.reply_text("⚠️ The replied message must contain AUDIO or VOICE for /repaud.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = build_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        raw_bytes = await download_telegram_file_as_bytes(context, file_id)
        wav_bytes, fname = convert_audio_to_wav_bytes(raw_bytes)

        transcript = openai_transcribe(wav_bytes, fname)
        if not transcript:
            await update.effective_message.reply_text("⚠️ Could not transcribe the audio. Please try again.")
            return

        # Keep response concise even for long audio (max tokens already enforced)
        answer = openai_chat_reply(transcript)

        await update.effective_message.reply_text(
            header + answer,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info("Executed /repaud successfully.")
    except (RateLimitError, APITimeoutError):
        await update.effective_message.reply_text("⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await update.effective_message.reply_text("⚠️ OpenAI API error. Please try again later.")
    except Exception:
        await update.effective_message.reply_text(
            "⚠️ Unexpected error. If you replied to a voice message, make sure FFmpeg is installed."
        )
        logger.exception("Error in /repaud (no user content logged).")


async def cortex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Correct replied TEXT grammar/spelling + short note about corrections."""
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await update.effective_message.reply_text("❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied or not replied.text:
        await update.effective_message.reply_text("⚠️ Please reply to a TEXT message when using /cortex.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = build_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        text = replied.text.strip()
        # Use the same behavior prompt, but instruct correction in the user message
        correction_task = (
            "Corrige la grammaire et l’orthographe du texte ci-dessous. "
            "Retourne d’abord la version corrigée, puis en dessous une courte note "
            "expliquant ce qui a été corrigé (très bref).\n\n"
            f"TEXTE:\n{text}"
        )

        answer = openai_chat_reply(correction_task)

        await update.effective_message.reply_text(
            header + answer,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info("Executed /cortex successfully.")
    except (RateLimitError, APITimeoutError):
        await update.effective_message.reply_text("⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await update.effective_message.reply_text("⚠️ OpenAI API error. Please try again later.")
    except Exception:
        await update.effective_message.reply_text("⚠️ Unexpected error occurred. Please try again later.")
        logger.exception("Error in /cortex (no user content logged).")


async def coraud(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe replied AUDIO/VOICE, then correct grammar/spelling + note."""
    if not is_target_group(update):
        return

    if not await is_admin(update, context):
        await update.effective_message.reply_text("❌ Only group admins can use this command.")
        return

    replied = get_replied_message(update)
    if not replied:
        await update.effective_message.reply_text("⚠️ Please reply to an audio/voice message when using /coraud.")
        return

    file_id = extract_audio_file_id(replied)
    if not file_id:
        await update.effective_message.reply_text("⚠️ The replied message must contain AUDIO or VOICE for /coraud.")
        return

    admin_user = update.effective_user
    sender_user = replied.from_user
    header = build_header(admin_user, sender_user, sender_user.id if sender_user else 0, replied.message_id)

    try:
        raw_bytes = await download_telegram_file_as_bytes(context, file_id)
        wav_bytes, fname = convert_audio_to_wav_bytes(raw_bytes)

        transcript = openai_transcribe(wav_bytes, fname)
        if not transcript:
            await update.effective_message.reply_text("⚠️ Could not transcribe the audio. Please try again.")
            return

        correction_task = (
            "Voici une transcription d’un message audio. "
            "Corrige la grammaire et l’orthographe. "
            "Retourne d’abord la version corrigée, puis en dessous une courte note "
            "expliquant ce qui a été corrigé (très bref).\n\n"
            f"TRANSCRIPTION:\n{transcript}"
        )

        answer = openai_chat_reply(correction_task)

        await update.effective_message.reply_text(
            header + answer,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info("Executed /coraud successfully.")
    except (RateLimitError, APITimeoutError):
        await update.effective_message.reply_text("⚠️ OpenAI is busy or timed out. Please try again.")
    except APIError:
        await update.effective_message.reply_text("⚠️ OpenAI API error. Please try again later.")
    except Exception:
        await update.effective_message.reply_text(
            "⚠️ Unexpected error. If you replied to a voice message, make sure FFmpeg is installed."
        )
        logger.exception("Error in /coraud (no user content logged).")


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "PASTE_TELEGRAM_BOT_TOKEN_HERE":
        raise RuntimeError("Missing TELEGRAM bot token. Set BOT_TOKEN env var or edit the placeholder.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register commands
    app.add_handler(CommandHandler("reptex", reptex))
    app.add_handler(CommandHandler("repaud", repaud))
    app.add_handler(CommandHandler("cortex", cortex))
    app.add_handler(CommandHandler("coraud", coraud))

    logger.info("Bot started. Listening in group: %s", GROUP_NAME)

    # Polling is simplest for beginners.
    # For production on servers, webhooks are also possible.
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

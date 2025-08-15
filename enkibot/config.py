# enkibot/config.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================================
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# === EnkiBot Configuration ===
# ==================================================================================================
# Central configuration file for API keys, model IDs, database settings, and other constants.
# It is best practice to load sensitive information from environment variables.
# ==================================================================================================

import os
import logging

# --- Core Bot Settings ---
TELEGRAM_BOT_TOKEN = os.getenv('ENKI_BOT_TOKEN')
# A list of bot nicknames that trigger a response in group chats.
BOT_NICKNAMES_TO_CHECK = ["enki", "энки", "енки"]

# Feature toggles
ENABLE_SPAM_DETECTION = os.getenv('ENKI_BOT_ENABLE_SPAM_DETECTION', 'true').lower() == 'true'
CAPTCHA_TIMEOUT_SECONDS = int(os.getenv('ENKI_BOT_CAPTCHA_TIMEOUT_SECONDS', '60'))
CAPTCHA_MAX_ATTEMPTS = int(os.getenv('ENKI_BOT_CAPTCHA_MAX_ATTEMPTS', '3'))

# Usage quotas
DAILY_LLM_QUOTA = int(os.getenv('ENKI_BOT_DAILY_LLM_QUOTA', '100'))
DAILY_IMAGE_QUOTA = int(os.getenv('ENKI_BOT_DAILY_IMAGE_QUOTA', '20'))

# Zero-Trust moderation configuration. These values control how the spam
# detector treats new or unverified users and how it scores their first
# messages. All values can be overridden with environment variables.
ZERO_TRUST_SETTINGS = {
    "watch_new_user_window_sec": int(os.getenv('ENKI_BOT_ZT_NEW_USER_WINDOW', '3600')),
    "watch_first_messages": int(os.getenv('ENKI_BOT_ZT_FIRST_MESSAGES', '5')),
    "global_thresholds": {
        "delete": float(os.getenv('ENKI_BOT_ZT_DELETE_THRESHOLD', '0.75')),
        "ban": float(os.getenv('ENKI_BOT_ZT_BAN_THRESHOLD', '0.95')),
        "mute_then_captcha": float(os.getenv('ENKI_BOT_ZT_MUTE_THRESHOLD', '0.80')),
    },
    "heuristics": {
        "url_in_first_msg": float(os.getenv('ENKI_BOT_ZT_URL_RISK', '0.10')),
        "many_mentions": float(os.getenv('ENKI_BOT_ZT_MANY_MENTIONS_RISK', '0.10')),
        "all_caps_long": float(os.getenv('ENKI_BOT_ZT_ALL_CAPS_RISK', '0.10')),
        "repeated_chars": float(os.getenv('ENKI_BOT_ZT_REPEAT_RISK', '0.05')),
        "keyword_hits": float(os.getenv('ENKI_BOT_ZT_KEYWORD_RISK', '0.15')),
    },
    "lists": {
        "domain_blocklist": set(filter(None, os.getenv(
            'ENKI_BOT_ZT_DOMAIN_BLOCKLIST', 't.me/joinchat,bit.ly,cutt.ly'
        ).split(','))),
        "keyword_blocklist": set(filter(None, os.getenv(
            'ENKI_BOT_ZT_KEYWORD_BLOCKLIST',
            'free crypto,giveaway,xxx,adult cam,telegram view,loan approval,work from home $,forex vip'
        ).split(','))),
    },
    "logging": {
        "admin_chat_id": int(os.getenv('ENKI_BOT_ZT_ADMIN_CHAT_ID'))
        if os.getenv('ENKI_BOT_ZT_ADMIN_CHAT_ID')
        else None,
    },
}

# Community moderation settings
DEFAULT_SPAM_VOTE_THRESHOLD = int(os.getenv('ENKI_BOT_SPAM_VOTE_THRESHOLD', '3'))
SPAM_VOTE_TIME_WINDOW_MINUTES = int(os.getenv('ENKI_BOT_SPAM_VOTE_WINDOW_MINUTES', '60'))
REPORTS_CHANNEL_ID = int(os.getenv('ENKI_BOT_REPORTS_CHANNEL_ID')) if os.getenv('ENKI_BOT_REPORTS_CHANNEL_ID') else None

# Fact-check / maintenance admin user IDs (comma-separated Telegram user IDs). Optional.
_factcheck_admin_ids_raw = os.getenv('ENKI_BOT_FACTCHECK_ADMIN_IDS', '')
FACTCHECK_ADMIN_USER_IDS = set()
if _factcheck_admin_ids_raw:
    try:
        FACTCHECK_ADMIN_USER_IDS = set(int(x.strip()) for x in _factcheck_admin_ids_raw.split(',') if x.strip())
    except ValueError:
        logging.warning("Invalid ENKI_BOT_FACTCHECK_ADMIN_IDS format; ignoring.")
# Fallback: include REPORTS_CHANNEL_ID if defined (so existing single-admin setups continue to work)
if REPORTS_CHANNEL_ID:
    FACTCHECK_ADMIN_USER_IDS.add(REPORTS_CHANNEL_ID)

# NSFW filtering
NSFW_FILTER_DEFAULT_ENABLED = os.getenv('ENKI_BOT_NSFW_FILTER_DEFAULT', 'false').lower() == 'true'
NSFW_DETECTION_THRESHOLD = float(os.getenv('ENKI_BOT_NSFW_THRESHOLD', '0.8'))

# Timeout settings for Telegram HTTP requests (in seconds).
TELEGRAM_CONNECT_TIMEOUT = float(os.getenv('ENKI_BOT_TELEGRAM_CONNECT_TIMEOUT', '30'))
TELEGRAM_READ_TIMEOUT = float(os.getenv('ENKI_BOT_TELEGRAM_READ_TIMEOUT', '30'))
# The write timeout is rarely distinct from the read timeout, but httpx
# requires it to be provided when specifying individual timeouts. Use the
# read timeout as the default if no explicit environment variable is set.
TELEGRAM_WRITE_TIMEOUT = float(
    os.getenv('ENKI_BOT_TELEGRAM_WRITE_TIMEOUT', str(TELEGRAM_READ_TIMEOUT))
)
# httpx also expects a pool timeout when customising timeouts. Default to the
# connect timeout so existing behaviour is preserved without requiring an
# additional setting from the user.
TELEGRAM_POOL_TIMEOUT = float(
    os.getenv('ENKI_BOT_TELEGRAM_POOL_TIMEOUT', str(TELEGRAM_CONNECT_TIMEOUT))
)

# --- Database Configuration (MS SQL Server) ---
SQL_SERVER_NAME = os.getenv('ENKI_BOT_SQL_SERVER_NAME')
SQL_DATABASE_NAME = os.getenv('ENKI_BOT_SQL_DATABASE_NAME')
DB_CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_SERVER_NAME};"
    f"DATABASE={SQL_DATABASE_NAME};"
    f"Trusted_Connection=yes;"
) if SQL_SERVER_NAME and SQL_DATABASE_NAME else None

# --- LLM Provider API Keys & Models ---
# OpenAI
OPENAI_API_KEY = os.getenv('ENKI_BOT_OPENAI_API_KEY')
OPENAI_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_MODEL_ID', 'gpt-4o-mini')                 # General orchestrator (vision-capable)
# Default deep‑research model. Override if you have access to o3-deep-research, etc.
OPENAI_DEEP_RESEARCH_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_DEEP_RESEARCH_MODEL_ID', 'gpt-4o-mini')
# Vision-capable model to use for OCR / image understanding (chat with images)
OPENAI_MULTIMODAL_IMAGE_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_MULTIMODAL_IMAGE_MODEL_ID', 'gpt-4o')
OPENAI_EMBEDDING_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_EMBEDDING_MODEL_ID', 'text-embedding-3-large')
OPENAI_CLASSIFICATION_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_CLASSIFICATION_MODEL_ID', 'gpt-4o-mini')
OPENAI_TRANSLATION_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_TRANSLATION_MODEL_ID', 'gpt-4o-mini')
OPENAI_DALLE_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_DALLE_MODEL_ID', 'dall-e-3')
OPENAI_WHISPER_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_WHISPER_MODEL_ID', 'whisper-1')
OPENAI_SEARCH_CONTEXT_SIZE = os.getenv('ENKI_BOT_OPENAI_SEARCH_CONTEXT_SIZE')
OPENAI_SEARCH_USER_LOCATION = os.getenv('ENKI_BOT_OPENAI_SEARCH_USER_LOCATION')
DEFAULT_IMAGE_N = 1
DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "standard"
# Groq
GROQ_API_KEY = os.getenv('ENKI_BOT_GROQ_API_KEY')
GROQ_MODEL_ID = os.getenv('ENKI_BOT_GROQ_MODEL_ID', 'llama-3.3-70b-versatile')
GROQ_ENDPOINT_URL = "https://api.groq.com/openai/v1/chat/completions"

# OpenRouter
OPENROUTER_API_KEY = os.getenv('ENKI_BOT_OPENROUTER_API_KEY')
OPENROUTER_MODEL_ID = os.getenv('ENKI_BOT_OPENROUTER_MODEL_ID', 'mistralai/mistral-7b-instruct:free')
OPENROUTER_ENDPOINT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Google AI
GOOGLE_AI_API_KEY = os.getenv('ENKI_BOT_GOOGLE_AI_API_KEY')
GOOGLE_AI_MODEL_ID = os.getenv('ENKI_BOT_GOOGLE_AI_MODEL_ID', 'gemini-1.5-flash-latest')

# Estimated cost per 1K tokens for tracking purposes (USD)
OPENAI_COST_PER_1K_TOKENS = float(os.getenv('ENKI_BOT_OPENAI_COST_PER_1K_TOKENS', '0'))
GROQ_COST_PER_1K_TOKENS = float(os.getenv('ENKI_BOT_GROQ_COST_PER_1K_TOKENS', '0'))
OPENROUTER_COST_PER_1K_TOKENS = float(os.getenv('ENKI_BOT_OPENROUTER_COST_PER_1K_TOKENS', '0'))

# --- External Service API Keys ---
NEWS_API_KEY = os.getenv('ENKI_BOT_NEWS_API_KEY')
WEATHER_API_KEY = os.getenv('ENKI_BOT_WEATHER_API_KEY')

# --- Group Chat Access Control ---
ALLOWED_GROUP_IDS_STR = os.getenv('ENKI_BOT_ALLOWED_GROUP_IDS')
ALLOWED_GROUP_IDS = set()
if ALLOWED_GROUP_IDS_STR:
    try:
        ALLOWED_GROUP_IDS = set(int(id_str.strip()) for id_str in ALLOWED_GROUP_IDS_STR.split(','))
        if ALLOWED_GROUP_IDS:
            logging.info(f"Bot access is restricted to group IDs: {ALLOWED_GROUP_IDS}")
    except ValueError:
        logging.error(f"Invalid format for ENKI_BOT_ALLOWED_GROUP_IDS: '{ALLOWED_GROUP_IDS_STR}'. No group restrictions applied.")
else:
    logging.info("ENKI_BOT_ALLOWED_GROUP_IDS is not set. The bot will operate in all groups.")

# --- Language and Prompts Configuration ---
# The default language to use if detection fails.
DEFAULT_LANGUAGE = "en"
# The directory where language-specific prompt files (e.g., en.json, ru.json) are stored.
LANGUAGE_PACKS_DIR = os.path.join(os.path.dirname(__file__), 'lang')

# Number of independent sources required to consider a claim "confirmed"
FACTCHECK_CONFIRMATION_THRESHOLD = int(os.getenv('ENKI_BOT_FACTCHECK_CONFIRM_THRESHOLD', '3'))
# Domains that must NOT be used as confirmations for forwarded news (comma-separated)
FACTCHECK_DOMAIN_BLOCKLIST = set(
    d.strip().lower() for d in os.getenv('ENKI_BOT_FACTCHECK_DOMAIN_BLOCKLIST', 't.me,telegram.me').split(',') if d.strip()
)
# Tracking / analytics query parameters to strip from gathered evidence URLs (comma-separated).
# Supports wildcard prefixes by ending an entry with '*', e.g. 'utm_*' will match utm_source, utm_campaign, etc.
FACTCHECK_TRACKING_PARAM_BLOCKLIST = [
    p.strip().lower() for p in os.getenv(
        'ENKI_BOT_FACTCHECK_TRACKING_PARAMS',
        'utm_*,ref,referrer,fbclid,gclid,gclsrc,igshid,mc_cid,mc_eid,ga_source,ga_medium,ga_campaign,ga_content'
    ).split(',') if p.strip()
]
# Persistent URL sanitize cache configuration
FACTCHECK_URL_CACHE_PATH = os.getenv('ENKI_BOT_FACTCHECK_URL_CACHE_PATH', os.path.join(os.getcwd(), 'factcheck_url_cache.json'))
FACTCHECK_URL_CACHE_MAX_SIZE = int(os.getenv('ENKI_BOT_FACTCHECK_URL_CACHE_MAX', '5000'))
FACTCHECK_URL_CACHE_PERSIST_INTERVAL_SEC = int(os.getenv('ENKI_BOT_FACTCHECK_URL_CACHE_PERSIST_SEC', '300'))

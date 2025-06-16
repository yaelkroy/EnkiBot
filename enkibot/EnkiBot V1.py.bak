# ==================================================================================================
# === EnkiBot - Умный ассистент для Telegram с долгосрочной памятью и AI-функциями ===
# ==================================================================================================
#
# Ключевые особенности:
# - Оркестрация нескольких LLM: Одновременные запросы к разным моделям (OpenAI, Groq и др.)
#   с выбором самого быстрого ответа для скорости и надежности.
# - Долгосрочная память: Интеграция с базой данных MS SQL Server для хранения истории чатов,
#   профилей пользователей и логов.
# - Контекстный поиск по памяти: Возможность спрашивать о прошлых событиях или людях в чате.
#   Бот динамически находит релевантную информацию в логах.
# - Динамическое распознавание имен: Понимает имена пользователей (включая транслит) и уточняет,
#   если запрос неоднозначен.
# - Автоматическое профилирование пользователей: Бот анализирует сообщения пользователя для
#   создания и обновления профиля его интересов.
# - Встроенные функции: Получение новостей и актуального прогноза погоды.
#
# ==================================================================================================

import logging
import json
import os
import openai # Для OpenAI API
import pyodbc # Для MS SQL Server
import requests # Для синхронных запросов к News API
import httpx # Для асинхронных HTTP-запросов к API
import traceback # Для детального логирования исключений
import re # Для регулярных выражений
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction
import asyncio
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pymorphy3 # Для морфологии русского языка
from datetime import datetime # Для работы с датой и временем
from transliterate import translit
from transliterate.exceptions import LanguagePackNotFound # ИСПРАВЛЕННЫЙ ИМПОРТ

# --- Инициализация библиотек ---
DetectorFactory.seed = 0
morph = pymorphy3.MorphAnalyzer()

# --- Конфигурация: Загрузка ключей и настроек из переменных окружения ---
# Важно: Никогда не храните ключи API прямо в коде.
TELEGRAM_BOT_TOKEN = os.getenv('ENKI_BOT_TOKEN')
NEWS_API_KEY = os.getenv('ENKI_BOT_NEWS_API_KEY')
WEATHER_API_KEY = os.getenv('ENKI_BOT_WEATHER_API_KEY') # Ключ для OpenWeatherMap

# Настройки подключения к базе данных
SQL_SERVER_NAME = os.getenv('ENKI_BOT_SQL_SERVER_NAME')
SQL_DATABASE_NAME = os.getenv('ENKI_BOT_SQL_DATABASE_NAME')

# Ключи и модели для различных LLM-провайдеров
OPENAI_API_KEY = os.getenv('ENKI_BOT_OPENAI_API_KEY')
OPENAI_MODEL_ID = os.getenv('ENKI_BOT_OPENAI_MODEL_ID', 'gpt-4o-mini')
GROQ_API_KEY = os.getenv('ENKI_BOT_GROQ_API_KEY')
GROQ_MODEL_ID = os.getenv('ENKI_BOT_GROQ_MODEL_ID', 'llama3-8b-8192')
GROQ_ENDPOINT_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv('ENKI_BOT_OPENROUTER_API_KEY')
OPENROUTER_MODEL_ID = os.getenv('ENKI_BOT_OPENROUTER_MODEL_ID', 'mistralai/mistral-7b-instruct:free')
OPENROUTER_ENDPOINT_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_API_KEY = os.getenv('ENKI_BOT_GOOGLE_AI_API_KEY')
GOOGLE_AI_MODEL_ID = os.getenv('ENKI_BOT_GOOGLE_AI_MODEL_ID', 'gemini-1.5-flash-latest')

# --- Константы ---
# Имена и никнеймы, на которые бот будет реагировать в групповых чатах
BOT_NICKNAMES_TO_CHECK = [ "enki", "enkibot", "энки", "энкибот", "бот", "bot" ]

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot_activity.log", encoding='utf-8'), # Логи в файл
        logging.StreamHandler() # Логи в консоль
    ]
)
logger = logging.getLogger(__name__)

# --- Конфигурация разрешенных групп ---
ALLOWED_GROUP_IDS_STR = os.getenv('ENKI_BOT_ALLOWED_GROUP_IDS') 
ALLOWED_GROUP_IDS = set()
if ALLOWED_GROUP_IDS_STR:
    try:
        # Эта логика для разбора строки у вас правильная
        ALLOWED_GROUP_IDS = set(int(id_str.strip()) for id_str in ALLOWED_GROUP_IDS_STR.split(','))
        if ALLOWED_GROUP_IDS:
            logger.info(f"Бот ограничен группами с ID: {ALLOWED_GROUP_IDS}")
    except ValueError:
        # Эта логика для обработки ошибок у вас тоже правильная
        logger.error(f"Неверный формат ENKI_BOT_ALLOWED_GROUP_IDS: '{ALLOWED_GROUP_IDS_STR}'. Ограничение по группам снято.")
        ALLOWED_GROUP_IDS = set()
else:
    logger.info("ENKI_BOT_ALLOWED_GROUP_IDS не задан. Бот будет работать во всех группах.")


# --- Подключение к базе данных и инициализация ---
DB_CONNECTION_STRING = None
if SQL_SERVER_NAME and SQL_DATABASE_NAME:
    DB_CONNECTION_STRING = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER_NAME};"
        f"DATABASE={SQL_DATABASE_NAME};"
        f"Trusted_Connection=yes;"
    )
    logger.info(f"Строка подключения к БД сконфигурирована для {SQL_SERVER_NAME}/{SQL_DATABASE_NAME}")
else:
    logger.warning("Переменные окружения для SQL Server не заданы. Функционал БД отключен.")

def get_db_connection():
    if not DB_CONNECTION_STRING: return None
    try: return pyodbc.connect(DB_CONNECTION_STRING, autocommit=False)
    except pyodbc.Error as ex:
        logger.error(f"Ошибка подключения к БД: {ex.args[0]} - {ex}")
        return None

def initialize_database():
    # Эта функция проверяет наличие необходимых таблиц в БД и создает их, если они отсутствуют.
    # Вызывается один раз при старте бота.
    if not DB_CONNECTION_STRING:
        logger.warning("Невозможно инициализировать БД: строка подключения не настроена.")
        return
    # Словарь с запросами на создание таблиц и индексов к ним для ускорения выборок
    table_creation_queries = {
        "ConversationHistory": "CREATE TABLE ConversationHistory (MessageDBID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageID BIGINT NULL, Role NVARCHAR(50) NOT NULL, Content NVARCHAR(MAX) NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ConversationHistory_ChatID_Timestamp": "CREATE INDEX IX_ConversationHistory_ChatID_Timestamp ON ConversationHistory (ChatID, Timestamp DESC);",
        "ChatLog": "CREATE TABLE ChatLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, MessageID BIGINT NOT NULL, MessageText NVARCHAR(MAX) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ChatLog_ChatID_Timestamp": "CREATE INDEX IX_ChatLog_ChatID_Timestamp ON ChatLog (ChatID, Timestamp DESC);",
        "IX_ChatLog_UserID": "CREATE INDEX IX_ChatLog_UserID ON ChatLog (UserID);",
        "ErrorLog": "CREATE TABLE ErrorLog (ErrorID INT IDENTITY(1,1) PRIMARY KEY, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, LogLevel NVARCHAR(50) NOT NULL, LoggerName NVARCHAR(255) NULL, ModuleName NVARCHAR(255) NULL, FunctionName NVARCHAR(255) NULL, LineNumber INT NULL, ErrorMessage NVARCHAR(MAX) NOT NULL, ExceptionInfo NVARCHAR(MAX) NULL);",
        "IX_ErrorLog_Timestamp": "CREATE INDEX IX_ErrorLog_Timestamp ON ErrorLog (Timestamp DESC);",
        "UserProfiles": "CREATE TABLE UserProfiles (UserID BIGINT PRIMARY KEY, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, LastName NVARCHAR(255) NULL, LastSeen DATETIME2 DEFAULT GETDATE(), MessageCount INT DEFAULT 0, PreferredLanguage NVARCHAR(10) NULL, Notes NVARCHAR(MAX) NULL, ProfileLastUpdated DATETIME2 DEFAULT GETDATE());",
        # --- НАЧАЛО НОВЫХ СТРОК ---
        "UserNameVariations": "CREATE TABLE UserNameVariations (VariationID INT IDENTITY(1,1) PRIMARY KEY, UserID BIGINT NOT NULL, NameVariation NVARCHAR(255) NOT NULL, FOREIGN KEY (UserID) REFERENCES UserProfiles(UserID) ON DELETE CASCADE);",
        "IX_UserNameVariations_NameVariation": "CREATE UNIQUE INDEX IX_UserNameVariations_NameVariation ON UserNameVariations (UserID, NameVariation);"
        # --- КОНЕЦ НОВЫХ СТРОК ---
    }
    conn = None
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING, autocommit=True)
        cursor = conn.cursor()
        logger.info("Проверка и создание таблиц БД при необходимости...")
        for item_name, query in table_creation_queries.items():
            is_index = item_name.startswith("IX_")
            table_name_for_check = item_name.split("_")[1] if is_index else item_name
            if is_index:
                try:
                    cursor.execute(query)
                    logger.info(f"Индекс {item_name} создан или подтвержден.")
                except pyodbc.ProgrammingError as pe:
                    if "already an index" in str(pe).lower() or "already exists" in str(pe).lower():
                        logger.info(f"Индекс {item_name} уже существует.")
                    else: raise
                continue
            check_sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
            cursor.execute(check_sql, table_name_for_check)
            if cursor.fetchone(): logger.info(f"Таблица {table_name_for_check} уже существует.")
            else:
                logger.info(f"Таблица {table_name_for_check} не найдена. Создание..."); cursor.execute(query); logger.info(f"Таблица {table_name_for_check} создана.")
        cursor.close(); logger.info("Проверка инициализации БД завершена.")
    except pyodbc.Error as ex: logger.error(f"Ошибка инициализации БД: {ex}", exc_info=True)
    except Exception as e: logger.error(f"Неожиданная ошибка во время инициализации БД: {e}", exc_info=True)
    finally:
        if conn: conn.close()

# --- Логирование ошибок в БД ---
class SQLDBLogHandler(logging.Handler):
    # Этот класс позволяет автоматически записывать критические ошибки Python прямо в таблицу ErrorLog в БД.
    def __init__(self): super().__init__(); self.conn = None
    def _get_db_conn_for_logging(self):
        if not DB_CONNECTION_STRING: return None
        try: return pyodbc.connect(DB_CONNECTION_STRING, autocommit=True)
        except pyodbc.Error: return None
    def emit(self, record: logging.LogRecord):
        if self.conn is None: self.conn = self._get_db_conn_for_logging()
        if self.conn:
            try:
                msg, exc_info_str = self.format(record), traceback.format_exc() if record.exc_info else None
                sql = "INSERT INTO ErrorLog (LogLevel, LoggerName, ModuleName, FunctionName, LineNumber, ErrorMessage, ExceptionInfo) VALUES (?, ?, ?, ?, ?, ?, ?)"
                with self.conn.cursor() as c: c.execute(sql, record.levelname, record.name, record.module, record.funcName, record.lineno, msg, exc_info_str)
            except: self.handleError(record); self.conn = None
    def close(self):
        if self.conn:
            try: self.conn.close()
            except: pass
        super().close()

if DB_CONNECTION_STRING:
    db_log_handler = SQLDBLogHandler(); db_log_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    db_log_handler.setFormatter(formatter); logging.getLogger().addHandler(db_log_handler)
    logger.info("Настроено логирование ошибок в SQL БД.")
else: logger.warning("Логирование ошибок в SQL БД НЕ настроено.")

# --- Инициализация клиентов API ---
openai_async_client = None
if OPENAI_API_KEY:
    try: openai_async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY); logger.info("Клиент OpenAI AsyncOpenAI инициализирован.")
    except Exception as e: logger.error(f"Не удалось инициализировать клиент OpenAI AsyncOpenAI: {e}")
else: logger.warning("Ключ OpenAI API не найден. Вызовы к OpenAI отключены.")

# --- Вспомогательные функции ---
# --- В разделе "Вспомогательные функции" ---

# Эта функция находится в вашем основном файле EnkiBot.py

async def populate_name_variations_with_llm(user_id: int, first_name: str, last_name: str | None, username: str | None):
    """
    Использует LLM для генерации ЛИНГВИСТИЧЕСКИХ вариантов имени пользователя,
    включая уменьшительные формы, транслитерацию и падежи.
    """
    if not openai_async_client:
        logger.warning(f"Генерация вариантов имени для user {user_id} пропущена: OpenAI клиент не настроен.")
        return

    name_parts = [part for part in [first_name, last_name, username] if part]
    name_info = ", ".join(name_parts)
    
    logger.info(f"Запрос на ЛИНГВИСТИЧЕСКУЮ генерацию вариантов для пользователя {user_id} ({name_info}).")

    # --- НОВЫЙ, СУПЕР-СФОКУСИРОВАННЫЙ ПРОМПТ ---
    system_prompt = (
        "You are a language expert specializing in Russian and English names. Your task is to generate a list of linguistic variations for a user's name. Focus ONLY on realistic, human-used variations. DO NOT generate technical usernames with numbers or suffixes like '_dev'."
        "\n\n**Goal:** Create variations for recognition in natural language text."
        "\n\n**Categories for Generation:**"
        "\n1.  **Original Forms:** The original first name, last name, and combinations."
        "\n2.  **Diminutives & Nicknames:** Common short and affectionate forms (e.g., 'Антонина' -> 'Тоня'; 'Robert' -> 'Rob')."
        "\n3.  **Transliteration (with variants):** Provide multiple common Latin spellings for all Cyrillic forms (original and diminutives). Example for 'Тоня': 'tonya', 'tonia'."
        "\n4.  **Reverse Transliteration:** If the source name is Latin, provide plausible Cyrillic versions. Example for 'Yael': 'Яэль', 'Йаэль'."
        "\n5.  **Russian Declensions (Grammatical Cases):** For all primary Russian names (full and short forms), provide their forms in different grammatical cases (genitive, dative, accusative, instrumental, prepositional). Example for 'Саша': 'саши', 'саше', 'сашу', 'сашей', 'о саше'."
        "\n\n**Output Format:** Return a single JSON object: {\"variations\": [\"variation1\", \"variation2\", ...]}. All variations must be in lowercase."
    )
    
    user_prompt = f"Generate linguistic variations for the user with the following info: {name_info}"

    messages_for_api = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    name_variations = set()
    try:
        completion = await openai_async_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages_for_api,
            temperature=0.3, # Снижаем температуру для более предсказуемых, основанных на правилах результатов
            response_format={"type": "json_object"}
        )
        if completion.choices and completion.choices[0].message:
            response_str = completion.choices[0].message.content
            try:
                data = json.loads(response_str)
                variations_list = data.get('variations')
                if isinstance(variations_list, list):
                    name_variations.update([str(v).lower().strip() for v in variations_list if v and str(v).strip()])
                    logger.info(f"LLM сгенерировала {len(variations_list)} лингвистических вариантов для user {user_id}.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Ошибка декодирования JSON от LLM для user {user_id}: {response_str}. Ошибка: {e}")
    except Exception as e:
        logger.error(f"Ошибка OpenAI при генерации вариантов имени для {user_id}: {e}")

    # Добавляем оригинальные имена еще раз
    name_variations.update([p.lower() for p in name_parts])
    
    # Сохраняем все уникальные варианты в БД
    if name_variations:
        db_conn = get_db_connection()
        if db_conn:
            try:
                with db_conn.cursor() as cursor:
                    sql = """
                        MERGE INTO UserNameVariations AS t
                        USING (SELECT ? AS UserID, ? AS NameVariation) AS s
                        ON (t.UserID = s.UserID AND t.NameVariation = s.NameVariation)
                        WHEN NOT MATCHED THEN
                            INSERT (UserID, NameVariation) VALUES (s.UserID, s.NameVariation);
                    """
                    params_to_insert = [(user_id, var) for var in name_variations if var]
                    if params_to_insert:
                        cursor.executemany(sql, params_to_insert)
                        db_conn.commit()
                        logger.info(f"Сохранено/обновлено {len(params_to_insert)} вариантов имени для user {user_id}.")
            except pyodbc.Error as ex:
                logger.error(f"Ошибка БД при сохранении вариантов имени для {user_id}: {ex}")
                if db_conn: db_conn.rollback()
            finally:
                if db_conn: db_conn.close()
def get_translit_variations(name: str) -> set[str]:
    """
    Создает набор вариаций имени, включая оригинал и его транслитерацию.
    Работает в обе стороны: с кириллицы на латиницу и наоборот.
    """
    variations = {name.lower()}
    try:
        # Проверяем, содержит ли имя кириллические символы
        if re.search('[а-яА-Я]', name):
            translit_name = translit(name, 'ru', reversed=True) # ru -> en
            variations.add(translit_name.lower())
        else:
            translit_name = translit(name, 'ru') # en -> ru
            variations.add(translit_name.lower())
    except LanguagePackNotFound: # ИСПРАВЛЕННАЯ ОШИБКА
        logger.warning(f"Пакет для транслитерации 'ru' не найден. Пропускаем для имени: {name}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при транслитерации имени {name}: {e}")
    return variations
async def analyze_replied_message(original_text: str, user_question: str) -> str:
    """
    Анализирует исходный текст (на который ответили) в контексте вопроса пользователя.
    """
    logger.info(f"Запрос на анализ текста. Длина исходного текста: {len(original_text)}, Вопрос: '{user_question}'")
    
    # Создаем очень четкий промпт для LLM, чтобы она поняла свою задачу
    system_prompt = (
        "Ты — AI-аналитик. Твоя задача — проанализировать 'Исходный текст' и дать содержательный ответ на 'Вопрос пользователя' об этом тексте. "
        "Твой анализ должен быть объективным, кратким и по существу. Если вопрос общий (например, 'что думаешь?'), "
        "сделай краткое резюме, выделив ключевые тезисы или настроения в исходном тексте."
    )

    user_prompt = f"""
Исходный текст для анализа:
---
"{original_text}"
---

Вопрос пользователя об этом тексте:
---
"{user_question}"
---

Твой анализ:
"""

    messages_for_api = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Для этой задачи можно использовать любую из ваших LLM. OpenAI хорошо подходит для анализа.
    if openai_async_client:
        try:
            completion = await openai_async_client.chat.completions.create(
                model='gpt-4o-mini', # Быстрая и умная модель для таких задач
                messages=messages_for_api,
                temperature=0.5, # Чуть больше креативности для анализа
                max_tokens=1000
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка OpenAI при анализе сообщения: {e}", exc_info=True)
            return "К сожалению, произошла ошибка во время анализа текста."
    
    return "Функция анализа не может быть выполнена, так как AI-клиент не настроен."
# --- Функции для вызова API ---
async def call_openai_llm(messages_for_api: list) -> str | None:
    if not openai_async_client: logger.warning("Клиент OpenAI недоступен."); return None
    logger.info(f"Вызов OpenAI (модель: {OPENAI_MODEL_ID}) с {len(messages_for_api)} сообщениями контекста.")
    try:
        completion = await openai_async_client.chat.completions.create(model=OPENAI_MODEL_ID, messages=messages_for_api, temperature=0.7, max_tokens=2000)
        if completion.choices and completion.choices[0].message: return completion.choices[0].message.content.strip()
    except Exception as e: logger.error(f"Ошибка при работе с OpenAI API: {e}", exc_info=True)
    return None

async def call_llm_api(p_name: str, key: str | None, url: str | None, model: str, msgs: list) -> str | None:
    # Универсальная функция для вызова LLM API, совместимых с OpenAI (Groq, OpenRouter)
    if not key: logger.warning(f"Ключ API для {p_name} недоступен."); return None
    if not url: logger.warning(f"URL для {p_name} недоступен."); return None
    logger.info(f"Вызов {p_name} ({model}) с {len(msgs)} сообщениями контекста.")
    hdrs = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if p_name == "OpenRouter": hdrs.update({"HTTP-Referer": "YOUR_PROJECT_URL", "X-Title": "EnkiBot"})
    payload = {"model": model, "messages": msgs, "max_tokens": 2000, "temperature": 0.7}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=hdrs, timeout=30.0); resp.raise_for_status()
            data = resp.json()
            if data.get("choices") and data["choices"][0].get("message"): return data["choices"][0]["message"].get("content","").strip()
    except Exception as e: logger.error(f"Ошибка при работе с API {p_name}: {e}", exc_info=True)
    return None

async def call_google_ai_llm_specific(messages_for_api: list) -> str | None:
    # Отдельная функция для Google AI, так как их SDK имеет другой формат
    if not GOOGLE_AI_API_KEY: logger.warning("Ключ Google AI API недоступен."); return None
    try:
        import google.generativeai as genai; genai.configure(api_key=GOOGLE_AI_API_KEY)
        sys_instr, gem_hist, final_prompt = "You are a helpful assistant.", [], ""
        # ... (логика преобразования истории в формат Gemini) ...
        # Эта часть кода требует тщательной адаптации формата сообщений
        return "Google AI call not fully implemented in this example"
    except ImportError: logger.error("Библиотека google.generativeai не установлена.")
    except Exception as e: logger.error(f"Ошибка Google AI API: {e}", exc_info=True)
    return None

# --- Основные функции бота ---
async def analyze_weather_request_with_llm(text: str) -> dict:
    """
    Анализирует запрос о погоде и определяет, нужна ли текущая погода или прогноз.
    Возвращает словарь, например: {'type': 'forecast', 'days': 7} или {'type': 'current'}.
    """
    logger.info(f"Анализ типа погодного запроса из текста: '{text}'")
    
    system_prompt = (
        "You are an expert in analyzing weather-related requests. Your task is to determine the user's intent. "
        "Does the user want the 'current' weather or a 'forecast' for several days? "
        "If it is a forecast, also determine for how many days. Your answer MUST be a valid JSON object and nothing else. "
    
        "Examples:\n"
        # --- Basic Cases ---
        "- User text: 'погода в Лондоне' -> Your response: {\"type\": \"current\"}\n"
        "- User text: 'what's the weather like?' -> Your response: {\"type\": \"current\"}\n"
    
        # --- Multi-Day Forecast Cases ---
        "- User text: 'погода в Тампе на неделю' -> Your response: {\"type\": \"forecast\", \"days\": 7}\n"
        "- User text: 'прогноз на 5 дней в Берлине' -> Your response: {\"type\": \"forecast\", \"days\": 5}\n"
    
        # --- NEW: Cases for specific or relative days ---
        "- User text: 'какая погода будет завтра?' -> Your response: {\"type\": \"forecast\", \"days\": 2}\n"
        "- User text: 'дай прогноз на субботу' -> Your response: {\"type\": \"forecast\", \"days\": 7}\n"
    
        # --- NEW: Cases for generic forecast requests ---
        "- User text: 'просто дай прогноз погоды' -> Your response: {\"type\": \"forecast\", \"days\": 5}\n"
        "- User text: 'прогноз на выходные' -> Your response: {\"type\": \"forecast\", \"days\": 3}\n"

        # --- Fallback Rule ---
        "If you are unsure, always default to 'current'."
    )
    
    messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    
    try:
        if openai_async_client:
            completion = await openai_async_client.chat.completions.create(
                model='gpt-4o-mini', messages=messages_for_api, temperature=0, response_format={"type": "json_object"}
            )
            if completion.choices and completion.choices[0].message:
                response_str = completion.choices[0].message.content.strip()
                logger.info(f"LLM вернула для анализа погоды: {response_str}")
                return json.loads(response_str)
    except Exception as e:
        logger.error(f"Ошибка LLM при анализе запроса погоды: {e}")

    # Возвращаем значение по умолчанию в случае ошибки
    return {"type": "current"}
async def extract_location_with_llm(text: str) -> str | None:
    """
    Использует LLM для извлечения названия города из текста пользователя.
    Возвращает название города на английском языке, готовое для API, или None.
    """
    logger.info(f"Запрос на извлечение локации из текста: '{text}'")
    
    # Промпт специально разработан, чтобы LLM вернула только название или 'None'
    system_prompt = (
        "You are an expert text analysis tool. Your task is to extract a city or location name from the user's text. "
        "Analyze the following text and identify the geographical location (city, region, country) mentioned. "
        "Return ONLY the name of the location in English, suitable for a weather API query. "
        "For example, if the text is 'какая погода в Санкт-Петербурге', you must return 'Saint Petersburg'. "
        "If the text is 'покажи погоду в Астане', you must return 'Astana'. "
        "If no specific location is found, you MUST return the single word: None"
    )
    
    messages_for_api = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    location = None
    
    # Пытаемся использовать OpenAI как приоритетный вариант
    if openai_async_client:
        try:
            completion = await openai_async_client.chat.completions.create(
                model='gpt-4o-mini', # Используем быструю и умную модель
                messages=messages_for_api,
                temperature=0, # Нам нужна точность, а не креативность
                max_tokens=50
            )
            if completion.choices and completion.choices[0].message:
                location = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка OpenAI при извлечении локации: {e}")

    # Если OpenAI не сработал, пробуем Groq как резервный вариант
    if not location and GROQ_API_KEY:
         location = await call_llm_api("Groq (Location)", GROQ_API_KEY, GROQ_ENDPOINT_URL, GROQ_MODEL_ID, messages_for_api)

    # Проверяем ответ от LLM. Если она вернула 'None' или пустую строку, считаем, что город не найден.
    if location and location.lower() != 'none' and location.strip() != "":
        logger.info(f"LLM успешно извлекла локацию: '{location}'")
        return location
    
    logger.warning("LLM не смогла извлечь локацию из текста.")
    return None
async def extract_news_topic_with_llm(text: str) -> str | None:
    """
    Использует LLM для извлечения темы/ключа для поиска новостей.
    """
    logger.info(f"Запрос на извлечение темы новостей из текста: '{text}'")
    
    system_prompt = (
        "You are an expert text analysis tool. Your task is to extract the main topic, keyword, or location from a user's request for news. "
        "Analyze the text. If it contains a specific subject, you MUST return that subject in its base (nominative) case and in the original language. "
        "For example, for a request 'новости в москве', you must return 'Москва'. For 'news about cars', return 'cars'. "
        "If the request is general (e.g., 'what's the news?', 'latest headlines'), you MUST return the single word: None"
    )
    
    messages_for_api = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    topic = None
    # Используем быструю модель для этой задачи
    if openai_async_client:
        try:
            completion = await openai_async_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages_for_api,
                temperature=0,
                max_tokens=50
            )
            if completion.choices and completion.choices[0].message:
                topic = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка OpenAI при извлечении темы новостей: {e}")

    if topic and topic.lower() != 'none' and topic.strip() != "":
        logger.info(f"LLM успешно извлекла тему новостей: '{topic}'")
        return topic
    
    logger.info("LLM не нашла конкретной темы, будут запрошены общие новости.")
    return None
from datetime import datetime, timedelta # Убедитесь, что timedelta импортирована

# Переименуем для ясности
async def get_weather_data(location: str, forecast_type: str = 'current', days: int = 7) -> str:
    """
    Получает данные о погоде: текущие или прогноз на несколько дней.
    """
    if not WEATHER_API_KEY:
        return "Функция погоды не настроена: отсутствует ключ API."

    # --- БЛОК ДЛЯ ТЕКУЩЕЙ ПОГОДЫ (остался почти без изменений) ---
    if forecast_type == 'current':
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ru"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
            data = response.json()
            city = data.get("name")
            desc = data["weather"][0].get("description")
            temp = data["main"].get("temp")
            feels = data["main"].get("feels_like")
            wind = data["wind"].get("speed")
            return (
                f"Погода в городе {city}:\n"
                f"  - Сейчас: {desc.capitalize()}\n"
                f"  - Температура: {temp:.1f}°C (ощущается как {feels:.1f}°C)\n"
                f"  - Ветер: {wind:.1f} м/с"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404: return f"К сожалению, я не смог найти город '{location}'."
            logger.error(f"HTTP ошибка при запросе погоды для {location}: {e}")
            return "Не удалось получить прогноз погоды из-за ошибки сервера."
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе погоды: {e}", exc_info=True)
            return "Произошла непредвиденная ошибка при запросе прогноза."

    # --- НОВЫЙ БЛОК ДЛЯ ПРОГНОЗА НА НЕСКОЛЬКО ДНЕЙ ---
    elif forecast_type == 'forecast':
        # Используем другой endpoint для прогноза
        url = "https://api.openweathermap.org/data/2.5/forecast"
        # Запрашиваем на 5 дней, так как это стандарт для бесплатного API
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ru", "cnt": 40} # 40 записей = 5 дней * 8 записей/день
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
            
            data = response.json()
            city = data.get("city", {}).get("name")
            forecast_list = data.get("list", [])
            
            if not forecast_list:
                return f"Не удалось получить прогноз для города '{location}'."

            daily_forecasts = {}
            for forecast in forecast_list:
                # Группируем по дням, чтобы избежать дубликатов
                day_str = datetime.fromtimestamp(forecast["dt"]).strftime('%Y-%m-%d')
                if day_str not in daily_forecasts:
                    daily_forecasts[day_str] = {
                        'day_name': datetime.fromtimestamp(forecast["dt"]).strftime('%A'), # Название дня недели
                        'temp': forecast['main']['temp'],
                        'description': forecast['weather'][0]['description']
                    }

            # Форматируем красивый ответ
            report_lines = [f"Прогноз погоды в городе {city} на ближайшие дни:"]
            for day_data in list(daily_forecasts.values())[:days]: # Ограничиваем кол-вом запрошенных дней
                report_lines.append(
                    f"  - {day_data['day_name'].capitalize()}: {day_data['temp']:.0f}°C, {day_data['description']}"
                )
            return "\n".join(report_lines)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404: return f"К сожалению, я не смог найти город '{location}' для прогноза."
            logger.error(f"HTTP ошибка при запросе прогноза для {location}: {e}")
            return "Не удалось получить прогноз погоды из-за ошибки сервера."
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе прогноза: {e}", exc_info=True)
            return "Произошла непредвиденная ошибка при запросе прогноза."
    
    return "Неизвестный тип запроса погоды."

async def analyze_and_update_user_profile(user_id: int, message_text: str):
    """
    Создает или обновляет структурированный психологический профиль пользователя,
    анализируя его сообщения.
    """
    if not openai_async_client:
        logger.warning(f"Профилирование для user {user_id} пропущено: OpenAI клиент не настроен.")
        return

    MAX_PROFILE_SIZE = 4000  # Увеличим размер, т.к. профиль стал более детальным
    logger.info(f"Запуск создания/обновления психологического профиля для пользователя {user_id}...")

    # Шаг 1: Получаем текущий профиль из БД
    current_profile_notes = ""
    db_conn = get_db_connection()
    if db_conn:
        try:
            with db_conn.cursor() as cursor:
                cursor.execute("SELECT Notes FROM UserProfiles WHERE UserID = ?", user_id)
                row = cursor.fetchone()
                if row and row[0]:
                    current_profile_notes = row[0]
        except pyodbc.Error as ex:
            logger.error(f"Ошибка БД при чтении профиля для {user_id}: {ex}")
        finally:
            db_conn.close()

    # Шаг 2: Выбираем стратегию и промпт (создание или обновление)
    
    system_prompt = ""
    user_prompt = ""

    if not current_profile_notes:
        # СТРАТЕГИЯ 1: СОЗДАНИЕ ПЕРВОНАЧАЛЬНОГО ПРОФИЛЯ
        logger.info(f"Существующий профиль для {user_id} не найден. Создание нового...")
        system_prompt = (
            "Ты — AI-психолог и профайлер. Твоя задача — создать первоначальный психологический портрет пользователя на основе его сообщения. "
            "Проанализируй текст на предмет стиля общения, возможных черт личности (используй модель 'Большая пятерка' как ориентир: Открытость, Добросовестность, Экстраверсия, Доброжелательность), а также ключевых интересов. "
            "Твой ответ ДОЛЖЕН быть структурирован строго по предложенному формату с заголовками Markdown. Будь объективен и основывайся только на предоставленном тексте."
        )
        user_prompt = f"""
            Проанализируй следующее сообщение от нового пользователя и создай его психологический профиль.

            Сообщение пользователя:
            ---
            "{message_text}"
            ---

            Твой результат (строго в формате Markdown):
            """
    else:
        # СТРАТЕГИЯ 2: ОБНОВЛЕНИЕ И СИНТЕЗ СУЩЕСТВУЮЩЕГО ПРОФИЛЯ
        logger.info(f"Обновление существующего профиля для {user_id}...")
        system_prompt = (
            "Ты — AI-психолог, обновляющий досье на пациента. Тебе предоставлены 'Существующий психологический профиль' и 'Новое сообщение' от пользователя. "
            "Твоя задача — не просто добавить новую информацию, а **переосмыслить и синтезировать весь профиль**. "
            "Если новое сообщение подтверждает черту — усиль ее описание. Если противоречит — скорректируй или смягчи. Если открывает что-то новое — интегрируй это в существующую структуру. "
            "Цель — получить эволюционировавший, но все еще лаконичный профиль. Сохраняй исходную структуру Markdown."
        )
        user_prompt = f"""
            Существующий психологический профиль:
            ---
            {current_profile_notes}
            ---

            Новое сообщение от пользователя для анализа:
            ---
            "{message_text}"
            ---

            Твой обновленный и переосмысленный психологический профиль:
            """

    # Шаг 3: Вызов LLM для анализа
    analysis_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    updated_profile_notes = None
    try:
        # Используем более мощную модель для сложных задач анализа
        completion = await openai_async_client.chat.completions.create(
            model='gpt-4o-mini',  # gpt-4o даст еще лучшие результаты
            messages=analysis_messages,
            temperature=0.5,
            max_tokens=1000 
        )
        if completion.choices and completion.choices[0].message:
            updated_profile_notes = completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка OpenAI при анализе профиля для {user_id}: {e}")

    # Шаг 4: Обновляем профиль в БД
    if updated_profile_notes and updated_profile_notes.strip():
        db_conn_update = get_db_connection()
        if db_conn_update:
            try:
                with db_conn_update.cursor() as cursor:
                    final_notes = updated_profile_notes[:MAX_PROFILE_SIZE]
                    sql = "UPDATE UserProfiles SET Notes = ?, ProfileLastUpdated = GETDATE() WHERE UserID = ?"
                    cursor.execute(sql, final_notes, user_id)
                    db_conn_update.commit()
                    logger.info(f"Успешно обновлен психологический профиль для пользователя {user_id}.")
            except pyodbc.Error as ex:
                logger.error(f"Ошибка БД при сохранении профиля для {user_id}: {ex}")
                if db_conn_update: db_conn_update.rollback()
            finally:
                if db_conn_update: db_conn_update.close()
    else:
        logger.warning(f"Анализ профиля не вернул результат для пользователя {user_id}.")
def find_search_query_in_text(text: str) -> str | None:
    """
    Анализирует текст, приводя слова к начальной форме (лемме),
    и ищет комбинацию триггерных слов и предлогов для извлечения имени.
    Возвращает имя пользователя для поиска или None.
    """
    # Словари триггеров (используем леммы - начальные формы слов)
    # Легко расширять, добавляя новые ключевые слова
    TELL_LEMMAS = {'рассказать', 'поведать', 'сообщить', 'описать'}
    INFO_LEMMAS = {'информация', 'инфо', 'справка', 'досье', 'данные'}
    WHO_LEMMAS = {'кто', 'что'}
    EXPLAIN_LEMMAS = {'пояснить', 'объяснить'}
    REMEMBER_LEMMAS = {'помнить', 'напомнить'}

    # Предлоги, которые обычно следуют за триггерами
    PREPOSITIONS = {'о', 'про', 'за', 'на', 'по'}

    words = re.findall(r"[\w'-]+", text.lower())
    
    for i, word in enumerate(words):
        try:
            # Получаем лемму слова
            lemma = morph.parse(word)[0].normal_form
            
            # Проверяем, является ли лемма одним из наших триггеров
            is_trigger = (lemma in TELL_LEMMAS or 
                          lemma in INFO_LEMMAS or 
                          lemma in WHO_LEMMAS or 
                          lemma in EXPLAIN_LEMMAS or 
                          lemma in REMEMBER_LEMMAS)

            if is_trigger:
                # Мы нашли триггерное слово. Теперь нужно найти имя, которое идет после него.
                # Индекс, с которого начинается имя
                start_index = i + 1
                
                # Если следующее слово - предлог, пропускаем его
                if start_index < len(words) and words[start_index] in PREPOSITIONS:
                    start_index += 1
                
                # Все, что идет дальше (до 3 слов), считаем именем
                if start_index < len(words):
                    # Захватываем от 1 до 3 слов после триггера/предлога
                    name_parts = words[start_index : start_index + 3]
                    return " ".join(name_parts)

        except Exception as e:
            logger.error(f"Ошибка при лемматизации слова '{word}': {e}")
            continue
            
    return None
async def get_orchestrated_llm_response(prompt_text: str, chat_id: int, user_id: int, message_id: int, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Это "мозг" бота. Функция определяет, нужно ли искать информацию в памяти,
    собирает весь необходимый контекст и управляет вызовами к LLM.
    """
    history_from_db, keyword_context_messages, profile_context_messages = [], [], []
    conn = get_db_connection()
    
    # Сначала получаем общую историю чата
    if conn:
        try:
            with conn.cursor() as c:
                MAX_RECENT_HISTORY = 100
                sql_hist = "SELECT TOP (?) Role, Content FROM ConversationHistory WHERE ChatID = ? ORDER BY Timestamp DESC"
                c.execute(sql_hist, MAX_RECENT_HISTORY, chat_id)
                history_from_db = [{"role": row.Role.lower(), "content": row.Content} for row in reversed(c.fetchall())]
        except pyodbc.Error as ex: 
            logger.error(f"DB error during history fetch: {ex}")
    

    # --- Шаг 1: Лингвистический поиск ключевой фразы ---
    search_term_original = find_search_query_in_text(prompt_text)

    if search_term_original:
        logger.info(f"Лингвистический анализ нашел запрос о пользователе: '{search_term_original}'")
    
    # --- Шаг 2: Поиск пользователя, его профиля и последних сообщений ---
    if conn and search_term_original:
        try:
            with conn.cursor() as c:
                # --- НАЧАЛО НОВОЙ ЛОГИКИ ПОИСКА ---
                # Теперь мы ищем точное совпадение в новой таблице вариантов имен.
                # Это быстрее и точнее, чем поиск с LIKE по нескольким полям.
                logger.info(f"Ищу UserID в таблице UserNameVariations по запросу: '{search_term_original.lower()}'")
                
                # Сначала находим ID всех пользователей, у которых есть такой вариант имени.
                sql_find_user_ids = "SELECT DISTINCT UserID FROM UserNameVariations WHERE NameVariation = ?"
                c.execute(sql_find_user_ids, search_term_original.lower())
                user_ids = [row.UserID for row in c.fetchall()]

                matched_profiles = []
                if user_ids:
                    # Если ID найдены, одним запросом получаем полные профили этих пользователей.
                    id_placeholders = ','.join('?' for _ in user_ids)
                    sql_find_user = f"""
                        SELECT UserID, FirstName, LastName, Username, Notes 
                        FROM UserProfiles 
                        WHERE UserID IN ({id_placeholders})
                    """
                    c.execute(sql_find_user, *user_ids)
                    matched_profiles = c.fetchall()
                
                logger.info(f"Найдено {len(matched_profiles)} профилей по запросу '{search_term_original}'.")
                # --- КОНЕЦ НОВОЙ ЛОГИКИ ПОИСКА ---

                # <<< НАЧАЛО БЛОКА СИНТЕЗА ДАННЫХ (остался без изменений) >>>
                if len(matched_profiles) == 1:
                    profile = matched_profiles[0]
                    target_user_id = profile.UserID
                    user_identifier = profile.FirstName or profile.Username or f"User ID {target_user_id}"

                    # 1. Получаем готовое досье из профиля
                    if profile.Notes and profile.Notes.strip():
                        profile_context_messages.append({
                            "role": "system",
                            "content": f"Важнейший контекст (готовое досье) по пользователю '{user_identifier}':\n---\n{profile.Notes}\n---"
                        })
                        logger.info(f"Загружен профиль (досье) для '{user_identifier}'.")

                    # 2. Получаем последние 50 сообщений из лога
                    logger.info(f"Запрашиваю до 50 последних сообщений для пользователя {user_identifier} (ID: {target_user_id})...")
                    sql_get_messages = """
                        SELECT TOP 50 MessageText 
                        FROM ChatLog
                        WHERE UserID = ? AND ChatID = ?
                        ORDER BY Timestamp DESC
                    """
                    c.execute(sql_get_messages, target_user_id, chat_id)
                    recent_messages_rows = c.fetchall()

                    if recent_messages_rows:
                        formatted_messages = "\n".join([f'- "{row.MessageText}"' for row in recent_messages_rows if row.MessageText and row.MessageText.strip()])
                        keyword_context_messages.append({
                            "role": "system",
                            "content": f"Дополнительный контекст для анализа (сырые данные): Вот до 50 последних сообщений от '{user_identifier}'. Используй их вместе с досье для составления самого актуального ответа.\n---\n{formatted_messages}\n---"
                        })
                        logger.info(f"Загружено {len(recent_messages_rows)} последних сообщений для анализа.")
                    else:
                        logger.info(f"Последние сообщения для {user_identifier} не найдены в логах этого чата.")
                # <<< КОНЕЦ БЛОКА СИНТЕЗА ДАННЫХ >>>

                elif len(matched_profiles) > 1:
                    user_options = [f"@{p.Username}" if p.Username else f"{p.FirstName or ''} {p.LastName or ''}".strip() for p in matched_profiles]
                    user_options = [opt for opt in user_options if opt]
                    logger.info(f"Найдено несколько пользователей: {user_options}. Запрашиваю уточнение.")
                    return f"Я нашел несколько пользователей с таким именем: {', '.join(user_options)}. О ком именно вы спрашиваете? Пожалуйста, уточните (можно по @username)."
                
                else:
                    logger.info(f"Профили не найдены для '{search_term_original}'.") # Сообщение остается, но теперь оно означает, что в таблице UserNameVariations нет такого имени.

        except pyodbc.Error as ex:
            logger.error(f"Ошибка БД при поиске в памяти: {ex}")
        finally:
            if conn:
                conn.close() # Закрываем соединение здесь, так как все операции с БД в этой функции завершены
                conn = None # Устанавливаем в None, чтобы избежать двойного закрытия

    # --- Шаг 3: Оркестрация LLM (остался без изменений) ---
    sys_prompt_content = (
        "Ты EnkiBot, умный и дружелюбный ассистент-аналитик в Telegram-чате, созданный Yael Demedetskaya. "
        "Твоя задача — помогать пользователям, отвечая на их вопросы. Ты обладаешь долгосрочной памятью о разговорах и профилях участников. "
        "Когда тебя просят рассказать о ком-то, твоя главная задача — СИНТЕЗИРОВАТЬ ИНФОРМАЦИЮ. "
        "Тебе будут предоставлены два типа данных: готовое досье из профиля и набор последних 'сырых' сообщений от этого человека. "
        "Проанализируй ОБА источника и составь на их основе новый, краткий, но содержательный и актуальный ответ. Не просто пересказывай досье, а обогащай его свежей информацией из сообщений. "
        "Отвечай развернуто, естественно и по-русски. Будь вежлив, но не слишком формален."
    )
    
    messages_for_api = [{"role": "system", "content": sys_prompt_content}] + profile_context_messages + keyword_context_messages + history_from_db + [{"role": "user", "content": prompt_text}]

    MAX_MSG_CTX = 40
    if len(messages_for_api) > MAX_MSG_CTX:
        sys_p = [m for m in messages_for_api if m["role"] == "system"]
        usr_hist = [m for m in messages_for_api if m["role"] != "system"]
        messages_for_api = sys_p + usr_hist[-(MAX_MSG_CTX - len(sys_p)):]

    tasks, task_names = [], []
    if openai_async_client: 
        tasks.append(call_openai_llm(messages_for_api))
        task_names.append("OpenAI")
    if GROQ_API_KEY: 
        tasks.append(call_llm_api("Groq", GROQ_API_KEY, GROQ_ENDPOINT_URL, GROQ_MODEL_ID, messages_for_api))
        task_names.append("Groq")

    if not tasks:
        return "Извините, ни один из моих AI-ассистентов сейчас не доступен."

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_reply = None
    for i, res in enumerate(results):
        if isinstance(res, str) and res.strip():
            logger.info(f"Используем успешный ответ от {task_names[i]}.")
            final_reply = res.replace('**', '')
            break
        elif isinstance(res, Exception):
            logger.error(f"Провайдер {task_names[i]} вернул ошибку: {res}")

    if not final_reply:
        final_reply = "К сожалению, я не смог получить четкий ответ. Пожалуйста, попробуйте еще раз."

    # Сохранение в историю вынесено из блока `if conn...`, так как соединение уже может быть закрыто
    db_conn_for_saving = get_db_connection()
    if db_conn_for_saving:
        try:
            with db_conn_for_saving.cursor() as c:
                sql_save = "INSERT INTO ConversationHistory (ChatID, UserID, MessageID, Role, Content) VALUES (?, ?, ?, ?, ?)"
                c.execute(sql_save, chat_id, user_id, message_id, 'user', prompt_text)
                c.execute(sql_save, chat_id, context.bot.id, None, 'assistant', final_reply)
                db_conn_for_saving.commit()
        except pyodbc.Error as ex:
            logger.error(f"Ошибка БД при сохранении истории: {ex}")
            if db_conn_for_saving: db_conn_for_saving.rollback()
        finally:
            if db_conn_for_saving: db_conn_for_saving.close()
            
    return final_reply
# --- Обработчики команд и сообщений ---

async def log_message_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Логирует каждое сообщение в ChatLog и обновляет профиль пользователя
    if not update.message or not update.message.text: return
    chat_id, user, message = update.effective_chat.id, update.effective_user, update.message
    if ALLOWED_GROUP_IDS and chat_id not in ALLOWED_GROUP_IDS: return

    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as c: 
                # SQL MERGE - это мощная команда "UPSERT" (UPDATE or INSERT).
                # Она проверяет, есть ли юзер с таким UserID. Если есть - обновляет его данные.
                # Если нет - вставляет новую запись. Это избавляет от лишних проверок в коде.
                upsert_user_sql = """
                    MERGE UserProfiles AS t
                    USING (VALUES(?,?,?,?,GETDATE())) AS s(UserID,Username,FirstName,LastName,LastSeen)
                    ON t.UserID = s.UserID
                    WHEN MATCHED THEN
                        UPDATE SET Username=s.Username, FirstName=s.FirstName, LastName=s.LastName, LastSeen=s.LastSeen, MessageCount=ISNULL(t.MessageCount,0)+1
                    WHEN NOT MATCHED THEN
                        INSERT(UserID,Username,FirstName,LastName,LastSeen,MessageCount,ProfileLastUpdated)
                        VALUES(s.UserID,s.Username,s.FirstName,s.LastName,s.LastSeen,1,GETDATE())
                    OUTPUT $action AS Action;
                """
                c.execute(upsert_user_sql, user.id, user.username, user.first_name, user.last_name)
                
                sql_chatlog = "INSERT INTO ChatLog (ChatID, UserID, Username, FirstName, MessageID, MessageText) VALUES (?, ?, ?, ?, ?, ?)"
                c.execute(sql_chatlog, chat_id, user.id, user.username, user.first_name, message.message_id, message.text)
                conn.commit()
            
            # Запускаем анализ профиля в фоновом режиме, чтобы не задерживать основной поток
            if message.text and len(message.text.strip()) > 10 :
                asyncio.create_task(analyze_and_update_user_profile(user_id=user.id, message_text=message.text))
        except pyodbc.Error as ex: 
            logger.error(f"Ошибка БД при логировании сообщения: {ex}")
            if conn: conn.rollback() 
        finally: 
            if conn: conn.close()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Главный обработчик всех текстовых сообщений.
    if not update.message or not update.message.text:
        return

    # Шаг 1: Логируем все сообщения в БД.
    # Эта функция также запускает фоновое обновление профиля пользователя.
    await log_message_to_db(update, context)

    chat_id = update.effective_chat.id
    user_msg_txt = update.message.text

    # Проверяем, разрешена ли работа в этой группе
    if ALLOWED_GROUP_IDS and chat_id not in ALLOWED_GROUP_IDS:
        return

    # <<< НАЧАЛО БЛОКА: Анализ сообщения, на которое ответили >>>
    # Проверяем, является ли это сообщение ответом на другое сообщение с текстом
    if update.message.reply_to_message and update.message.reply_to_message.text:
        msg_lower = user_msg_txt.lower()
        bot_user_lower = context.bot.username.lower()
        
        # Проверяем, упомянули ли бота в тексте ответа, используя тот же список никнеймов
        is_bot_mentioned = (f"@{bot_user_lower}" in msg_lower or
                            any(re.search(r'\b' + re.escape(n) + r'\b', msg_lower, re.I) for n in BOT_NICKNAMES_TO_CHECK))

        # Условие срабатывает, если упомянули бота И отвечают НЕ на его собственное сообщение
        if is_bot_mentioned and update.message.reply_to_message.from_user.id != context.bot.id:
            logger.info("Сработал триггер анализа сообщения по ответу.")
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            original_text = update.message.reply_to_message.text
            user_question = user_msg_txt  # Вопрос — это сам текст ответа

            # <<< НАЧАЛО ИЗМЕНЕНИЙ: ОБРАБОТКА "ПУСТОГО" ВОПРОСА >>>
            # Проверяем, содержит ли вопрос что-то кроме имени бота
            question_check = user_msg_txt.lower()
            # Удаляем все известные никнеймы бота из текста вопроса
            for nickname in BOT_NICKNAMES_TO_CHECK:
                question_check = question_check.replace(nickname, '').strip()
            # Дополнительно удаляем прямое упоминание @
            question_check = question_check.replace(f"@{bot_user_lower}", '').strip()
            
            # Если после очистки почти ничего не осталось, задаем вопрос по умолчанию
            if len(question_check) < 5: # Используем небольшое число, чтобы отсечь знаки препинания и короткий мусор
                logger.info(f"Вопрос в ответе почти пуст ('{user_msg_txt}'). Используется вопрос по умолчанию.")
                user_question = "Проанализируй этот текст, выдели главную мысль и выскажи свое мнение."
            # <<< КОНЕЦ ИЗМЕНЕНИЙ >>>

            # Вызываем нашу новую функцию-анализатор с (возможно) новым вопросом
            analysis_result = await analyze_replied_message(original_text, user_question)
            
            # Отвечаем на сообщение пользователя (которое само является ответом)
            await update.message.reply_text(analysis_result)
            return  # ВАЖНО: Завершаем дальнейшую обработку
    # <<< КОНЕЦ БЛОКА >>>

    # --- Стандартная логика определения триггера для обычных сообщений ---
    is_group = update.message.chat.type in ['group', 'supergroup']
    msg_lower = user_msg_txt.lower()
    bot_user_lower = context.bot.username.lower()
    triggered = (f"@{bot_user_lower}" in msg_lower or
                 any(re.search(r'\b' + re.escape(n) + r'\b', msg_lower, re.I) for n in BOT_NICKNAMES_TO_CHECK) or
                 (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id))

    # В группе реагируем только на прямое обращение
    if is_group and not triggered:
        return

    # --- Маршрутизация по специальным функциям (погода, новости) ---
    if re.search(r'\b(погод|прогноз|weather|forecast)\b', msg_lower, re.I):
        logger.info(f"Сработал триггер погоды. Запускаю анализ запроса: '{user_msg_txt}'")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        intent_data = await analyze_weather_request_with_llm(user_msg_txt)
        forecast_type = intent_data.get("type", "current")
        days_to_forecast = intent_data.get("days", 7)
        city = await extract_location_with_llm(user_msg_txt)

        if city:
            weather_report = await get_weather_data(
                location=city,
                forecast_type=forecast_type,
                days=days_to_forecast
            )
            await update.message.reply_text(weather_report)
        else:
            await update.message.reply_text("Я готов показать погоду, но не смог понять, для какого города. Пожалуйста, уточните.")
        return

    if re.search(r'\b(новост|news|события|заголовки|headlines|что нового)\b', msg_lower, re.I):
        logger.info(f"Сработал триггер новостей. Запускаю извлечение темы из текста: '{user_msg_txt}'")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        topic = await extract_news_topic_with_llm(user_msg_txt)
        news_report = await get_latest_news(query=topic)
        await update.message.reply_text(news_report, disable_web_page_preview=True)
        return

    # --- Вызов основного "мозга" для всех остальных случаев ---
    logger.info(f"Обработка сообщения через LLM в чате {chat_id}")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    reply = await get_orchestrated_llm_response(
        prompt_text=user_msg_txt,
        chat_id=chat_id,
        user_id=update.effective_user.id,
        message_id=update.message.message_id,
        context=context
    )
    if reply:
        await update.message.reply_text(reply)

# --- Остальные обработчики и главная функция ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None: 
    logger.error(f'Update "{update}" caused error "{context.error}"', exc_info=context.error)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(rf"Привет, {user.mention_html()}! Я EnkiBot, создан Yael Demedetskaya. Чем могу помочь?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = ("Я EnkiBot, AI-ассистент от Yael Demedetskaya.\n"
           "В группах я отвечаю, когда вы упоминаете меня по имени (@EnkiBot, Энки) или отвечаете на мои сообщения.\n" 
           "Вы можете спросить меня 'расскажи о [имя/тема]', чтобы я поискал информацию в истории чата.\n"
           "Чтобы узнать погоду, спросите 'какая погода в [город]?'\n\n"
           "**Команды:**\n/start - Начало работы\n/help - Эта справка\n/news - Последние новости")
    await update.message.reply_text(msg)

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    report = await get_latest_news() # <<< ИЗМЕНЕНИЕ ЗДЕСЬ
    await update.message.reply_text(report, disable_web_page_preview=True)

# <<< ОБНОВЛЕННАЯ ВЕРСИЯ >>>
# <<< ОБНОВЛЕННАЯ ВЕРСИЯ, СПОСОБНАЯ ИСКАТЬ ПО ТЕМЕ >>>
async def get_latest_news(query: str | None = None, country="us", category="general", num=5) -> str:
    """
    Асинхронно получает новости.
    - Если 'query' указан, ищет по всему миру по этому ключевому слову (на всех языках).
    - Если 'query' не указан, возвращает главные новости для указанной страны.
    """
    if not NEWS_API_KEY:
        return "Ключ News API отсутствует."

    params = {"apiKey": NEWS_API_KEY, "pageSize": num}
    base_url = "https://newsapi.org/v2/"

    if query:
        logger.info(f"Выполняется поиск новостей по запросу: '{query}'")
        # Используем endpoint для поиска по ключевому слову
        endpoint = "everything"
        # <<< ИЗМЕНЕНИЕ ЗДЕСЬ: параметр 'language' полностью убран >>>
        params.update({"q": query, "sortBy": "publishedAt"})
    else:
        logger.info(f"Запрашиваются главные новости для страны '{country}'")
        # Используем endpoint для главных заголовков
        endpoint = "top-headlines"
        params.update({"country": country, "category": category})

    url = base_url + endpoint
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        articles = response.json().get("articles", [])
        if not articles:
            return f"Новости по запросу '{query}' не найдены." if query else "Новости не найдены."

        # Форматирование ответа
        title = f"Новости по запросу '{query}':" if query else "Последние новости:"
        headlines = [title] + [f"- {a.get('title','N/A')} ({a.get('source',{}).get('name','N/A')})\n  Читать: {a.get('url','#')}" for a in articles]
        return "\n\n".join(headlines)
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Ошибка HTTP при запросе новостей: {e}")
        return f"Не удалось получить новости. Сервис вернул ошибку {e.response.status_code}."
    except Exception as e:
        logger.error(f"Неожиданная ошибка NewsAPI: {e}", exc_info=True)
        return "Произошла непредвиденная ошибка при получении новостей."
async def backfill_existing_user_name_variations():
    """
    Разовый скрипт для заполнения вариантов имен для всех пользователей,
    которые уже существуют в базе данных.
    """
    logger.info("Запуск скрипта миграции для заполнения вариантов имен существующих пользователей...")
    conn = get_db_connection()
    if not conn:
        logger.error("Миграция невозможна: нет подключения к БД.")
        return

    users_to_migrate = []
    try:
        with conn.cursor() as cursor:
            # Получаем всех пользователей из профилей
            sql = "SELECT UserID, FirstName, Username FROM UserProfiles"
            cursor.execute(sql)
            users_to_migrate = cursor.fetchall()
    except pyodbc.Error as e:
        logger.error(f"Ошибка при получении списка пользователей для миграции: {e}")
        conn.close()
        return

    logger.info(f"Найдено {len(users_to_migrate)} существующих пользователей для обработки.")

    # Для каждого пользователя запускаем уже существующую функцию генерации имен
    for user in users_to_migrate:
        logger.info(f"Обработка пользователя ID: {user.UserID}, Имя: {user.FirstName}")
        try:
            # Мы можем повторно использовать нашу функцию!
            await populate_name_variations_with_llm(user.UserID, user.FirstName, user.Username)
            # Добавим небольшую задержку, чтобы не перегружать API
            await asyncio.sleep(1) 
        except Exception as e:
            logger.error(f"Ошибка при миграции пользователя {user.UserID}: {e}")

    conn.close()
    logger.info("Миграция имен пользователей завершена.")
def main() -> None:
    """Главная функция, которая запускает бота."""
    initialize_database()
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("Токен бота отсутствует. Запуск невозможен.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("news", news_command)) 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler) 

    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
# EnkiBot: Advanced Multilingual Telegram AI Assistant

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
**Author:** Yael Demedetskaya
**Contact:** yaelkroy@gmail.com
**Project Status:** Actively Developed (as of May 30, 2025)

EnkiBot is an intelligent and adaptable Telegram assistant designed for rich, context-aware interactions. It leverages multiple Large Language Models (LLMs), maintains long-term memory, and features dynamic, evolving capabilities, including on-the-fly language pack generation.

## Key Features

* **Multilingual Support**:
    * Automatically detects message language using a combination of the current message and recent chat history for improved accuracy.
    * Responds in the detected language.
    * Can attempt to generate new language packs via LLM translation if an unsupported language is detected with high confidence.
    * Prioritized fallback: Detected Language -> English (`en`) -> Russian (`ru`) -> First available loaded pack.
* **Multi-LLM Orchestration**: 
    * Configurable to query multiple LLM providers (e.g., OpenAI, Groq, OpenRouter, Google AI).
    * Features a "race" mechanism to use the first successful/fastest response, enhancing reliability and speed.
* **Long-Term Memory & User Profiling**:
    * Stores conversation history in an MS SQL Server database for persistent memory.
    * Automatically builds and updates user profiles based on their messages, analyzing communication style and interests using LLMs.
    * Can recall information about users and past conversations based on natural language queries.
    * Generates linguistic variations of user names for better recognition in text.
* **Advanced Intent Recognition**: 
    * Uses an LLM-based master intent classifier to understand the primary goal of user requests (e.g., weather, news, information about a person, general chat, message analysis).
    * Specialized LLM prompts for extracting entities like locations for weather or topics for news.
* **Contextual Understanding**:
    * Can analyze messages replied to in group chats when explicitly asked.
    * Utilizes recent chat messages to improve language detection accuracy for short or ambiguous inputs.
* **Built-in Functions**:
    * Current weather and multi-day forecasts via OpenWeatherMap, with localized descriptions.
    * Latest news (general or topic-specific) via NewsAPI, with support for language/country biasing.
* **Modular & Extensible Architecture**:
    * Designed with a clean, modular structure (application core, language service, Telegram handlers, individual functional modules) for better maintainability, testability, and scalability.
* **Natural Language Trigger**:
    * Start a message with "Hey, Enki!" to ask questions or say "Hey, Enki! Draw ..." to generate images on demand.
* **Configurable**: Group access restrictions, API keys, and LLM model IDs are managed via environment variables for security and flexibility.
* **Robust Error Handling**: Includes database logging for critical errors and user-friendly error messages.
* **Moderation Logging**: Spam and disallowed content detections are logged to the MS SQL database for auditing and future tuning.
* **(Planned) Darwinian Self-Improvement**: The project includes a foundational structure and conceptual plan for future integration of self-rewriting code and evolutionary capabilities, inspired by concepts like the Darwin Gödel Machine, aiming for autonomous advancement of the bot's Python modules and LLM prompts.
* **Two-Tier Local Model Support**:
    * Optional router for fully local inference using `llama.cpp` compatible models.
    * Tier A: fast 7–8B models (e.g., Mistral‑7B or Llama‑3‑8B).
    * Tier B: deep 70B/72B models (e.g., Llama‑3‑70B or Qwen‑2‑72B).
    * The router escalates from Tier A to Tier B on `/deep` commands or complex prompts.
    * Includes a basic web search tool and FAISS powered RAG module for citations and offline notes.

## Technology Stack

* **Python**: 3.10+
* **Telegram Bot Framework**: `python-telegram-bot`
* **LLM APIs**: OpenAI, Groq, OpenRouter, Google AI (configurable)
* **Database**: Microsoft SQL Server (via `pyodbc`)
* **HTTP Requests**: `httpx` (asynchronous)
* **Language Detection**: `langdetect`
* **Russian Morphological Analysis**: `pymorphy3` (for specific fact extraction)
* **Transliteration**: `transliterate`
* **Configuration**: `python-dotenv` (recommended for `.env` files)

## Project Structure

The project is organized into logical modules:

* `enkibot/app.py`: Main application class for service initialization and dependency injection.
* `enkibot/core/language_service.py`: Manages language detection, loading/creation of language packs, and localized string retrieval.
* `enkibot/core/telegram_handlers.py`: Contains all Telegram-specific update handlers and interaction logic.
* `enkibot/core/llm_services.py`: Centralizes interactions with various LLM providers.
* `enkibot/modules/`: Specialized modules for:
    * `intent_recognizer.py`: LLM-based intent classification and entity extraction.
    * `profile_manager.py`: User profiling and name variation generation.
    * `api_router.py`: Interfacing with external APIs (Weather, News).
    * `response_generator.py`: Composing final bot responses, including context aggregation.
    * `fact_extractor.py`: Rule-based/linguistic fact extraction.
* `enkibot/utils/`: Utility functions for `database.py` and `logging_config.py`.
* `enkibot/lang/`: JSON-based language packs (e.g., `en.json`, `ru.json`).
* `enkibot/evolution/`: Placeholders for future self-improvement features.
* `config.py`: Centralized configuration loading from environment variables.
* `main.py`: Main entry point to initialize and run the bot.

## Getting Started

### Prerequisites

* Python 3.10 or higher.
* Microsoft SQL Server instance.
* [Microsoft ODBC Driver for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) installed.
* Git installed.
* API keys for: Telegram, OpenAI (and/or others), NewsAPI, OpenWeatherMap.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_GITHUB_REPO_URL>
    cd EnkiBot 
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv enkibot_env
    # Windows: enkibot_env\Scripts\activate
    # macOS/Linux: source enkibot_env/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the project root:
    ```txt
    python-telegram-bot>=20.0
    openai>=1.0
    pyodbc
    httpx>=0.20
    langdetect
    pymorphy3
    transliterate
    python-dotenv
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

    If you plan to run the local two-tier models you also need:
    ```txt
    llama-cpp-python
    duckduckgo-search
    trafilatura
    faiss-cpu
    sentence-transformers
    requests
    ```

4.  **Database Setup:**
    * Ensure your MS SQL Server is running and accessible.
    * The bot will attempt to create necessary tables/indexes on first run (see `enkibot/utils/database.py:initialize_database()`). The database specified in `ENKI_BOT_SQL_DATABASE_NAME` must exist, and the connection must have permissions to create tables.

5.  **Configuration (Environment Variables):**
    Create a `.env` file in the project root (`EnkiBot/`) or set environment variables directly.

    **Example `.env` file:**
    ```env
    # Telegram
    ENKI_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"

    # SQL Server (ensure your SQL Server allows TCP/IP connections and check firewall)
    ENKI_BOT_SQL_SERVER_NAME="YOUR_SERVER_NAME\YOUR_INSTANCE_NAME" # e.g., localhost\SQLEXPRESS or your_server.database.windows.net
    ENKI_BOT_SQL_DATABASE_NAME="EnkiBotDB" # Create this database first if not using 'Trusted_Connection=yes' with integrated security that allows DB creation

    # LLM API Keys & Models (provide at least OpenAI or one other)
    ENKI_BOT_OPENAI_API_KEY="sk-YOUR_OPENAI_KEY"
    ENKI_BOT_OPENAI_MODEL_ID="gpt-4o-mini" 

    # ENKI_BOT_GROQ_API_KEY="gsk_YOUR_GROQ_KEY"
    # ENKI_BOT_GROQ_MODEL_ID="llama3-8b-8192"

    # ENKI_BOT_OPENROUTER_API_KEY="sk-or-YOUR_OPENROUTER_KEY"
    # ENKI_BOT_OPENROUTER_MODEL_ID="mistralai/mistral-7b-instruct:free"
    
    # ENKI_BOT_GOOGLE_AI_API_KEY="YOUR_GOOGLE_AI_KEY"
    # ENKI_BOT_GOOGLE_AI_MODEL_ID="gemini-1.5-flash-latest"

    # External Services
    ENKI_BOT_NEWS_API_KEY="YOUR_NEWSAPI_KEY"
    ENKI_BOT_WEATHER_API_KEY="YOUR_OPENWEATHERMAP_KEY"

    # Optional: Restrict bot to specific Telegram Group IDs (comma-separated negative integers)
    # Example: ENKI_BOT_ALLOWED_GROUP_IDS="-100123456789,-100987654321" 
    ```
    **Note:** If using a `.env` file, ensure `python-dotenv` is installed and add `from dotenv import load_dotenv; load_dotenv()` at the beginning of `enkibot/config.py`.

### Running the Bot

1.  Activate your virtual environment.
2.  Navigate to the project's root directory (where `enkibot/` is a subdirectory).
3.  Run:
    ```bash
    python -m enkibot.main
    ```
    The bot will start, and logs will be written to `bot_activity.log` and the console.

### Using the Local Two-Tier Models

1. Ensure the optional dependencies for local models are installed (see above).
2. Download GGUF weights for the models you want to run, for example:
   ```bash
   wget https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/mistral-7b-instruct.Q5_K_M.gguf
   wget https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/llama-3-70b-instruct.Q4_K_M.gguf
   ```
3. Launch the minimal Telegram bot that uses the local router:
   ```bash
   TELEGRAM_BOT_TOKEN=... python -m enkibot.local_telegram_bot
   ```
   Normal messages go to the fast 7–8B model. Use `/deep` to force the 70B/72B model or `/web <query>` to run a duckduckgo search and summarise the top pages with citations.
4. (Optional) For Retrieval‑Augmented Generation, index your own documents:
   ```python
   from enkibot.modules.rag_service import RAGService
   rag = RAGService(); rag.add_documents(["My notes...", "Another doc..."])
   rag.query("question about my notes")
   ```

## Testing

To run the project's test suite, execute:

```bash
pytest
```

## How It Works

1.  **Initialization (`main.py` & `app.py`):**
    * Sets up logging and database schema.
    * `EnkiBotApplication` initializes all core services (Database, LLMs, Language, Intent, Profile, API, Response) and injects dependencies.
    * `TelegramHandlerService` is initialized with these services and registers all Telegram command and message handlers.
2.  **Message Handling (`telegram_handlers.py`):**
    * When a message is received, `LanguageService` determines the language context using the current message and recent chat history.
    * Messages are logged, and user profiling tasks are queued.
    * The `_is_triggered` method checks if the bot should respond (private chat, or in an allowed group if mentioned/replied to).
    * If triggered, `IntentRecognizer` classifies the master intent (Weather, News, User Info, General Chat, etc.) using an LLM.
    * Based on the intent, the request is routed to a specific handler method (e.g., `_handle_weather_intent`).
    * These handlers use services like `ApiRouter` (for external APIs) and `IntentRecognizer` (for entity extraction).
    * `ResponseGenerator` is used for complex replies, aggregating context from history and user profiles.
    * All user-facing text is retrieved via `LanguageService` for localization.
3.  **Language Packs (`lang/` directory & `language_service.py`):**
    * Contain JSON files (`en.json`, `ru.json`, etc.) with structured prompts for LLMs and user-facing response strings.
    * If an unsupported language is detected with high confidence, `LanguageService` attempts to translate the English pack using an LLM and saves it as a new JSON file.
    * A fallback mechanism (e.g., to English, then Russian) is in place if a specific language pack is unavailable.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to report bugs, suggest features, and submit pull requests.

This project is also looking towards implementing self-improvement mechanisms in the future. Ideas and contributions in this area are particularly encouraged.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for full details.

Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>

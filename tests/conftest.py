import sys
import types

# Stub optional dependencies to simplify test environment
sys.modules.setdefault("pyodbc", types.SimpleNamespace(Connection=object))
sys.modules.setdefault("httpx", types.SimpleNamespace())
filters_stub = types.SimpleNamespace(TEXT=1, COMMAND=2, CAPTION=4)
sys.modules.setdefault(
    "telegram",
    types.SimpleNamespace(
        InlineKeyboardButton=object,
        InlineKeyboardMarkup=object,
        Update=object,
        Message=object,
        ReplyKeyboardRemove=object,
        ChatPermissions=object,
        BotCommand=object,
        BotCommandScopeDefault=object,
        BotCommandScopeChat=object,
        BotCommandScopeChatAdministrators=object,
    ),
)
sys.modules.setdefault("telegram.constants", types.SimpleNamespace(ChatAction=object))
sys.modules.setdefault(
    "telegram.helpers", types.SimpleNamespace(mention_html=lambda *args, **kwargs: "")
)
sys.modules.setdefault(
    "telegram.ext",
    types.SimpleNamespace(
        Application=object,
        CallbackQueryHandler=object,
        CommandHandler=object,
        ContextTypes=types.SimpleNamespace(DEFAULT_TYPE=object),
        MessageHandler=object,
        filters=filters_stub,
        ConversationHandler=object,
    ),
)
sys.modules.setdefault("tldextract", types.SimpleNamespace())

def _fake_detect(text: str) -> str:
    for ch in text:
        if ch.lower() in "їєіґ":
            return "uk"
    if any("а" <= ch.lower() <= "я" for ch in text):
        return "ru"
    return "en"

sys.modules.setdefault(
    "langdetect",
    types.SimpleNamespace(
        detect=_fake_detect,
        DetectorFactory=types.SimpleNamespace(seed=0),
        LangDetectException=Exception,
    ),
)


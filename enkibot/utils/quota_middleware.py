import logging
from enkibot import config
from .database import DatabaseManager

logger = logging.getLogger(__name__)

async def enforce_user_quota(db_manager: DatabaseManager, user_id: int, usage_type: str) -> bool:
    """Check and increment usage for a user. Returns True if under quota."""
    usage_type = usage_type.lower()
    if usage_type not in {"llm", "image"}:
        logger.error("Invalid usage_type '%s' for quota check", usage_type)
        return False
    limit = config.DAILY_LLM_QUOTA if usage_type == "llm" else config.DAILY_IMAGE_QUOTA
    if limit <= 0:
        return True
    allowed = await db_manager.check_and_increment_usage(user_id, usage_type, limit)
    if not allowed:
        logger.info("User %s exceeded %s quota", user_id, usage_type)
    return allowed

# enkibot/modules/admin_tools.py
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
"""Administrative moderation helpers for Telegram.

This module provides a *best in class* reference implementation of
administrative tools for Telegram chats.  It combines progressive
discipline (warn → mute → temp-ban → permaban), extensive auditing and
reversible actions.  The implementation is intentionally self contained
so it can be plugged into :class:`~telegram.ext.Application` directly or
integrated into the :class:`~enkibot.app.EnkiBotApplication` pipeline.

Only a subset of the full system described in the project specification
is implemented.  The code is structured so additional features like
crowd‑moderation, trust scores and external policy engines can be added
without major refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    String,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ChatPermissions,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

Base = declarative_base()

class ModerationAction(Base):
    __tablename__ = "moderation_actions"
    action_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, index=True)
    target_user_id = Column(BigInteger, index=True)
    action = Column(String(32))
    reason = Column(String(64))
    params_json = Column(JSON, default=dict)
    until_ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Warning(Base):
    __tablename__ = "warnings"
    warn_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, index=True)
    user_id = Column(BigInteger, index=True)
    reason = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)

class ModeratorNote(Base):
    __tablename__ = "moderator_notes"
    note_id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, index=True)
    user_id = Column(BigInteger, index=True)
    note_text = Column(String(1024))
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_duration(duration: str) -> Optional[int]:
    """Parse a short duration spec like ``10m`` or ``1h``.

    Returns the number of seconds or ``None`` for invalid input.
    """
    if not duration:
        return None
    try:
        value = int(duration[:-1])
    except (ValueError, TypeError):
        return None
    unit = duration[-1].lower()
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return None


PERM_READ_ONLY = ChatPermissions(can_send_messages=False)
FULL_PERMS = ChatPermissions(
    can_send_messages=True,
    can_send_media_messages=True,
    can_send_polls=True,
    can_add_web_page_previews=True,
)

# ---------------------------------------------------------------------------
# Administrative service
# ---------------------------------------------------------------------------

@dataclass
class AdminTools:
    """Container for moderation logic.

    Parameters
    ----------
    app:
        PTB :class:`~telegram.ext.Application` to which handlers will be
        attached.
    engine_url:
        SQLAlchemy connection string.  A tiny SQLite database is used by
        default which is sufficient for tests and local runs.
    """

    app: Application
    engine_url: str = "sqlite:///admin_tools.sqlite3"

    def __post_init__(self) -> None:
        self.engine = create_engine(self.engine_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------
    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("ban", self.cmd_ban))
        self.app.add_handler(CommandHandler("unban", self.cmd_unban))
        self.app.add_handler(CommandHandler("mute", self.cmd_mute))
        self.app.add_handler(CommandHandler("unmute", self.cmd_unmute))
        self.app.add_handler(CommandHandler("warn", self.cmd_warn))
        self.app.add_handler(CommandHandler("warns_list", self.cmd_warns))
        self.app.add_handler(CommandHandler("rm_warn", self.cmd_rm_warn))
        self.app.add_handler(CommandHandler("note", self.cmd_note))
        self.app.add_handler(CommandHandler("shadowdel", self.cmd_shadowdel))
        self.app.add_handler(CallbackQueryHandler(self.on_confirm, pattern=r"^mod:"))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    async def _is_admin(self, update: Update) -> bool:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        member = await self.app.bot.get_chat_member(chat_id, user_id)
        return member.status in {"creator", "administrator"}

    async def _ensure_admin(self, update: Update) -> bool:
        if await self._is_admin(update):
            return True
        await update.effective_message.reply_text("Admins only.")
        return False

    def _log_action(self, chat_id: int, target_id: int, action: str, reason: str | None = None, until: datetime | None = None) -> None:
        with self.Session() as s:
            s.add(ModerationAction(chat_id=chat_id, target_user_id=target_id, action=action, reason=reason or "", until_ts=until))
            s.commit()

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------
    async def cmd_ban(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to a user with /ban [reason].")
            return
        reason = " ".join(context.args) if context.args else ""
        target = update.message.reply_to_message.from_user
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("Confirm ban", callback_data=f"mod:ban:{target.id}:{reason}")]])
        await update.effective_message.reply_text("Confirm permanent ban?", reply_markup=kb)

    async def cmd_unban(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        target_id = None
        if update.message.reply_to_message:
            target_id = update.message.reply_to_message.from_user.id
        elif context.args:
            try:
                target_id = int(context.args[0])
            except ValueError:
                pass
        if not target_id:
            await update.effective_message.reply_text("Reply or pass user id to /unban")
            return
        await context.bot.unban_chat_member(update.effective_chat.id, target_id)
        self._log_action(update.effective_chat.id, target_id, "unban")
        await update.effective_message.reply_text("User unbanned.")

    async def cmd_mute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message or not context.args:
            await update.effective_message.reply_text("Usage: /mute <10m|1h> [reason]")
            return
        seconds = parse_duration(context.args[0])
        if not seconds:
            await update.effective_message.reply_text("Invalid duration.")
            return
        reason = " ".join(context.args[1:]) if len(context.args) > 1 else ""
        target = update.message.reply_to_message.from_user
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("Confirm mute", callback_data=f"mod:mute:{target.id}:{seconds}:{reason}")]])
        await update.effective_message.reply_text("Confirm mute?", reply_markup=kb)

    async def cmd_unmute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to a user with /unmute.")
            return
        target = update.message.reply_to_message.from_user
        await context.bot.restrict_chat_member(update.effective_chat.id, target.id, permissions=FULL_PERMS)
        self._log_action(update.effective_chat.id, target.id, "unmute")
        await update.effective_message.reply_text("User unmuted.")

    async def cmd_warn(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to a user with /warn [reason].")
            return
        target = update.message.reply_to_message.from_user
        reason = " ".join(context.args) if context.args else ""
        with self.Session() as s:
            s.add(Warning(chat_id=update.effective_chat.id, user_id=target.id, reason=reason))
            s.commit()
            count = (
                s.query(Warning)
                .filter(Warning.chat_id == update.effective_chat.id, Warning.user_id == target.id)
                .count()
            )
        await update.effective_message.reply_html(f"⚠️ {target.mention_html()} warned (#{count}). Reason: {reason or '—'}")

    async def cmd_warns(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        target = None
        if update.message.reply_to_message:
            target = update.message.reply_to_message.from_user
        elif context.args:
            try:
                target = await context.bot.get_chat_member(update.effective_chat.id, int(context.args[0]))
                target = target.user
            except Exception:
                target = None
        if not target:
            await update.effective_message.reply_text("Usage: /warns_list (reply or user_id)")
            return
        with self.Session() as s:
            warns = (
                s.query(Warning)
                .filter(Warning.chat_id == update.effective_chat.id, Warning.user_id == target.id)
                .order_by(Warning.created_at.desc())
                .all()
            )
        if not warns:
            await update.effective_message.reply_text("No warnings.")
            return
        lines = [f"{w.created_at:%Y-%m-%d} • {w.reason or '-'}" for w in warns[:20]]
        await update.effective_message.reply_text("\n".join(lines))

    async def cmd_rm_warn(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        target = None
        if update.message.reply_to_message:
            target = update.message.reply_to_message.from_user
        elif context.args:
            try:
                target = await context.bot.get_chat_member(update.effective_chat.id, int(context.args[0]))
                target = target.user
            except Exception:
                target = None
        if not target:
            await update.effective_message.reply_text("Usage: /rm_warn (reply or user_id) [count|all]")
            return
        count_arg = context.args[1] if len(context.args) > 1 else "1"
        with self.Session() as s:
            q = s.query(Warning).filter(Warning.chat_id == update.effective_chat.id, Warning.user_id == target.id)
            if count_arg == "all":
                removed = q.count()
                q.delete()
            else:
                try:
                    n = int(count_arg)
                except ValueError:
                    n = 1
                warns = q.order_by(Warning.created_at.desc()).limit(n).all()
                removed = len(warns)
                for w in warns:
                    s.delete(w)
            s.commit()
        await update.effective_message.reply_text(f"Removed {removed} warning(s).")

    async def cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message or not context.args:
            await update.effective_message.reply_text("Reply to a user with /note <text>.")
            return
        target = update.message.reply_to_message.from_user
        text = " ".join(context.args)
        with self.Session() as s:
            s.add(ModeratorNote(chat_id=update.effective_chat.id, user_id=target.id, note_text=text))
            s.commit()
        await update.effective_message.reply_text("Note stored (private).")

    async def cmd_shadowdel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_admin(update):
            return
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to a message with /shadowdel.")
            return
        try:
            await update.message.reply_to_message.delete()
        except Exception:
            pass
        self._log_action(update.effective_chat.id, update.message.reply_to_message.from_user.id, "delete")

    # ------------------------------------------------------------------
    # Confirmation callbacks
    # ------------------------------------------------------------------
    async def on_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        await q.answer()
        try:
            _, action, user_id, *rest = q.data.split(":", 3)
        except ValueError:
            await q.edit_message_text("Malformed callback.")
            return
        chat_id = update.effective_chat.id
        user_id_int = int(user_id)
        if action == "ban":
            reason = rest[0] if rest else ""
            await context.bot.ban_chat_member(chat_id, user_id_int)
            self._log_action(chat_id, user_id_int, "ban", reason)
            await q.edit_message_text("User banned.")
        elif action == "mute":
            seconds = int(rest[0]) if rest else 0
            reason = rest[1] if len(rest) > 1 else ""
            until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            await context.bot.restrict_chat_member(chat_id, user_id_int, permissions=PERM_READ_ONLY, until_date=int(until.timestamp()))
            self._log_action(chat_id, user_id_int, "mute", reason, until)
            await q.edit_message_text("User muted.")
        else:
            await q.edit_message_text("Unknown action.")

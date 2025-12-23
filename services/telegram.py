# telegram.py - Telegram bot interface service (Native API).
"""
This is "The Mouth" - it handles all user interaction via Telegram.
Routes ALL messages to the AI agent for natural language processing.

Uses native Telegram Bot API with aiohttp (no framework dependency).

Update this file to:
- Change UI/UX behavior
- Modify notification formatting

Does NOT affect: ibkr.py, agent.py, tiingo.py, strategies.py
"""

import os
import asyncio
from typing import Any

import aiohttp

import context
from services.time_utils import get_et_timestamp

# === TESTING FEATURE TOGGLE ===
# Set to True to enable /test command with inline keyboard buttons
# Set to False to disable testing buttons completely
ENABLE_TESTING_BUTTONS = True

# Conditional import (only if enabled)
if ENABLE_TESTING_BUTTONS:
    from strategies import component_test


class TelegramBot:
    """
    Telegram bot using native Bot API with aiohttp.

    Handles:
    - All messages routed to AI agent
    - Notifications from log queue
    """

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self, agent: Any = None):
        """
        Initialize the Telegram bot.

        Args:
            agent: GeminiAgent instance for processing messages
        """
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

        # Load whitelist (comma-separated user IDs, or empty to allow all)
        whitelist_str = os.getenv("TELEGRAM_WHITELIST", "")
        if whitelist_str.strip():
            self._whitelist = set(int(uid.strip()) for uid in whitelist_str.split(",") if uid.strip())
            print(f"📋 Telegram whitelist enabled: {len(self._whitelist)} user(s)")
        else:
            self._whitelist = None
            print("⚠️  Telegram whitelist DISABLED - all users allowed")

        self.agent = agent
        self._admin_chat_id: int | None = None
        self._session: aiohttp.ClientSession | None = None
        self._offset: int = 0
        self._running: bool = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user ID is in whitelist (or whitelist is disabled)."""
        if self._whitelist is None:
            return True  # Whitelist disabled, allow all
        return user_id in self._whitelist

    async def _api_call(self, method: str, data: dict | None = None) -> dict:
        """Make a Telegram Bot API call."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{self.token}/{method}"

        async with session.post(url, json=data or {}) as response:
            result = await response.json()
            if not result.get("ok"):
                print(f"Telegram API error: {result}")
            return result

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict | None = None
    ) -> dict:
        """Send a message to a chat."""
        data = {"chat_id": chat_id, "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        if reply_markup:
            data["reply_markup"] = reply_markup
        return await self._api_call("sendMessage", data)

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str | None = None
    ) -> dict:
        """Answer a callback query (from inline keyboard button)."""
        data = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text
        return await self._api_call("answerCallbackQuery", data)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> dict:
        """Send a chat action (typing indicator)."""
        return await self._api_call("sendChatAction", {
            "chat_id": chat_id,
            "action": action
        })

    async def get_updates(self, timeout: int = 30) -> list[dict]:
        """Get updates using long polling."""
        allowed = ["message"]
        if ENABLE_TESTING_BUTTONS:
            allowed.append("callback_query")

        result = await self._api_call("getUpdates", {
            "offset": self._offset,
            "timeout": timeout,
            "allowed_updates": allowed
        })

        updates = result.get("result", [])

        if updates:
            self._offset = updates[-1]["update_id"] + 1

        return updates

    async def _handle_update(self, update: dict) -> None:
        """Route update to AI agent or handle testing callbacks."""

        # Handle callback queries (inline keyboard button presses)
        if ENABLE_TESTING_BUTTONS and "callback_query" in update:
            await self._handle_callback_query(update["callback_query"])
            return

        # Handle regular messages
        message = update.get("message")
        if not message:
            return

        text = message.get("text", "")
        chat_id = message["chat"]["id"]

        if not text:
            return

        # Store admin chat ID for notifications
        if self._admin_chat_id is None:
            self._admin_chat_id = chat_id

        # Get user info for logging
        user = message.get("from", {})
        user_id = user.get("id")
        username = user.get("username", user.get("first_name", "Unknown"))
        print(f"\n💬 [{username}]: {text}")

        # Check whitelist
        if not self._is_user_allowed(user_id):
            print(f"🚫 User {user_id} ({username}) not in whitelist - ignoring message")
            await self.send_message(chat_id, "⛔ Unauthorized. Contact admin.")
            return

        # Check for /test command (if testing enabled)
        if ENABLE_TESTING_BUTTONS and component_test.is_test_command(text):
            message_text, reply_markup = component_test.handle_test_command()
            await self.send_message(chat_id, message_text, reply_markup=reply_markup)
            return

        # Route everything else to AI agent
        if not self.agent:
            await self.send_message(chat_id, "Agent not available")
            return

        # Show typing indicator
        await self.send_chat_action(chat_id, "typing")

        # Get response from agent
        print(f"🤔 Processing with AI agent...")
        response = await self.agent.chat(text, chat_id)

        # Log response
        response_preview = response[:100] + "..." if len(response) > 100 else response
        print(f"🤖 Bot: {response_preview}")

        # Send response
        await self.send_message(chat_id, response)

    async def _handle_callback_query(self, callback_query: dict) -> None:
        """Handle inline keyboard button presses."""
        callback_id = callback_query["id"]
        callback_data = callback_query.get("data", "")
        chat_id = callback_query["message"]["chat"]["id"]

        # Get user info for logging
        user = callback_query.get("from", {})
        user_id = user.get("id")
        username = user.get("username", user.get("first_name", "Unknown"))
        print(f"\n🖱️  [{username}]: Button pressed: {callback_data}")

        # Check whitelist
        if not self._is_user_allowed(user_id):
            print(f"🚫 User {user_id} ({username}) not in whitelist - ignoring callback")
            await self.answer_callback_query(callback_id, "⛔ Unauthorized")
            return

        # Handle test callbacks
        if component_test.is_test_callback(callback_data):
            response = await component_test.handle_callback(callback_data)

            # Answer the callback (removes loading state from button)
            await self.answer_callback_query(callback_id)

            # Send response message
            await self.send_message(chat_id, response)
        else:
            # Unknown callback
            await self.answer_callback_query(callback_id, "Unknown action")

    async def _polling_loop(self) -> None:
        """Main polling loop to receive updates."""
        print("Telegram bot polling started...")

        while self._running and not context.shutdown_event.is_set():
            try:
                updates = await self.get_updates(timeout=30)

                for update in updates:
                    try:
                        await self._handle_update(update)
                    except Exception as e:
                        print(f"Error handling update: {e}")

            except asyncio.CancelledError:
                break
            except aiohttp.ClientError as e:
                print(f"Telegram connection error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Polling error: {e}")
                await asyncio.sleep(1)

    async def _notification_loop(self) -> None:
        """Background task to send notifications from log queue."""
        while self._running and not context.shutdown_event.is_set():
            try:
                try:
                    log_msg = context.log_queue.get_nowait()
                except:
                    await asyncio.sleep(0.5)
                    continue

                if self._admin_chat_id:
                    timestamp = get_et_timestamp()
                    if log_msg.level == "trade":
                        text = f"[{timestamp}] 📊 Trade: {log_msg.message}"
                    elif log_msg.level == "error":
                        text = f"[{timestamp}] ❌ Error: {log_msg.message}"
                    else:
                        text = f"[{timestamp}] {log_msg.message}"

                    try:
                        print(f"🔔 Notification: {text[:80]}...")
                        await self.send_message(self._admin_chat_id, text)
                    except Exception as e:
                        print(f"Failed to send notification: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Notification loop error: {e}")
                await asyncio.sleep(1)

    async def start(self) -> None:
        """Start the bot polling and notification loops."""
        self._running = True

        notification_task = asyncio.create_task(self._notification_loop())

        try:
            await self._polling_loop()
        finally:
            self._running = False
            notification_task.cancel()
            try:
                await notification_task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        """Stop the bot and close session."""
        self._running = False

        if self._session and not self._session.closed:
            await self._session.close()
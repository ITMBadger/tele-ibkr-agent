# agent.py - AI agent with tool calling capabilities using Poe API.
"""
This is "The Brain" - it processes user messages and decides what actions to take.

Uses Poe's OpenAI-compatible API (https://api.poe.com/v1) with GLM-4.7.
Supports function calling / tool use for trading operations.

Account data is loaded at startup and cached in context.
Agent reads from cache for instant responses.
Only refreshes when user explicitly asks.

Update this file to:
- Add new tools
- Change LLM model
- Modify agent behavior

Does NOT affect: ibkr.py, telegram.py, tiingo.py, strategies/
"""

import os
import json
from typing import Callable, Any

from openai import OpenAI

import context
from services import guardrails
from tools import admin_tools, strategy_tools


class PoeAgent:
    """
    Poe-powered agent that can execute tools based on user commands.

    Uses Poe's OpenAI-compatible API with GLM-4.7 (with thinking).
    Reads account data from context (cached at startup).
    Only requests fresh data when user explicitly asks to refresh.
    """

    # Poe API configuration
    POE_BASE_URL = "https://api.poe.com/v1"

    @staticmethod
    def _build_system_prompt() -> str:
        """Build system prompt with dynamic guardrail information."""
        allowed_accounts = ", ".join(guardrails.ALLOWED_ACCOUNTS) if guardrails.ALLOWED_ACCOUNTS else "ALL"
        max_quantity = guardrails.MAX_ORDER_QUANTITY

        return f"""You are a helpful trading assistant. You help users:
- Check market prices and positions
- Activate/deactivate automated trading strategies
- Monitor their portfolio and strategy performance
- Manage multiple trading accounts

**STRATEGY-ONLY TRADING**
This bot uses automated strategies only - no manual buy/sell orders.
If a user asks to manually buy/sell, explain that they should:
1. Use "list strategies" to see available strategies
2. Apply a strategy to their symbol (e.g., "apply strategy 1 to QQQ")
3. Let the strategy manage entries/exits automatically

**GUARDRAILS ACTIVE**
The following safety limits are enforced on ALL strategy orders:
- Allowed Accounts: {allowed_accounts}
- Max Order Quantity: {max_quantity} shares per trade

Orders that violate these limits will be BLOCKED automatically.

**Important - When to use tools vs conversation:**
- Use tools ONLY when you need fresh data you don't already have
- If data was just shown in the conversation, USE THAT DATA - don't call tools again
- You can do math, calculations, and analysis on data already in the conversation
- For follow-up questions like "add them", "what's the total", "compare them" - use the data you already have

**Account Data:**
- Account balances are loaded at startup and cached
- Use get_account_info to show cached balances (instant response)
- Only use refresh_balances when user explicitly asks to refresh/update balances

**Multi-Account Support:**
- Users may have multiple trading accounts
- Use get_account_info to show all account balances
- Use switch_account to change active account
- Always show which account is active when relevant

**Strategy Activation:**
- User says "apply strategy 1 to QQQ" or "use strategy 3 on SPY"
- You propose the strategy and ask for confirmation
- User confirms with "yes"
- Strategy activates with ALL parameters pre-set (quantity, intervals, indicators)
- Strategies run automatically - no manual intervention needed
- To stop: "stop strategy for QQQ" or "deactivate strategy on SPY"

Be concise and clear in your responses. Do calculations yourself when data is already available.
"""

    def __init__(
        self,
        tiingo_service: Any = None,
        execution_handler: Callable | None = None
    ):
        self.api_key = os.getenv("POE_API_KEY")
        self.model_name = os.getenv("POE_MODEL", "glm-4.7")

        if not self.api_key:
            raise ValueError("POE_API_KEY not set in environment")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.POE_BASE_URL
        )
        self.tiingo = tiingo_service
        self.execution_handler = execution_handler
        self._current_chat_id: int | None = None
        self._tools = self._build_tools()
        self.system_prompt = self._build_system_prompt()

        # Log configuration on startup
        print(f"\n{'='*60}")
        print(f"AI Agent: Poe API ({self.model_name})")
        print(f"Guardrails Active:")
        print(f"   Allowed Accounts: {', '.join(guardrails.ALLOWED_ACCOUNTS) if guardrails.ALLOWED_ACCOUNTS else 'ALL'}")
        print(f"   Max Order Quantity: {guardrails.MAX_ORDER_QUANTITY} shares/trade")
        print(f"   Max Slippage Tolerance: {guardrails.SLIPPAGE_TOLERANCE}%")
        print(f"{'='*60}\n")

    def _build_tools(self) -> list[dict]:
        """Build the list of tools available to the agent (OpenAI format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_account_info",
                    "description": "Get account balances for all accounts. Reads from cached data (instant). Shows net value, cash, and available funds.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refresh_balances",
                    "description": "Refresh account balances from broker. Only use when user explicitly asks to refresh/update balances.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_account",
                    "description": "Switch to a different trading account",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_id": {"type": "string", "description": "Account ID to switch to"}
                        },
                        "required": ["account_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_positions",
                    "description": "Get all current stock positions. Reads from cached data.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refresh_positions",
                    "description": "Refresh positions from broker. Only use when user explicitly asks to refresh/update positions.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_price",
                    "description": "Get the current price of a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol"}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_strategies",
                    "description": "List all available trading strategies",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "activate_strategy",
                    "description": "Propose activating a strategy on a symbol. User says 'apply strategy 1 to QQQ'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy_id": {"type": "string", "description": "Strategy ID (1, 2, 3, etc.)"},
                            "symbol": {"type": "string", "description": "Stock symbol to trade"}
                        },
                        "required": ["strategy_id", "symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_strategy",
                    "description": "Confirm and start a pending strategy (when user says yes)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_strategy",
                    "description": "Cancel a pending strategy (when user says no)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_strategy",
                    "description": "Stop a running strategy for a symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol"}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_active_strategies",
                    "description": "List all currently running strategies",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tracked_positions",
                    "description": "Get all tracked positions from JSON storage. Shows strategy, entry price, TP/SL for each position.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_position_records",
                    "description": "Clear all tracked position records from JSON storage. Does NOT affect actual broker positions.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
        ]

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name and return the result."""
        try:
            match name:
                case "get_account_info":
                    return admin_tools.get_account_info()
                case "refresh_balances":
                    return admin_tools.refresh_balances()
                case "switch_account":
                    return admin_tools.switch_account(**args)
                case "get_positions":
                    return admin_tools.get_positions()
                case "refresh_positions":
                    return admin_tools.refresh_positions()
                case "get_price":
                    return await admin_tools.get_price(**args, tiingo_service=self.tiingo)
                case "list_strategies":
                    return strategy_tools.list_strategies()
                case "activate_strategy":
                    return strategy_tools.activate_strategy(**args, chat_id=self._current_chat_id)
                case "confirm_strategy":
                    return strategy_tools.confirm_strategy(chat_id=self._current_chat_id, tiingo_service=self.tiingo)
                case "cancel_strategy":
                    return strategy_tools.cancel_strategy(chat_id=self._current_chat_id)
                case "stop_strategy":
                    return strategy_tools.stop_strategy(**args)
                case "list_active_strategies":
                    return strategy_tools.list_active_strategies()
                case "get_tracked_positions":
                    return admin_tools.get_tracked_positions()
                case "clear_position_records":
                    return admin_tools.clear_position_records()
                case _:
                    return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    # === MAIN CHAT INTERFACE ===

    async def chat(self, user_message: str, chat_id: int | None = None) -> str:
        """Process a user message and return a response."""
        self._current_chat_id = chat_id

        try:
            chat_key = str(chat_id) if chat_id else "default"
            history = context.conversation_history.get(chat_key, [])

            # Build messages list with system prompt
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add conversation history
            for msg in history:
                messages.append(msg)

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Make API call with tools
            # GLM-4.7 specific: enable thinking mode for better reasoning
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self._tools,
                tool_choice="auto",
                extra_body={
                    "enable_thinking": True,
                    "temperature": 0.7
                }
            )

            assistant_message = response.choices[0].message

            # Check if model wants to call tools
            if assistant_message.tool_calls:
                # Execute all tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    result = await self._execute_tool(func_name, func_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result
                    })

                # Add assistant message with tool calls to messages
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Add tool results
                for result in tool_results:
                    messages.append(result)

                # Get final response after tool execution
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={
                        "enable_thinking": True,
                        "temperature": 0.7
                    }
                )

                final_text = final_response.choices[0].message.content or "I processed your request."

                # Update history (exclude system prompt)
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": final_text})

                # Keep history bounded
                if len(history) > 40:
                    history = history[-40:]

                context.conversation_history.set(chat_key, history)
                return final_text

            else:
                # No tool calls, just return the response
                response_text = assistant_message.content or "I'm not sure how to help with that."

                # Update history
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": response_text})

                if len(history) > 40:
                    history = history[-40:]

                context.conversation_history.set(chat_key, history)
                return response_text

        except Exception as e:
            return f"Error processing message: {str(e)}"


# Alias for backwards compatibility
GeminiAgent = PoeAgent

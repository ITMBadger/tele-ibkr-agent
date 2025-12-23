# agent.py - Gemini AI agent with tool calling capabilities.
"""
This is "The Brain" - it processes user messages and decides what actions to take.

Account data is loaded at startup and cached in context.
Agent reads from cache for instant responses.
Only refreshes when user explicitly asks.

Update this file to:
- Add new tools
- Change LLM provider
- Modify agent behavior

Does NOT affect: ibkr.py, telegram.py, tiingo.py, strategies/
"""

import os
from typing import Callable, Any

from google import genai
from google.genai import types

import context
from services import guardrails
from tools import admin_tools, strategy_tools


class GeminiAgent:
    """
    Gemini-powered agent that can execute tools based on user commands.

    Reads account data from context (cached at startup).
    Only requests fresh data when user explicitly asks to refresh.
    """

    @staticmethod
    def _build_system_prompt() -> str:
        """Build system prompt with dynamic guardrail information."""
        # Get guardrail values from guardrails module
        allowed_accounts = ", ".join(guardrails.ALLOWED_ACCOUNTS) if guardrails.ALLOWED_ACCOUNTS else "ALL"
        max_quantity = guardrails.MAX_ORDER_QUANTITY

        return f"""You are a helpful trading assistant. You help users:
    - Check market prices and positions
    - Activate/deactivate automated trading strategies
    - Monitor their portfolio and strategy performance
    - Manage multiple trading accounts

    ⚠️ **STRATEGY-ONLY TRADING** ⚠️
    This bot uses automated strategies only - no manual buy/sell orders.
    If a user asks to manually buy/sell, explain that they should:
    1. Use "list strategies" to see available strategies
    2. Apply a strategy to their symbol (e.g., "apply strategy 1 to QQQ")
    3. Let the strategy manage entries/exits automatically

    ⚠️ **GUARDRAILS ACTIVE** ⚠️
    The following safety limits are enforced on ALL strategy orders:
    • Allowed Accounts: {allowed_accounts}
    • Max Order Quantity: {max_quantity} shares per trade

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
    - Examples of refresh requests: "refresh balances", "update account data", "get latest balances"

    **Multi-Account Support:**
    - Users may have multiple IBKR accounts
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

    SYSTEM_PROMPT = ""  # Will be set dynamically in __init__

    def __init__(
        self,
        tiingo_service: Any = None,
        execution_handler: Callable | None = None
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        self.client = genai.Client(api_key=self.api_key)
        self.tiingo = tiingo_service
        self.execution_handler = execution_handler
        self._current_chat_id: int | None = None
        self._tools = self._build_tools()

        # Build system prompt with guardrail info
        self.SYSTEM_PROMPT = self._build_system_prompt()

        # Log guardrails on startup
        print(f"\n{'='*60}")
        print("⚠️  GUARDRAILS ACTIVE")
        print(f"   Allowed Accounts: {', '.join(guardrails.ALLOWED_ACCOUNTS) if guardrails.ALLOWED_ACCOUNTS else 'ALL'}")
        print(f"   Max Order Quantity: {guardrails.MAX_ORDER_QUANTITY} shares/trade")
        print(f"   Max Slippage Tolerance: {guardrails.SLIPPAGE_TOLERANCE}%")
        print(f"{'='*60}\n")

    def _build_tools(self) -> list[types.Tool]:
        """Build the list of tools available to the agent."""
        tool_functions = [
            # Account info (reads from cache)
            types.FunctionDeclaration(
                name="get_account_info",
                description="Get account balances for all accounts. Reads from cached data (instant). Shows net value, cash, and available funds.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            
            # Refresh balances (requests fresh data from IBKR)
            types.FunctionDeclaration(
                name="refresh_balances",
                description="Refresh account balances from IBKR. Only use when user explicitly asks to refresh/update balances.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),

            # Switch account
            types.FunctionDeclaration(
                name="switch_account",
                description="Switch to a different trading account",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "account_id": types.Schema(type=types.Type.STRING, description="Account ID to switch to"),
                    },
                    required=["account_id"]
                )
            ),

            # Positions (reads from cache)
            types.FunctionDeclaration(
                name="get_positions",
                description="Get all current stock positions. Reads from cached data.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            
            # Refresh positions
            types.FunctionDeclaration(
                name="refresh_positions",
                description="Refresh positions from IBKR. Only use when user explicitly asks to refresh/update positions.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),

            # Market data
            types.FunctionDeclaration(
                name="get_price",
                description="Get the current price of a stock",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol"),
                    },
                    required=["symbol"]
                )
            ),

            # Strategy tools
            types.FunctionDeclaration(
                name="list_strategies",
                description="List all available trading strategies",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            
            types.FunctionDeclaration(
                name="activate_strategy",
                description="Propose activating a strategy on a symbol. User says 'apply strategy 1 to QQQ'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "strategy_id": types.Schema(type=types.Type.STRING, description="Strategy ID (1, 2, 3, etc.)"),
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol to trade"),
                    },
                    required=["strategy_id", "symbol"]
                )
            ),
            
            types.FunctionDeclaration(
                name="confirm_strategy",
                description="Confirm and start a pending strategy (when user says yes)",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            
            types.FunctionDeclaration(
                name="cancel_strategy",
                description="Cancel a pending strategy (when user says no)",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),

            types.FunctionDeclaration(
                name="stop_strategy",
                description="Stop a running strategy for a symbol",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(type=types.Type.STRING, description="Stock symbol"),
                    },
                    required=["symbol"]
                )
            ),

            types.FunctionDeclaration(
                name="list_active_strategies",
                description="List all currently running strategies",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),

            # Position tracking tools
            types.FunctionDeclaration(
                name="get_tracked_positions",
                description="Get all tracked positions from JSON storage. Shows strategy, entry price, TP/SL for each position.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),

            types.FunctionDeclaration(
                name="clear_position_records",
                description="Clear all tracked position records from JSON storage. Does NOT affect actual IBKR positions.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
        ]

        return [types.Tool(function_declarations=tool_functions)]

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

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
            history.append(user_content)

            if len(history) > 20:
                history = history[-20:]

            contents = history.copy()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    tools=self._tools,
                )
            )

            if response.candidates and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts

                result_parts = []
                for part in parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_result = await self._execute_tool(fc.name, dict(fc.args))

                        history.append(types.Content(role="model", parts=parts))
                        history.append(types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name=fc.name,
                                response={"result": tool_result}
                            )]
                        ))

                        follow_up_contents = history.copy()

                        final_response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=follow_up_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=self.SYSTEM_PROMPT,
                            )
                        )

                        if final_response.text:
                            result_parts.append(final_response.text)
                            history.append(types.Content(
                                role="model",
                                parts=[types.Part.from_text(text=final_response.text)]
                            ))
                        else:
                            result_parts.append(tool_result)

                    elif hasattr(part, 'text') and part.text:
                        result_parts.append(part.text)

                final_result = "\n".join(result_parts) if result_parts else "I processed your request."

                if not any(hasattr(p, 'function_call') and p.function_call for p in parts):
                    history.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=final_result)]
                    ))

                context.conversation_history.set(chat_key, history[-20:])
                return final_result

            fallback_text = response.text if response.text else "I'm not sure how to help with that."
            history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=fallback_text)]
            ))
            context.conversation_history.set(chat_key, history[-20:])
            return fallback_text

        except Exception as e:
            return f"Error processing message: {str(e)}"
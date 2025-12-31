# guardrails.py - Centralized trading guardrails and safety checks.
"""
This module contains all hardcoded guardrails that protect against
unintended trades and enforce safety limits.

All guardrails are loaded from .env and enforced in order_service.submit_order().

To modify guardrails, update .env file:
- IBKR_ALLOWED_ACCOUNTS: Comma-separated list of allowed IBKR account IDs
- IBKR_MAX_ORDER_QUANTITY: Max shares per order for IBKR
- HYPERLIQUID_ALLOWED_ACCOUNTS: Comma-separated list of allowed Hyperliquid accounts
- HYPERLIQUID_MAX_ORDER_QUANTITY: Max contracts per order for Hyperliquid
- SLIPPAGE_TOLERANCE: Maximum % difference between trigger and market price
"""

import os

import context


# === GUARDRAIL CONFIGURATION (loaded from .env) ===

# Slippage tolerance is shared across brokers
SLIPPAGE_TOLERANCE = float(os.getenv("SLIPPAGE_TOLERANCE", "1.0"))

# Broker-specific settings
_IBKR_ACCOUNTS_STR = os.getenv("IBKR_ALLOWED_ACCOUNTS", "")
_IBKR_ACCOUNTS = [acc.strip() for acc in _IBKR_ACCOUNTS_STR.split(",") if acc.strip()]
_IBKR_MAX_QTY = float(os.getenv("IBKR_MAX_ORDER_QUANTITY", "100"))

_HL_ACCOUNTS_STR = os.getenv("HYPERLIQUID_ALLOWED_ACCOUNTS", "")
_HL_ACCOUNTS = [acc.strip() for acc in _HL_ACCOUNTS_STR.split(",") if acc.strip()]
_HL_MAX_QTY = float(os.getenv("HYPERLIQUID_MAX_ORDER_QUANTITY", "2.0"))


def get_allowed_accounts() -> list[str]:
    """Get allowed accounts for the active broker."""
    if context.active_broker == "hyperliquid":
        return _HL_ACCOUNTS
    return _IBKR_ACCOUNTS


def get_max_order_quantity() -> float:
    """Get max order quantity for the active broker."""
    if context.active_broker == "hyperliquid":
        return _HL_MAX_QTY
    return _IBKR_MAX_QTY


# === GUARDRAIL VALIDATION FUNCTIONS ===

def validate_account(account: str | None) -> tuple[bool, str]:
    """
    Validate if account is in allowed list for the active broker.

    """
    allowed = get_allowed_accounts()
    if allowed and account:
        # Case-insensitive comparison to support both IBKR (uppercase) and Hyperliquid (lowercase)
        allowed_lower = [a.lower() for a in allowed]
        if account.lower() not in allowed_lower:
            return False, (
                f"🚫 GUARDRAIL BLOCKED: Account '{account}' is not in the allowed accounts list.\n"
                f"Allowed accounts: {', '.join(allowed)}"
            )
    return True, ""


def validate_quantity(quantity: float) -> tuple[bool, str]:
    """
    Validate if quantity is within allowed limit for the active broker.

    """
    max_qty = get_max_order_quantity()
    if quantity > max_qty:
        return False, (
            f"🚫 GUARDRAIL BLOCKED: Order quantity {quantity} exceeds maximum allowed ({max_qty}).\n"
            f"Max allowed per trade: {max_qty} units"
        )
    return True, ""


def validate_order_guardrails(account: str | None, quantity: float) -> tuple[bool, str]:
    """
    Validate order against all hardcoded guardrails.

    """
    # Check account
    is_valid, error_msg = validate_account(account)
    if not is_valid:
        return False, error_msg

    # Check quantity
    is_valid, error_msg = validate_quantity(quantity)
    if not is_valid:
        return False, error_msg

    return True, ""

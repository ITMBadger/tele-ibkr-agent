# guardrails.py - Centralized trading guardrails and safety checks.
"""
This module contains all hardcoded guardrails that protect against
unintended trades and enforce safety limits.

All guardrails are loaded from .env and enforced in order_service.submit_order().

To modify guardrails, update .env file:
- ALLOWED_ACCOUNTS: Comma-separated list of allowed account IDs
- MAX_ORDER_QUANTITY: Maximum shares per order
- SLIPPAGE_TOLERANCE: Maximum % difference between trigger and market price
"""

import os


# === GUARDRAIL CONFIGURATION (loaded from .env) ===

def load_guardrails() -> tuple[list[str], int, float]:
    """
    Load guardrail settings from environment variables.

    Returns:
        (allowed_accounts, max_quantity, slippage_tolerance)
    """
    allowed_accounts_str = os.getenv("ALLOWED_ACCOUNTS", "")
    allowed_accounts = [acc.strip().upper() for acc in allowed_accounts_str.split(",") if acc.strip()]
    max_quantity = int(os.getenv("MAX_ORDER_QUANTITY", "100"))
    slippage_tolerance = float(os.getenv("SLIPPAGE_TOLERANCE", "1.0"))  # Default 1%
    return allowed_accounts, max_quantity, slippage_tolerance


# Load guardrails on module import
ALLOWED_ACCOUNTS, MAX_ORDER_QUANTITY, SLIPPAGE_TOLERANCE = load_guardrails()


# === GUARDRAIL VALIDATION FUNCTIONS ===

def validate_account(account: str | None) -> tuple[bool, str]:
    """
    Validate if account is in allowed list.

    Args:
        account: Account ID to validate

    Returns:
        (is_valid, error_message)
    """
    if ALLOWED_ACCOUNTS and account:
        if account.upper() not in ALLOWED_ACCOUNTS:
            return False, (
                f"🚫 GUARDRAIL BLOCKED: Account '{account}' is not in the allowed accounts list.\n"
                f"Allowed accounts: {', '.join(ALLOWED_ACCOUNTS)}"
            )
    return True, ""


def validate_quantity(quantity: int) -> tuple[bool, str]:
    """
    Validate if quantity is within allowed limit.

    Args:
        quantity: Number of shares

    Returns:
        (is_valid, error_message)
    """
    if quantity > MAX_ORDER_QUANTITY:
        return False, (
            f"🚫 GUARDRAIL BLOCKED: Order quantity {quantity} exceeds maximum allowed ({MAX_ORDER_QUANTITY}).\n"
            f"Max allowed per trade: {MAX_ORDER_QUANTITY} shares"
        )
    return True, ""


def validate_order_guardrails(account: str | None, quantity: int) -> tuple[bool, str]:
    """
    Validate order against all hardcoded guardrails.

    Args:
        account: Account ID
        quantity: Number of shares

    Returns:
        (is_valid, error_message) - if is_valid is False, error_message explains why
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

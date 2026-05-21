"""Pseudonymization utilities for privacy-preserving data processing.

This module implements Layer 1 of the BizSentinel privacy architecture:
deterministic pseudonymization of customer identifiers using HMAC-SHA256.

Pseudonymization vs Anonymization:
- Pseudonymization replaces identifiers with reversible tokens (with secret key)
- Anonymization irreversibly removes all identifying information
- Pseudonymization maintains referential integrity for joins while protecting identity

Why HMAC-SHA256 with salt:
- HMAC ensures cryptographic strength and prevents length-extension attacks
- SHA-256 produces fixed 256-bit output suitable for database keys
- Salt prevents rainbow table attacks and ensures domain separation
- Plain hashes (SHA-256 without salt) are vulnerable to precomputation attacks

Data Scope:
- Pseudonymized: customer_id fields in all raw data tables
- Not pseudonymized: behavioral features, aggregated statistics, model outputs
"""

import hashlib
import hmac
import re

import pandas as pd


def pseudonymize_customer_id(customer_id: str, salt: str) -> str:
    """Deterministically hash a customer ID using HMAC-SHA256.

    This function is a pure function, safe to test.

    Args:
        customer_id: Original customer identifier
        salt: Cryptographic salt for HMAC

    Returns:
        Hex-encoded SHA-256 hash of customer_id

    Raises:
        ValueError: If customer_id or salt is empty
    """
    if not customer_id:
        raise ValueError("customer_id cannot be empty")
    if not salt:
        raise ValueError("salt cannot be empty")

    return hmac.new(salt.encode("utf-8"), customer_id.encode("utf-8"), hashlib.sha256).hexdigest()


def validate_hash_format(hash_value: str) -> bool:
    """Validate that a string is a valid SHA-256 hex digest.

    Args:
        hash_value: String to validate

    Returns:
        True if hash_value is a valid 64-character hex string
    """
    return bool(re.fullmatch(r"[0-9a-f]{64}", hash_value))


def pseudonymize_dataframe(
    df: pd.DataFrame, id_column: str, salt: str, drop_original: bool = True
) -> pd.DataFrame:
    """Apply pseudonymization to a customer ID column in a DataFrame.

    Args:
        df: Input DataFrame
        id_column: Name of column containing customer IDs
        salt: Cryptographic salt for HMAC
        drop_original: Whether to remove the original ID column

    Returns:
        Copy of DataFrame with new pseudonymized column

    Raises:
        ValueError: If id_column doesn't exist or salt is empty
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")
    if not salt:
        raise ValueError("salt cannot be empty")

    # Create a copy to avoid mutating input
    result_df = df.copy()

    # Apply pseudonymization to the ID column
    hash_column = f"{id_column}_hash"
    result_df[hash_column] = result_df[id_column].apply(lambda x: pseudonymize_customer_id(x, salt))

    # Remove original column if requested
    if drop_original:
        result_df = result_df.drop(columns=[id_column])

    return result_df


if __name__ == "__main__":
    # Smoke test for development
    test_id = "customer_12345"
    test_salt = "test_salt_67890"

    # Generate hash
    hash_result = pseudonymize_customer_id(test_id, test_salt)
    print(f"Pseudonymized '{test_id}' -> {hash_result}")

    # Validate format
    is_valid = validate_hash_format(hash_result)
    print(f"Hash format valid: {is_valid}")

    # Test with DataFrame
    test_df = pd.DataFrame(
        {"customer_id": ["cust_a", "cust_b", "cust_c"], "value": [100, 200, 300]}
    )

    pseudonymized_df = pseudonymize_dataframe(test_df, "customer_id", test_salt)
    print("\nDataFrame pseudonymization:")
    print(pseudonymized_df)

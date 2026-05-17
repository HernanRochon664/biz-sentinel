"""Privacy module for BizSentinel.

This module implements Layer 1 of the privacy design: pseudonymization.
See docs/privacy_design.md for the complete privacy architecture.

Exports:
    pseudonymize_customer_id: Deterministic HMAC-SHA256 hashing of customer IDs
    validate_hash_format: Regex validation for SHA-256 hex digests
"""

from .pseudonymizer import pseudonymize_customer_id, validate_hash_format

__all__ = ['pseudonymize_customer_id', 'validate_hash_format']
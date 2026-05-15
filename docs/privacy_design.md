# BizSentinel Privacy Architecture

Privacy is a first-class design concern in BizSentinel, not an afterthought. This document describes the technical and architectural decisions made to protect sensitive e-commerce data while maintaining model utility for business intelligence.

## 1. Privacy Threat Model

In an e-commerce context, several types of data are considered sensitive:

- **Customer Identifiers**: Customer IDs, names, email addresses, phone numbers
- **Transactional Data**: Purchase history, itemized receipts, payment information
- **Behavioral Patterns**: Browsing patterns, purchase frequency, seasonal trends
- **Geospatial Information**: Shipping addresses, delivery locations

### Primary Privacy Threats

1. **Model Inversion**: An attacker could attempt to reconstruct individual customer records by analyzing the trained model's parameters and outputs. This is particularly concerning for our supervised learning Module C (Churn/Risk Scoring).

2. **Re-identification from Aggregated Features**: Even when individual identifiers are removed, combinations of features can sometimes be used to re-identify individuals, especially in sparse datasets with unique behavioral patterns.

### Out of Scope

This project explicitly does not implement:
- Federated learning
- Homomorphic encryption
- Full GDPR compliance (as this is a portfolio project, not a production legal system)

## 2. Privacy Techniques Implemented

### Layer 1 — Pseudonymization (at ingestion)

All customer_id values are replaced with a deterministic HMAC-SHA256 hash before entering any ML pipeline. This ensures:

- Original identifiers never enter the ML training or inference systems
- Records can still be joined across tables using the hashed identifiers
- Salt is stored separately (not in the same database as the pseudonymized data)
- Reversible only with access to the salt, which is tightly controlled

```
raw data → hashing function → pseudonymized data → pipeline
```

### Layer 2 — Differential Privacy (at training — Module C only)

We implement differential privacy specifically for Module C (Churn/Risk Scoring), our supervised learning component:

- **Library**: diffprivlib (IBM's differential privacy library)
- **Mechanism**: DP-SGD (differentially private stochastic gradient descent) applied to the LightGBM surrogate or a logistic regression baseline
- **Parameter**: epsilon (ε) — tuned experimentally in Phase 3 with an upper bound of ε ≤ 5
- **Expected Trade-off**: Accuracy loss of 3–8% is acceptable for privacy protection
- **Documentation**: A comparison table of model metrics at ε = [0.5, 1, 2, 5, ∞ (no DP)]

### Layer 3 — API Security (at serving)

Security controls are implemented at our FastAPI endpoints:

- **JWT-based Authentication**: All API endpoints require JWT tokens for access
- **Scoped Access Control**: 
  - Raw data endpoints: Internal use only with strict access controls
  - Scoring endpoints: Exposed externally but only return risk scores, not raw features
- **MCP Server**: Read-only tools with no access to raw data, only aggregated insights

## 3. Privacy vs Utility Trade-off

We acknowledge that privacy has a quantifiable cost in terms of model performance:

- **Design Decision**: 60% weight on model utility, 40% on privacy protection
- **Target Range**: ε ≈ 1–5 based on our experiments
- **Documentation**: This trade-off is explicitly documented through experimental results rather than treated as a black-box parameter

The trade-off will be systematically evaluated using our validation framework to optimize for both business value and privacy protection.

## 4. What This Project Does NOT Implement

This portfolio project intentionally excludes several advanced privacy techniques:

- **Federated Learning**: Not implemented due to scope constraints
- **Homomorphic Encryption**: Would add significant computational overhead not justified for this demonstration
- **Full GDPR Compliance**: While privacy preserving, this is not a production legal system

These would be natural next steps for a production deployment:
1. Federated learning would allow training on device data without centralizing it
2. Homomorphic encryption would enable computation on encrypted data
3. Full legal compliance framework would address all GDPR requirements

## 5. References

1. IBM diffprivlib documentation - https://github.com/IBM/differential-privacy-library
2. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." Foundations and Trends® in Theoretical Computer Science, 9(3–4), 211–407.
3. NIST Privacy Framework: A Tool for Improving Privacy through Enterprise Risk Management - https://www.nist.gov/privacy-framework
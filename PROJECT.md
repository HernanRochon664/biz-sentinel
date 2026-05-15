# BizSentinel: Machine Learning Project Framing

## 1. Problem Statement

### SME Context
BizSentinel targets small to medium-sized e-commerce businesses that collect transactional data but lack dedicated data science teams. These businesses need actionable insights from their data without requiring extensive technical expertise or costly consulting services.

### Business Problems
This project addresses three interconnected business challenges:
1. **Anomaly Detection**: Identify suspicious transactions or unusual patterns that may indicate fraud, operational issues, or business opportunities.
2. **Customer Segmentation**: Understand distinct customer groups to enable personalized marketing and improve retention strategies.
3. **Churn/Risk Prediction**: Predict which customers are likely to stop purchasing or pose credit/default risks.

### Connection Between Problems
These modules form a layered intelligence system:
- Module A (anomaly detection) identifies outliers at the transaction level
- Module B (segmentation) aggregates behavioral patterns to create customer profiles
- Module C (churn scoring) leverages both anomaly scores and segment labels to predict future risk

Each layer builds upon the previous one, creating a comprehensive view of business health and customer behavior.

## 2. ML Task Definitions

### Module A — Anomaly Detection
- **Task Type**: Unsupervised anomaly detection
- **Input Features**: Order value, purchase frequency, temporal patterns, payment methods, geographic distribution
- **Output**: Anomaly score (0-1) per transaction/customer; binary flag using threshold tuning
- **Algorithms**: Isolation Forest (primary), Autoencoder (secondary)
- **Metrics**: 
  - Primary: AUC-PR (Precision-Recall Area Under Curve)
  - Secondary: Contamination rate sensitivity analysis
- **Notes**: Focus on business-relevant anomalies, not just statistical outliers

### Module B — Customer Segmentation
- **Task Type**: Unsupervised clustering
- **Input Features**: RFM (Recency, Frequency, Monetary) + review sentiment + product category diversity
- **Output**: Cluster label for each customer (targeting 4-6 clusters)
- **Algorithms**: K-Means (primary), DBSCAN (secondary for noise handling)
- **Evaluation**: Silhouette Score, Davies-Bouldin Index, business interpretability
- **Integration**: Cluster labels feed as categorical features into Module C

### Module C — Churn/Risk Scoring
- **Task Type**: Supervised binary classification
- **Input Features**: RFM features + anomaly scores (from Module A) + cluster labels (from Module B) + temporal trends
- **Output**: Churn probability per customer (0-1 continuous score)
- **Algorithm**: LightGBM with SHAP for interpretability
- **Metrics**:
  - ROC-AUC
  - F1 Score
  - Precision@K (top 10% riskiest customers)
- **Privacy Constraint**: Model trained with Differential Privacy (diffprivlib), epsilon parameter tuning in Phase 3

## 3. Feature Pipeline

### Data Sources
Olist Brazilian E-commerce Public Dataset including:
- Orders and order items
- Customer information
- Payment details
- Product reviews
- Geolocation data

### Engineered Features (12+)
1. **Monetary**: Average order value (AOV) = total_spent / number_of_orders
2. **Frequency**: Purchase frequency = number_of_orders / days_since_first_purchase
3. **Recency**: Days since last purchase
4. **Payment Diversity**: Count of distinct payment types used
5. **Geographic Spread**: Number of distinct states where customer has shipped orders
6. **Review Sentiment**: Average review score with text sentiment analysis
7. **Product Category Diversity**: Number of unique categories purchased
8. **Temporal Pattern**: Weekend vs weekday purchase ratio
9. **Return Frequency**: Ratio of canceled/returned orders to total orders
10. **Payment Delay**: Average delay between order date and payment approval
11. **Basket Size**: Average number of items per order
12. **Anomaly Score Aggregation**: Mean anomaly score from Module A applied per customer

### Feature Groups
- **Temporal**: Recency, weekend/weekday patterns, payment delays
- **Behavioral**: Frequency, return rate, product diversity
- **Monetary**: AOV, total spent, payment method diversity
- **Quality**: Review sentiment scores, delivery satisfaction

## 4. Data Split Strategy

### Approach
Temporal split (not random) to prevent data leakage and simulate real-world deployment conditions.

### Rationale
E-commerce data has temporal dependencies; future events should not influence model training on past events. This approach better reflects how models will perform in production.

### Implementation
Specific train/validation/test cutoff dates will be determined after initial data exploration to ensure adequate sample sizes for all segments.

## 5. Business Success Criteria

### Module A
- Identifies anomalies that trigger actionable business responses
- Reduces false positives to minimize alert fatigue
- Provides interpretable reasons for flagged anomalies

### Module B
- Produces clearly defined segments understandable by non-technical stakeholders
- Segments exhibit distinct purchasing behaviors enabling targeted campaigns
- Maintains stability over time while adapting to seasonal trends

### Module C
- Accurately identifies top 10-15% of at-risk customers
- Maintains low false positive rate (<30%) in high-risk predictions
- Provides feature importance explanations for business decisions

### Privacy
- Achieves acceptable model performance with epsilon ≤ 5
- Ensures no individual customer's data can be reconstructed from model outputs

## 6. Constraints and Assumptions

### Technical Constraints
- Batch inference (daily) as primary deployment mode
- No real-time requirements for MVP
- All models require full SHAP-based interpretability
- System must be reproducible end-to-end via single `docker-compose up` command (Phase 6 goal)

### Modeling Assumptions
- Historical patterns will remain relevant for near-term predictions
- Business definitions of "anomaly" and "churn" align with statistical findings
- Customer segments identified will remain relatively stable over monthly periods

### Privacy Requirements
- Differential privacy implementation is mandatory, not optional
- Pseudonymization applied during data ingestion phase
- Regular auditing of privacy parameters during model retraining cycles
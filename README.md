# BizSentinel

An end-to-end ML Engineering portfolio project: a business intelligence and anomaly detection platform for SMEs (small/medium e-commerce businesses). BizSentinel processes real e-commerce data to detect unusual patterns, segment customers, and predict churn risk while preserving data privacy through pseudonymization and differential privacy techniques.

## Motivation

Small and medium e-commerce businesses often lack the resources to implement sophisticated monitoring systems. Anomaly detection helps them identify fraud, inventory issues, and operational problems early. Privacy is critical as these businesses handle sensitive customer data. Proper MLOps practices ensure reproducible, reliable ML systems that can adapt to changing business needs.

## System Overview

```
┌─────────────┐    ┌──────────────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────────────────────────┐
│   Data      │───▶│   Kedro Pipeline │───▶│ MLflow   │───▶│ Prefect     │───▶│ Deployment Options          │
│(Olist)      │    │                  │    │ Tracking │    │ Flows       │    │                             │
└─────────────┘    └──────────────────┘    └──────────┘    └─────────────┘    │  Batch Inference            │
                                                                              │  REST API (FastAPI)         │
                                                                              │  Interactive Dashboard (Dash)│
                                                                              │  MCP Server (FastMCP+Ollama)│
                                                                              └─────────────────────────────┘
```

## ML Modules

1. **Anomaly Detection (Unsupervised)**: Identifies unusual patterns in business metrics like sales spikes/drops, inventory anomalies, and suspicious transactions.
2. **Customer Segmentation (Unsupervised)**: Groups customers based on purchasing behavior to enable targeted marketing strategies.
3. **Churn/Risk Scoring (Supervised)**: Predicts which customers are likely to stop purchasing or which accounts present credit/default risks.

## Tech Stack

| Category        | Technologies                                   |
|----------------|------------------------------------------------|
| ML/Data        | Kedro, Scikit-learn, Pandas, NumPy             |
| Orchestration  | Prefect, MLflow                                |
| Storage        | TBD                                            |
| Serving        | FastAPI, Dash, FastMCP                         |
| Privacy        | Pseudonymization, Differential Privacy         |
| Infrastructure | Docker, Docker Compose, GitHub Actions, DigitalOcean |

## Project Phases

0. Environment setup and tooling configuration
1. Data ingestion and exploratory analysis
2. Feature engineering and preprocessing pipelines
3. Model development and training with privacy considerations
4. Pipeline orchestration and experiment tracking integration
5. Deployment infrastructure setup with multiple serving options
6. Testing, validation, and documentation completion

## Dataset

This project uses the [Olist Brazilian E-commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) from Kaggle. It contains information from over 100,000 orders made at Olist Store between 2016 and 2018, including customer demographics, order details, product categories, and delivery information. This rich dataset provides realistic business scenarios for anomaly detection while maintaining enough complexity for meaningful customer segmentation.

## Privacy Design

Privacy is a first-class concern in BizSentinel. During data ingestion, we apply pseudonymization techniques to decouple personal identifiers from behavioral data. For model training, we implement differential privacy mechanisms to prevent reconstruction of individual records from model outputs, ensuring compliance with data protection regulations.

## Deployment Options

BizSentinel supports four deployment modes:
1. **Batch Inference**: Periodic processing of new data batches
2. **REST API**: Real-time predictions via FastAPI endpoints
3. **Interactive Dashboard**: Business intelligence dashboard with Dash for data visualization
4. **MCP Server**: Agent-based querying via FastMCP + Ollama integration (differentiating feature)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
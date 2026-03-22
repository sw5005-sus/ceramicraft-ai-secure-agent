# ceramicraft-ai-secure-agent

AI-powered secure agent for detecting risky users and abnormal transactions in Ceramicraft. The service combines deterministic business rules, a trained machine-learning model, and an OpenAI LLM inside a **LangGraph** workflow to produce real-time fraud risk assessments with human-readable explanations.

---

## Package Structure

```
src/ceramicraft_ai_secure_agent/
├── api/
│   └── risk_api.py          # FastAPI router – POST /risk/check
├── data/
│   └── mock_data.json       # Sample transactions for local testing
├── model/
│   └── fraud_model.pkl      # Serialised scikit-learn fraud-detection model
├── service/
│   ├── agent_service.py     # LangGraph orchestrator (main entry point)
│   ├── feature_service.py   # Feature extraction + extract_features_tool (@tool)
│   ├── rule_engine.py       # Business rules + evaluate_rules_tool (@tool)
│   ├── ml_model.py          # ML inference + predict_tool (@tool)
│   └── risk_scoring.py      # Composite risk scoring
├── utils/
│   └── logger.py            # Shared logging helper
└── app.py                   # FastAPI application factory
```

---

## Implemented Functions

### `feature_service.py`
| Function | Description |
|---|---|
| `extract_features(transaction)` | Converts a raw transaction dict into a numeric feature vector (`amount`, `is_high_risk_country`, `is_high_risk_category`, `amount_log`, `is_large_amount`). |
| `extract_features_tool` | LangChain `@tool`-decorated wrapper used by the LangGraph agent. |

### `rule_engine.py`
| Function | Description |
|---|---|
| `evaluate_rules(features)` | Applies hard-coded business rules (large amount, high-risk country/category, combined rule) and returns triggered rule names and a boolean risk flag. |
| `evaluate_rules_tool` | LangChain `@tool`-decorated wrapper used by the LangGraph agent. |

### `ml_model.py`
| Function | Description |
|---|---|
| `predict(features)` | Loads the serialised scikit-learn model and returns `fraud_probability` and `ml_prediction` for a feature vector. |
| `predict_tool` | LangChain `@tool`-decorated wrapper used by the LangGraph agent. |

### `risk_scoring.py`
| Function | Description |
|---|---|
| `compute_score(rule_result, ml_result)` | Combines the rule signal (weight 0.4) and ML fraud probability (weight 0.6) into a normalised composite score mapped to `HIGH` / `MEDIUM` / `LOW`. |

### `agent_service.py` – LangGraph Orchestrator
The agent is implemented as a deterministic **LangGraph `StateGraph`** with five sequential nodes:

```
extract_features → evaluate_rules → predict → compute_score → llm_judge → END
```

| Node | Tool used | Description |
|---|---|---|
| `extract_features` | `extract_features_tool` | Extracts numeric features from the raw transaction. |
| `evaluate_rules` | `evaluate_rules_tool` | Runs business-rule checks on the features. |
| `predict` | `predict_tool` | Obtains fraud probability from the ML model. |
| `compute_score` | `risk_scoring.compute_score` | Computes the composite risk score and level. |
| `llm_judge` | `ChatOpenAI` (gpt-4o-mini) | Uses an OpenAI LLM to produce a concise risk recommendation. Falls back to a rule-based recommendation when `OPENAI_API_KEY` is not set. |

The public entry point `assess_risk(transaction)` returns:

```json
{
  "transaction_id": "txn_001",
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "triggered_rules": ["large_amount", "high_risk_country", "large_amount_in_high_risk_country"],
  "fraud_probability": 0.9,
  "recommendation": "Block this transaction immediately – multiple high-risk signals detected."
}
```

---

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure OpenAI (optional – enables LLM recommendations)
```bash
export OPENAI_API_KEY=sk-...
```

### 3. Start the server
```bash
uv run uvicorn ceramicraft_ai_secure_agent.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/risk/check' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "transaction_id": "txn_002",
  "amount": 12500.0,
  "merchant_category": "unknown",
  "country": "NG",
  "user_id": "1",
  "merchant": "01982"
}'
```

### 5. Run tests
```bash
uv run pytest
```

### 6. Model Training
```
uv sync --group train
uv run --group train python src/ceramicraft_ai_secure_agent/data/train_data_gen.py
uv run --group train python src/ceramicraft_ai_secure_agent/model/train_model.py
# open mlflow
source .venv/bin/activate
mlflow ui
```

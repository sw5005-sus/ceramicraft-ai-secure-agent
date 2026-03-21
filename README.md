# ceramicraft-ai-secure-agent
1. install requirements
```
uv sync
```
2. start app
```
uv run uvicorn ceramicraft_ai_secure_agent.app:app --reload
```
3. test api
```
curl -X 'POST' \
  'http://127.0.0.1:8000/risk/check' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "transaction_id": "txn_002",
  "amount": 12500.0,
  "merchant_category": "unknown",
  "country": "NG",
  "user_id":"1",
  "merchant":"01982"
}'
```
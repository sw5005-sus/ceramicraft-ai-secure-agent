import os

import mlflow

print("1. importing mlflow ok")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
print("2. tracking uri set")

prompt_v1 = [
    {
        "role": "system",
        "content": "You are a fraud risk assessment expert.",
    },
    {
        "role": "user",
        "content": (
            "Based on the following automated risk analysis, "
            "provide exactly one concise "
            "recommendation sentence.\n\n"
            "Risk Score: {{risk_score}}\n"
            "Risk Level: {{risk_level}}\n"
            "Triggered Rules: {{triggered_rules}}\n"
            "Fraud Probability: {{fraud_probability}}"
        ),
    },
]

print("3. before register_prompt")
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
pv = mlflow.genai.register_prompt(
    name="fraud_recommendation_prompt",
    template=prompt_v1,
    commit_message="Initial fraud recommendation prompt",
)
print(pv)
print("4. registered:", pv)

mlflow.genai.set_prompt_alias(
    "fraud_recommendation_prompt",
    alias="production",
    version=1,
)

print("5. set alias to production")

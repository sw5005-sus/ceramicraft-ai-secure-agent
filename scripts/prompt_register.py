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

prompt_v2 = [
    {
        "role": "system",
        "content": """
        You are an AI fraud risk triage assistant for an e-commerce platform.
        Your job is to refine action recommendations based on structured fraud signals.
        You are NOT the primary fraud classifier.
        The risk score and fraud probability are already computed by upstream systems.

        Decision policy:
        - allow: low risk, no further action needed
        - watchlist: suspicious but insufficient evidence, allow but enhance monitoring
        - manual_review: medium/high risk, requires human review before further action
        - block: high confidence fraud risk, request should be blocked

        Return ONLY valid JSON with the following schema:
        {
            "recommended_action": "allow|watchlist|manual_review|block",
            "reason": "brief summary of key signals (max 10 words)",
            "analyst_summary": "brief 1-2 sentence explanation for a human analyst",
            "confidence": "low|medium|high"
        }
        Do not return markdown. Do not add extra text.
    """,
    },
    {
        "role": "user",
        "content": """
        Evaluate fraud signals and recommend an action.

        Risk Score: {{risk_score}}
        Risk Level: {{risk_level}}
        Triggered Rules: {{triggered_rules}}
        Fraud Probability: {{fraud_probability}}
        Previous Status: {{previous_status}}
        Feature Snapshot: {{feature_snapshot}}

        Guidance:
            - Prefer watchlist for moderate suspicion without strong evidence.
            - Prefer manual_review for borderline or multi-signal suspicious cases.
            - Prefer block only when the evidence is strong and risk is high.
            - Be conservative. Do not escalate unless justified by the signals.
    """,
    },
]

print("3. before register_prompt")
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
pv = mlflow.genai.register_prompt(
    name="fraud_recommendation_prompt",
    template=prompt_v2,
    commit_message="Initial fraud recommendation prompt",
)
print(pv)
print("4. registered:", pv)

mlflow.genai.set_prompt_alias(
    "fraud_recommendation_prompt",
    alias="production",
    version=2,
)

print("5. set alias to production")

import json


class RiskUserReview:
    def __init__(
        self,
        user_id: int,
        create_time: int,
        confidence: str,
        analyst_summary: str,
        decision: int,
        decision_source: str = "",
        risk_score: float = 0.0,
        risk_level: str = "",
        rule_score: float = 0.0,
        fraud_probability: float = 0.0,
        triggered_rules: list[str] | None = None,
        ml_top_contributor: list[tuple[str, float]] | None = None,
    ):
        self.user_id = user_id
        self.create_time = create_time
        self.confidence = confidence
        self.analyst_summary = analyst_summary
        self.decision_source = decision_source
        self.decision = decision
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.rule_score = rule_score
        self.fraud_probability = fraud_probability
        rule_list = triggered_rules if triggered_rules is not None else []
        self.rules = ",".join(rule_list)
        self.ml_top_contributor = (
            json.dumps(ml_top_contributor) if ml_top_contributor is not None else "[]"
        )

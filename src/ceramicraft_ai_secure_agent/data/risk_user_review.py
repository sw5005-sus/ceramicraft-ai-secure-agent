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
        triggered_rules: list[str] = [],
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
        self.rules = ",".join(triggered_rules)

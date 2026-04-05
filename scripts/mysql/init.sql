CREATE TABLE RiskUserReview (
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    create_time BIGINT NOT NULL,
    confidence VARCHAR(63) NOT NULL,
    analyst_summary TEXT NOT NULL,
    decision tinyint NOT NULL default 0,
    decision_source VARCHAR(63) DEFAULT '',
    risk_score FLOAT DEFAULT 0.0,
    risk_level VARCHAR(63) DEFAULT '',
    rule_score FLOAT DEFAULT 0.0,
    fraud_probability FLOAT DEFAULT 0.0,
    rules TEXT, 
    UNIQUE KEY uniq_user_id (user_id)
) engine = InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
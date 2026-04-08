from enum import IntEnum


class RiskUserReviewDecision(IntEnum):
    MANUAL_REVIEW = 1
    BLOCK = 2
    WATCHLIST = 3
    ALLOW = 4
    UNRECOGNIZED = 0

    RESOLVED_BLOCK = 10
    RESOLVED_WHITELIST = 11
    RESOLVED_WATCHLIST = 12

    @classmethod
    def from_str(cls, action_str: str) -> "RiskUserReviewDecision":
        mapping = {
            "manual_review": cls.MANUAL_REVIEW,
            "block": cls.BLOCK,
            "watchlist": cls.WATCHLIST,
            "allow": cls.ALLOW,
            "resoved_block": cls.RESOLVED_BLOCK,
            "resoved_whitelist": cls.RESOLVED_WHITELIST,
            "resoved_watchlist": cls.RESOLVED_WATCHLIST,
        }
        return mapping.get(action_str.lower(), cls.UNRECOGNIZED)

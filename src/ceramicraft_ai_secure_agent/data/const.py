from enum import IntEnum


class RiskUserReviewDecision(IntEnum):
    MANUAL_REVIEW = 1
    BLOCK = 2
    WATCHLIST = 3
    ALLOW = 4
    UNRECOGNIZED = 0

    @classmethod
    def from_str(cls, action_str: str) -> "RiskUserReviewDecision":
        mapping = {
            "manual_review": cls.MANUAL_REVIEW,
            "block": cls.BLOCK,
            "watchlist": cls.WATCHLIST,
            "allow": cls.ALLOW,
        }
        return mapping.get(action_str.lower(), cls.UNRECOGNIZED)

from enum import IntEnum
from http.client import UNAUTHORIZED


class RiskUserReviewDecision(IntEnum):
    MANUAL_REVIEW = 1
    BLOCK = 2
    WATCHLIST = 3
    ALLOW = 4
    UNRECOGNIZED = 0

    @classmethod
    def from_str(cls, action_str: str):
        mapping = {
            "manual_review": cls.MANUAL_REVIEW,
            "block": cls.BLOCK,
            "watchlist": cls.WATCHLIST,
            "allow": cls.ALLOW,
        }
        return mapping.get(action_str.lower(), UNAUTHORIZED)

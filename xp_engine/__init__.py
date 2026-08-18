"""XP Engine — public API."""

from .builds import (
    BADGE_TIERS,
    BADGE_XP_THRESHOLDS,
    BUILD_DESCRIPTIONS,
    BUILD_DISPLAY_NAMES,
    BUILD_RELEVANCE,
    INTENSITY_MULTIPLIERS,
    LEGENDARY_BADGE_INDEX,
    BuildType,
    Intensity,
    calculate_level,
    get_badge_index_for_xp,
    get_badge_name,
    get_relevance_score,
    get_xp_relevance_multiplier,
    is_primary_activity,
)
from .gamer_mechanics import (
    GAMER_BOSS_DAY_INTERVAL,
    GAMER_COMBO_BONUS_MAX,
    GAMER_COMBO_BONUSES,
    GAMER_OVERTIME_LIMIT,
    GAMER_SIDE_QUEST_MULTIPLIER,
    GAMER_SPEEDRUN_BONUS_XP,
    GAMER_SPEEDRUN_THRESHOLD_MINS,
    GamerXPResult,
    calculate_gamer_xp,
    get_combo_multiplier,
    get_gamer_daily_cap,
    get_gamer_relevance_multiplier,
    is_boss_day,
    qualifies_for_speedrun_bonus,
)
from .user_state import (
    ActivityLog,
    BuildProfile,
    UserState,
)
from .xp_calculator import (
    DAILY_CAP_TIERS,
    STREAK_MILESTONES,
    XPResult,
    best_activity_for_xp,
    calculate_xp,
    get_daily_cap,
    get_streak_multiplier,
)

__all__ = [
    # builds
    "BuildType", "Intensity", "INTENSITY_MULTIPLIERS",
    "BADGE_XP_THRESHOLDS", "BADGE_TIERS", "LEGENDARY_BADGE_INDEX",
    "BUILD_RELEVANCE", "BUILD_DISPLAY_NAMES", "BUILD_DESCRIPTIONS",
    "get_relevance_score", "is_primary_activity", "get_xp_relevance_multiplier",
    "get_badge_index_for_xp", "get_badge_name", "calculate_level",
    # xp_calculator
    "XPResult", "STREAK_MILESTONES", "DAILY_CAP_TIERS",
    "get_streak_multiplier", "get_daily_cap", "calculate_xp", "best_activity_for_xp",
    # user_state
    "ActivityLog", "BuildProfile", "UserState",
    # gamer_mechanics
    "GamerXPResult",
    "GAMER_COMBO_BONUSES", "GAMER_COMBO_BONUS_MAX", "GAMER_OVERTIME_LIMIT",
    "GAMER_BOSS_DAY_INTERVAL", "GAMER_SPEEDRUN_BONUS_XP",
    "GAMER_SPEEDRUN_THRESHOLD_MINS", "GAMER_SIDE_QUEST_MULTIPLIER",
    "get_combo_multiplier", "is_boss_day", "get_gamer_daily_cap",
    "qualifies_for_speedrun_bonus", "get_gamer_relevance_multiplier",
    "calculate_gamer_xp",
]

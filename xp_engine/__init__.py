"""XP Engine — public API."""

from .builds import (
    BuildType,
    Intensity,
    INTENSITY_MULTIPLIERS,
    BADGE_XP_THRESHOLDS,
    BADGE_TIERS,
    LEGENDARY_BADGE_INDEX,
    BUILD_RELEVANCE,
    BUILD_DISPLAY_NAMES,
    BUILD_DESCRIPTIONS,
    get_relevance_score,
    is_primary_activity,
    get_xp_relevance_multiplier,
    get_badge_index_for_xp,
    get_badge_name,
    calculate_level,
)

from .xp_calculator import (
    XPResult,
    STREAK_MILESTONES,
    DAILY_CAP_TIERS,
    get_streak_multiplier,
    get_daily_cap,
    calculate_xp,
    best_activity_for_xp,
)

from .user_state import (
    ActivityLog,
    BuildProfile,
    UserState,
)

from .gamer_mechanics import (
    GamerXPResult,
    GAMER_COMBO_BONUSES,
    GAMER_COMBO_BONUS_MAX,
    GAMER_OVERTIME_LIMIT,
    GAMER_BOSS_DAY_INTERVAL,
    GAMER_SPEEDRUN_BONUS_XP,
    GAMER_SPEEDRUN_THRESHOLD_MINS,
    GAMER_SIDE_QUEST_MULTIPLIER,
    get_combo_multiplier,
    is_boss_day,
    get_gamer_daily_cap,
    qualifies_for_speedrun_bonus,
    get_gamer_relevance_multiplier,
    calculate_gamer_xp,
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

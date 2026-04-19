"""
gamer_mechanics.py - Exclusive mechanics for the GAMER build.

The GAMER build has 5 unique mechanics that stack on top of the standard XP formula:

  1. Combo Multiplier  — Back-to-back primary sessions in the same day stack a bonus
                         multiplier: 2nd session +0.2x, 3rd +0.3x, 4th+ +0.4x.
  2. Overtime Mode     — After hitting the daily XP cap, primary GAMER activities
                         can earn up to 150 bonus XP (the "Overtime pool").
  3. Boss Day          — Every 7th consecutive streak day the daily XP cap is doubled.
  4. Speedrun Bonus    — An INTENSE activity completed in ≤25 min earns +15 flat XP.
  5. Side Quest Penalty— Non-Gamer activities score at 0.4x instead of the standard 0.5x.

None of these mechanics apply to other builds.
"""

from dataclasses import dataclass
from typing import Optional

from .builds import BuildType, Intensity, is_primary_activity


# ── Constants ─────────────────────────────────────────────────────────────────

GAMER_COMBO_BONUSES: dict[int, float] = {
    2: 0.20,   # 2nd primary session today → +0.2x
    3: 0.30,   # 3rd primary session today → +0.3x
}
GAMER_COMBO_BONUS_MAX: float = 0.40   # 4th+ primary session today → +0.4x

GAMER_OVERTIME_LIMIT: int  = 150      # Extra XP pool unlocked after daily cap
GAMER_BOSS_DAY_INTERVAL: int = 7      # Every Nth streak day → Boss Day
GAMER_SPEEDRUN_BONUS_XP: int = 15     # Flat bonus for INTENSE activity ≤ threshold
GAMER_SPEEDRUN_THRESHOLD_MINS: int = 25  # Minutes at-or-below to qualify
GAMER_SIDE_QUEST_MULTIPLIER: float = 0.40  # Non-Gamer relevance (vs 0.5x default)


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class GamerXPResult:
    """
    Extended XP result for a GAMER-build session.

    Wraps the base XPResult values and layers on the gamer-specific modifiers
    so the chatbot can explain each bonus individually.
    """
    # Base calculation (mirrored from XPResult for convenience)
    activity_type:     str
    duration_minutes:  int
    intensity:         Intensity
    base_final_xp:     int        # XP from the standard formula (post-cap)

    # ── Gamer modifiers ───────────────────────────────────────────────────────
    combo_multiplier:  float      # Extra multiplier from consecutive sessions (0 if N/A)
    combo_bonus_xp:    int        # Flat XP added by combo (int(base_final_xp * combo_mult))
    is_boss_day:       bool       # True if today is a Boss Day (streak % 7 == 0)
    boss_day_cap_used: bool       # True if Boss Day cap was actually needed
    speedrun_bonus:    int        # Flat +15 if speedrun qualified, else 0
    overtime_xp:       int        # XP drawn from Overtime pool (0 if cap not hit)
    side_quest_penalty:bool       # True if 0.4x penalty was applied

    # ── Final totals ──────────────────────────────────────────────────────────
    gamer_bonus_total: int        # Sum of all gamer bonuses (combo + speedrun + overtime)
    final_xp:          int        # Total XP credited this session

    # ── Remaining pools ───────────────────────────────────────────────────────
    overtime_pool_remaining: int  # Overtime XP left for today

    def summary(self) -> str:
        lines = [
            f"⚡ GAMER SESSION: {self.activity_type} "
            f"({self.duration_minutes} min, {self.intensity.value})",
            f"  Base XP        : {self.base_final_xp}",
        ]
        if self.combo_multiplier > 0:
            lines.append(
                f"  🔥 Combo Bonus  : +{self.combo_bonus_xp} XP "
                f"(+{self.combo_multiplier:.1f}x consecutive session)"
            )
        if self.is_boss_day:
            lines.append("  👾 BOSS DAY     : Daily cap doubled!")
        if self.speedrun_bonus:
            lines.append(
                f"  ⚡ Speedrun      : +{self.speedrun_bonus} XP "
                f"(≤{GAMER_SPEEDRUN_THRESHOLD_MINS} min intense)"
            )
        if self.overtime_xp:
            lines.append(
                f"  🕐 Overtime      : +{self.overtime_xp} XP "
                f"({self.overtime_pool_remaining} OT XP remaining)"
            )
        if self.side_quest_penalty:
            lines.append("  📉 Side Quest    : 0.4x (non-Gamer activity)")
        lines.append(f"  ─────────────────────────────────────")
        lines.append(f"  TOTAL EARNED    : {self.final_xp} XP")
        return "\n".join(lines)


# ── Core Functions ────────────────────────────────────────────────────────────

def get_combo_multiplier(primary_sessions_today: int) -> float:
    """
    Return the extra combo multiplier for the current session.

    primary_sessions_today — number of primary GAMER sessions ALREADY logged today
                             (i.e. before this new session).

    Session 1 (first of day)  → 0.0x bonus (no combo yet)
    Session 2                 → +0.2x
    Session 3                 → +0.3x
    Session 4+                → +0.4x
    """
    next_session_number = primary_sessions_today + 1
    if next_session_number <= 1:
        return 0.0
    return GAMER_COMBO_BONUSES.get(next_session_number, GAMER_COMBO_BONUS_MAX)


def is_boss_day(current_streak: int) -> bool:
    """
    True if today's streak count is a non-zero multiple of GAMER_BOSS_DAY_INTERVAL (7).
    Streak 7, 14, 21, 28 … are Boss Days.
    """
    return current_streak > 0 and (current_streak % GAMER_BOSS_DAY_INTERVAL == 0)


def get_gamer_daily_cap(base_cap: int, current_streak: int) -> int:
    """
    Return the effective daily cap for a Gamer.
    On Boss Days the cap is doubled; otherwise the standard cap applies.
    """
    if is_boss_day(current_streak):
        return base_cap * 2
    return base_cap


def qualifies_for_speedrun_bonus(
    activity_type: str,
    duration_minutes: int,
    intensity: Intensity,
) -> bool:
    """
    True when an INTENSE activity is completed at or under the speedrun threshold.
    Activity must be a Gamer primary activity (checked externally via is_primary_activity).
    """
    return (
        intensity == Intensity.INTENSE
        and duration_minutes <= GAMER_SPEEDRUN_THRESHOLD_MINS
    )


def get_gamer_relevance_multiplier(activity_type: str) -> float:
    """
    Relevance multiplier for GAMER build.

    Primary Gamer activity  → 1.0x  (standard)
    Non-primary activity    → 0.4x  (Side Quest Penalty — stricter than 0.5x default)
    """
    if is_primary_activity(activity_type, BuildType.GAMER):
        return 1.0
    return GAMER_SIDE_QUEST_MULTIPLIER


def calculate_gamer_xp(
    activity_type:           str,
    duration_minutes:        int,
    intensity:               Intensity,
    current_streak:          int,
    current_level:           int,
    daily_xp_so_far:         int,
    primary_sessions_today:  int,   # primary GAMER sessions logged earlier today
    overtime_xp_today:       int,   # Overtime XP already consumed today
) -> GamerXPResult:
    """
    Full GAMER-build XP calculation with all 5 unique mechanics.

    Args:
        activity_type          : key from BUILD_RELEVANCE table
        duration_minutes       : activity duration
        intensity              : Intensity enum value
        current_streak         : consecutive active days including today
        current_level          : user's level (determines base daily cap)
        daily_xp_so_far        : XP already earned today (standard pool)
        primary_sessions_today : primary GAMER activities logged today BEFORE this one
        overtime_xp_today      : Overtime XP already spent today

    Returns:
        GamerXPResult with full breakdown
    """
    from .xp_calculator import (
        get_daily_cap,
        get_streak_multiplier,
        INTENSITY_MULTIPLIERS,
    )
    from .builds import INTENSITY_MULTIPLIERS as _IM

    # ── Base XP formula (same as standard) ───────────────────────────────────
    base_xp         = float(duration_minutes)
    intensity_mult  = _IM[intensity]
    relevance_mult  = get_gamer_relevance_multiplier(activity_type)
    streak_mult, _  = get_streak_multiplier(current_streak)
    raw_xp          = base_xp * intensity_mult * relevance_mult * streak_mult

    is_primary      = is_primary_activity(activity_type, BuildType.GAMER)
    side_quest      = not is_primary  # penalty was applied

    # ── Mechanic 3: Boss Day cap ──────────────────────────────────────────────
    base_cap        = get_daily_cap(current_level)
    effective_cap   = get_gamer_daily_cap(base_cap, current_streak)
    boss_day        = is_boss_day(current_streak)

    # Standard cap enforcement (using Boss Day cap if applicable)
    remaining_std   = max(0, effective_cap - daily_xp_so_far)
    earned_std      = min(int(round(raw_xp)), remaining_std)
    cap_hit         = daily_xp_so_far + earned_std >= effective_cap

    # ── Mechanic 1: Combo Multiplier (primary sessions only) ──────────────────
    combo_mult      = get_combo_multiplier(primary_sessions_today) if is_primary else 0.0
    combo_bonus_xp  = int(earned_std * combo_mult)

    # ── Mechanic 4: Speedrun Bonus (primary sessions only) ────────────────────
    speedrun_bonus  = 0
    if is_primary and qualifies_for_speedrun_bonus(activity_type, duration_minutes, intensity):
        speedrun_bonus = GAMER_SPEEDRUN_BONUS_XP

    # ── Mechanic 2: Overtime Mode (primary sessions only, after cap) ──────────
    overtime_xp     = 0
    if is_primary and (cap_hit or daily_xp_so_far >= effective_cap):
        overtime_pool_left = max(0, GAMER_OVERTIME_LIMIT - overtime_xp_today)
        # Overtime earns on the raw XP that exceeded the cap
        overtime_raw   = max(0, int(round(raw_xp)) - remaining_std)
        overtime_xp    = min(overtime_raw, overtime_pool_left)

    # ── Total ─────────────────────────────────────────────────────────────────
    gamer_bonus_total   = combo_bonus_xp + speedrun_bonus + overtime_xp
    final_xp            = earned_std + gamer_bonus_total
    overtime_remaining  = max(0, GAMER_OVERTIME_LIMIT - overtime_xp_today - overtime_xp)

    return GamerXPResult(
        activity_type          = activity_type,
        duration_minutes       = duration_minutes,
        intensity              = intensity,
        base_final_xp          = earned_std,
        combo_multiplier       = combo_mult,
        combo_bonus_xp         = combo_bonus_xp,
        is_boss_day            = boss_day,
        boss_day_cap_used      = boss_day and (daily_xp_so_far > base_cap),
        speedrun_bonus         = speedrun_bonus,
        overtime_xp            = overtime_xp,
        side_quest_penalty     = side_quest,
        gamer_bonus_total      = gamer_bonus_total,
        final_xp               = final_xp,
        overtime_pool_remaining= overtime_remaining,
    )

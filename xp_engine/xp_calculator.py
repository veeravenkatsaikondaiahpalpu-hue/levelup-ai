"""
xp_calculator.py - Core XP formula engine.

Formula:
  final_xp = base_xp x intensity_multiplier x relevance_multiplier x streak_multiplier
  (capped at the user's daily XP limit)

Rules:
  - base_xp          = duration in minutes (1 min = 1 base XP)
  - intensity        = Light 1.0x | Moderate 1.5x | Intense 2.0x
  - relevance        = Primary activity 1.0x | Non-primary 0.5x
  - streak_multiplier= Day 5+ 1.5x | Day 10+ 1.75x | Day 20+ 2.0x | Day 30+ 2.25x | Day 60+ 2.5x
  - daily cap        = Level 1-10: 600 XP | Level 11-20: 800 XP | Level 21+: 1000 XP
"""

from dataclasses import dataclass
from typing import Optional

from .builds import (
    BuildType,
    Intensity,
    INTENSITY_MULTIPLIERS,
    get_xp_relevance_multiplier,
    is_primary_activity,
)


# ── Streak Milestones (day threshold, multiplier, display name) ───────────────

STREAK_MILESTONES = [
    (60, 2.50, "Legendary Streak"),
    (30, 2.25, "Unstoppable"),
    (20, 2.00, "Blazing"),
    (10, 1.75, "Charged"),
    (5,  1.50, "Power-Up"),
    (0,  1.00, None),
]


# ── Daily XP Cap by Level ─────────────────────────────────────────────────────

DAILY_CAP_TIERS = [
    (21, 1_000),
    (11,   800),
    (1,    600),
]


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class XPResult:
    """Full breakdown of a single XP calculation — used by the chatbot to explain earnings."""
    activity_type:        str
    duration_minutes:     int
    intensity:            Intensity

    base_xp:              float
    intensity_multiplier: float
    relevance_multiplier: float
    streak_multiplier:    float
    raw_xp:               float

    final_xp:             int
    was_capped:           bool

    daily_xp_before:      int
    daily_xp_after:       int
    daily_cap:            int
    daily_xp_remaining:   int

    is_primary:           bool
    power_up_name:        Optional[str]   # set when user just hit a new streak milestone

    def summary(self) -> str:
        """Human-readable summary for chatbot responses."""
        lines = [
            f"{self.activity_type} ({self.duration_minutes} min, {self.intensity.value})",
            f"  Base XP       : {self.base_xp:.0f}",
            f"  x Intensity   : {self.intensity_multiplier}x ({self.intensity.value})",
            f"  x Relevance   : {self.relevance_multiplier}x ({'primary' if self.is_primary else 'non-primary'})",
            f"  x Streak      : {self.streak_multiplier}x",
            f"  Raw XP        : {self.raw_xp:.1f}",
        ]
        if self.was_capped:
            lines.append(f"  Capped at     : {self.final_xp} XP (daily limit reached)")
        else:
            lines.append(f"  Earned        : {self.final_xp} XP")
        lines.append(f"  Daily total   : {self.daily_xp_after} / {self.daily_cap} XP")
        if self.power_up_name:
            lines.append(f"  *** {self.power_up_name} unlocked! ***")
        return "\n".join(lines)


# ── Core Functions ────────────────────────────────────────────────────────────

def get_streak_multiplier(streak_days: int) -> tuple[float, Optional[str]]:
    """
    Returns (multiplier, milestone_name).
    milestone_name is set only when streak_days exactly equals a threshold
    (i.e. the user just hit that milestone today).
    """
    current_multiplier = 1.0
    current_name = None

    for threshold, multiplier, name in STREAK_MILESTONES:
        if streak_days >= threshold:
            current_multiplier = multiplier
            # Only announce when the user lands exactly on the threshold
            if streak_days == threshold and threshold > 0:
                current_name = name
            break

    return current_multiplier, current_name


def get_daily_cap(level: int) -> int:
    """Daily XP cap for a given player level."""
    for min_level, cap in DAILY_CAP_TIERS:
        if level >= min_level:
            return cap
    return 600


def calculate_xp(
    activity_type:    str,
    duration_minutes: int,
    intensity:        Intensity,
    primary_build:    BuildType,
    current_streak:   int,
    current_level:    int,
    daily_xp_so_far:  int,
) -> XPResult:
    """
    Calculate XP earned for a single activity log.

    Args:
        activity_type:    key from BUILD_RELEVANCE table (e.g. "gym_session")
        duration_minutes: how long the activity lasted
        intensity:        Intensity.LIGHT / MODERATE / INTENSE
        primary_build:    user's current primary build
        current_streak:   consecutive active days including today
        current_level:    user's current level (determines daily cap)
        daily_xp_so_far:  XP already earned today before this activity

    Returns:
        XPResult with full breakdown
    """
    base_xp              = float(duration_minutes)
    intensity_mult       = INTENSITY_MULTIPLIERS[intensity]
    relevance_mult       = get_xp_relevance_multiplier(activity_type, primary_build)
    streak_mult, pw_name = get_streak_multiplier(current_streak)

    raw_xp   = base_xp * intensity_mult * relevance_mult * streak_mult
    daily_cap = get_daily_cap(current_level)
    remaining = max(0, daily_cap - daily_xp_so_far)

    earned    = min(int(round(raw_xp)), remaining)
    was_capped = (int(round(raw_xp)) > remaining) and remaining < int(round(raw_xp))

    return XPResult(
        activity_type        = activity_type,
        duration_minutes     = duration_minutes,
        intensity            = intensity,
        base_xp              = base_xp,
        intensity_multiplier = intensity_mult,
        relevance_multiplier = relevance_mult,
        streak_multiplier    = streak_mult,
        raw_xp               = raw_xp,
        final_xp             = earned,
        was_capped           = was_capped,
        daily_xp_before      = daily_xp_so_far,
        daily_xp_after       = daily_xp_so_far + earned,
        daily_cap            = daily_cap,
        daily_xp_remaining   = remaining - earned,
        is_primary           = is_primary_activity(activity_type, primary_build),
        power_up_name        = pw_name,
    )


def best_activity_for_xp(
    primary_build:   BuildType,
    current_streak:  int,
    current_level:   int,
    daily_xp_so_far: int,
    duration_minutes: int = 60,
) -> list[tuple[str, int]]:
    """
    Returns a ranked list of (activity_type, projected_xp) for a given duration,
    sorted by highest XP first. Used by the chatbot's XP Strategist mode.
    """
    from .builds import BUILD_RELEVANCE, Intensity

    results = []
    for activity in BUILD_RELEVANCE:
        result = calculate_xp(
            activity_type    = activity,
            duration_minutes = duration_minutes,
            intensity        = Intensity.INTENSE,
            primary_build    = primary_build,
            current_streak   = current_streak,
            current_level    = current_level,
            daily_xp_so_far  = daily_xp_so_far,
        )
        results.append((activity, result.final_xp))

    return sorted(results, key=lambda x: x[1], reverse=True)

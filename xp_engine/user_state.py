"""
user_state.py - User profile, build progression, streak logic, and shield system.

UserState is the single source of truth for everything the chatbot needs to know
about a user. It gets serialized to the database and injected into the LLM system
prompt on every request.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import json

from .builds import (
    BuildType,
    BADGE_XP_THRESHOLDS,
    BADGE_TIERS,
    LEGENDARY_BADGE_INDEX,
    get_badge_index_for_xp,
    get_badge_name,
    calculate_level,
)
from .xp_calculator import get_streak_multiplier, get_daily_cap
from .gamer_mechanics import (
    calculate_gamer_xp,
    get_gamer_daily_cap,
    is_boss_day as gamer_is_boss_day,
    GAMER_OVERTIME_LIMIT,
)


# ── Activity Log ──────────────────────────────────────────────────────────────

@dataclass
class ActivityLog:
    activity_type:    str
    duration_minutes: int
    intensity:        str
    xp_earned:        int
    logged_at:        str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "activity":  self.activity_type,
            "duration":  self.duration_minutes,
            "intensity": self.intensity,
            "xp":        self.xp_earned,
            "logged_at": self.logged_at,
        }


# ── Build Profile ─────────────────────────────────────────────────────────────

@dataclass
class BuildProfile:
    build_type: BuildType
    total_xp:   int  = 0
    is_legacy:  bool = False   # True = archived, no longer active

    @property
    def current_level(self) -> int:
        return calculate_level(self.total_xp)

    @property
    def badge_index(self) -> int:
        return get_badge_index_for_xp(self.total_xp)

    @property
    def badge_name(self) -> Optional[str]:
        return get_badge_name(self.build_type, self.badge_index)

    @property
    def next_badge_name(self) -> Optional[str]:
        return get_badge_name(self.build_type, self.badge_index + 1)

    @property
    def xp_to_next_badge(self) -> Optional[int]:
        next_idx = self.badge_index + 1
        if next_idx >= len(BADGE_XP_THRESHOLDS):
            return None
        return max(0, BADGE_XP_THRESHOLDS[next_idx] - self.total_xp)

    @property
    def xp_to_next_level(self) -> int:
        """XP needed to reach the next level."""
        lvl = self.current_level
        if lvl < 10:
            next_lvl_xp = lvl * 500
        elif lvl < 20:
            next_lvl_xp = 5_000 + (lvl - 10) * 1_500
        else:
            next_lvl_xp = 20_000 + (lvl - 20) * 3_000
        return max(0, next_lvl_xp - self.total_xp)

    @property
    def is_legendary(self) -> bool:
        """True if the user has reached the Legendary badge — unlocks new build slot."""
        return self.badge_index >= LEGENDARY_BADGE_INDEX

    def add_xp(self, amount: int) -> Optional[str]:
        """
        Add XP to this build profile.
        Returns the new badge name if a badge was just earned, else None.
        """
        old_badge = self.badge_index
        self.total_xp += amount
        new_badge = self.badge_index
        if new_badge > old_badge:
            return self.badge_name
        return None

    def to_dict(self) -> dict:
        return {
            "build":          self.build_type.value,
            "total_xp":       self.total_xp,
            "level":          self.current_level,
            "badge":          self.badge_name or "Unranked",
            "next_badge":     self.next_badge_name,
            "xp_to_badge":    self.xp_to_next_badge,
            "xp_to_level":    self.xp_to_next_level,
            "is_legendary":   self.is_legendary,
            "is_legacy":      self.is_legacy,
        }


# ── User State ────────────────────────────────────────────────────────────────

@dataclass
class UserState:
    user_id:        str
    username:       str
    primary_build:  BuildProfile
    secondary_build: Optional[BuildProfile] = None
    legacy_builds:  list = field(default_factory=list)  # list[BuildProfile]

    # Streak
    current_streak:      int            = 0
    longest_streak:      int            = 0
    last_activity_date:  Optional[str]  = None  # ISO date string

    # Daily XP (resets at 12:00 AM)
    daily_xp_today:   int  = 0
    last_reset_date:  str  = field(default_factory=lambda: date.today().isoformat())

    # Gamer-exclusive session tracking (resets daily alongside daily_xp_today)
    sessions_today:       int = 0   # primary GAMER sessions logged today
    overtime_xp_today:    int = 0   # Overtime XP consumed today (max 150)

    # Streak shields (4 per month, auto-resets on 1st of each month)
    shields_remaining:  int = 4
    shields_reset_month: int = field(default_factory=lambda: date.today().month)


    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _reset_daily_xp_if_needed(self):
        today = date.today().isoformat()
        if self.last_reset_date < today:
            self.daily_xp_today    = 0
            self.sessions_today    = 0
            self.overtime_xp_today = 0
            self.last_reset_date   = today

    def _reset_shields_if_needed(self):
        current_month = date.today().month
        if self.shields_reset_month != current_month:
            self.shields_remaining = 4
            self.shields_reset_month = current_month


    # ── Streak Management ─────────────────────────────────────────────────────

    def update_streak(self, activity_date: Optional[date] = None) -> dict:
        """
        Update streak for today's activity.
        Returns a status dict the chatbot uses to generate messages.

        Keys:
          streak_broken  : True if streak was reset to 1
          shield_used    : True if a shield was consumed
          new_milestone  : name of power-up milestone just hit, or None
          current_streak : updated streak value
        """
        self._reset_shields_if_needed()
        activity_date = activity_date or date.today()
        activity_str  = activity_date.isoformat()

        result = {
            "streak_broken": False,
            "shield_used":   False,
            "new_milestone": None,
            "current_streak": self.current_streak,
        }

        if self.last_activity_date is None:
            # First ever activity
            self.current_streak = 1
        else:
            last = date.fromisoformat(self.last_activity_date)
            days_gap = (activity_date - last).days

            if days_gap == 0:
                pass  # Same day — no streak change
            elif days_gap == 1:
                self.current_streak += 1
            elif days_gap == 2 and self.shields_remaining > 0:
                # Missed exactly one day — auto-consume shield
                self.shields_remaining -= 1
                self.current_streak += 1
                result["shield_used"] = True
            else:
                self.current_streak = 1
                result["streak_broken"] = True

        self.longest_streak = max(self.longest_streak, self.current_streak)
        self.last_activity_date = activity_str

        _, milestone = get_streak_multiplier(self.current_streak)
        result["new_milestone"]  = milestone
        result["current_streak"] = self.current_streak
        return result


    # ── XP Management ────────────────────────────────────────────────────────

    def add_xp(self, amount: int, to_secondary: bool = False) -> dict:
        """
        Add XP to the active build. Handles daily cap and badge upgrades.

        Returns:
          xp_added       : actual XP credited (after cap)
          badge_earned   : new badge name if just earned, else None
          cap_reached    : True if daily cap was hit this session
          daily_total    : XP earned today after this addition
          daily_cap      : effective cap (doubled on Gamer Boss Days)
        """
        self._reset_daily_xp_if_needed()

        level = self.primary_build.current_level
        base_cap = get_daily_cap(level)

        # Gamer build uses a Boss-Day-aware cap
        if self.primary_build.build_type.value == "gamer":
            cap = get_gamer_daily_cap(base_cap, self.current_streak)
        else:
            cap = base_cap

        available = max(0, cap - self.daily_xp_today)
        xp_to_add = min(amount, available)

        build = self.secondary_build if (to_secondary and self.secondary_build) else self.primary_build
        badge_earned = build.add_xp(xp_to_add)
        self.daily_xp_today += xp_to_add

        return {
            "xp_added":    xp_to_add,
            "badge_earned": badge_earned,
            "cap_reached":  self.daily_xp_today >= cap,
            "daily_total":  self.daily_xp_today,
            "daily_cap":    cap,
        }


    # ── Build Switching ───────────────────────────────────────────────────────

    def can_switch_build(self) -> bool:
        """User can unlock a new build only after reaching Legendary on primary."""
        return self.primary_build.is_legendary

    def archive_primary_and_start_new(self, new_build_type: BuildType) -> dict:
        """
        Option B: Archive current primary build and start fresh.
        Previous build's badge and XP are permanently preserved on the profile.
        New build starts at 0 XP with a 500 XP legacy bonus.
        """
        if not self.can_switch_build():
            return {"success": False, "reason": "Must reach Legendary badge first."}

        self.primary_build.is_legacy = True
        self.legacy_builds.append(self.primary_build)

        new_profile = BuildProfile(build_type=new_build_type, total_xp=500)  # legacy bonus
        self.primary_build = new_profile

        return {
            "success":      True,
            "archived":     self.legacy_builds[-1].build_type.value,
            "new_build":    new_build_type.value,
            "legacy_bonus": 500,
        }

    def unlock_secondary_build(self, new_build_type: BuildType) -> dict:
        """
        Option A: Keep primary, unlock a secondary build slot.
        Secondary earns XP at 0.75x on its own primary activities.
        """
        if not self.can_switch_build():
            return {"success": False, "reason": "Must reach Legendary badge first."}
        if self.secondary_build is not None:
            return {"success": False, "reason": "Secondary build slot already active."}

        self.secondary_build = BuildProfile(build_type=new_build_type)
        return {"success": True, "secondary_build": new_build_type.value}


    # ── Context for Chatbot ───────────────────────────────────────────────────

    def to_context_dict(self, recent_logs: list = None) -> dict:
        """
        Flat dict injected into the LLM system prompt on every request.
        Contains everything the chatbot needs to personalize its response.
        """
        self._reset_daily_xp_if_needed()
        self._reset_shields_if_needed()

        streak_mult, _ = get_streak_multiplier(self.current_streak)
        cap = get_daily_cap(self.primary_build.current_level)

        return {
            # Build
            "primary_build":     self.primary_build.build_type.value,
            "primary_badge":     self.primary_build.badge_name or "Unranked",
            "primary_level":     self.primary_build.current_level,
            "primary_xp_total":  self.primary_build.total_xp,
            "xp_to_next_badge":  self.primary_build.xp_to_next_badge,
            "next_badge":        self.primary_build.next_badge_name,
            "xp_to_next_level":  self.primary_build.xp_to_next_level,
            "can_unlock_build":  self.primary_build.is_legendary,

            # Secondary / Legacy
            "secondary_build":  self.secondary_build.build_type.value if self.secondary_build else None,
            "legacy_builds":    [b.to_dict() for b in self.legacy_builds],

            # Streak
            "current_streak":   self.current_streak,
            "longest_streak":   self.longest_streak,
            "streak_multiplier": streak_mult,

            # Shields
            "shields_remaining": self.shields_remaining,

            # Daily XP
            "daily_xp_today":   self.daily_xp_today,
            "daily_xp_cap":     cap,
            "daily_xp_remaining": max(0, cap - self.daily_xp_today),
            "daily_cap_reached": self.daily_xp_today >= cap,

            # Recent activity (last 5 logs)
            "recent_logs": [log.to_dict() for log in (recent_logs or [])[-5:]],

            # Gamer-exclusive fields (only meaningful when primary_build == gamer)
            "gamer_sessions_today":    self.sessions_today,
            "gamer_overtime_xp_today": self.overtime_xp_today,
            "gamer_overtime_remaining": max(0, GAMER_OVERTIME_LIMIT - self.overtime_xp_today),
            "gamer_is_boss_day":       gamer_is_boss_day(self.current_streak),
        }

    def to_json(self) -> str:
        """Serialize state to JSON string for database storage."""
        data = {
            "user_id":           self.user_id,
            "username":          self.username,
            "primary_build":     self.primary_build.to_dict(),
            "secondary_build":   self.secondary_build.to_dict() if self.secondary_build else None,
            "legacy_builds":     [b.to_dict() for b in self.legacy_builds],
            "current_streak":    self.current_streak,
            "longest_streak":    self.longest_streak,
            "last_activity_date": self.last_activity_date,
            "daily_xp_today":    self.daily_xp_today,
            "last_reset_date":   self.last_reset_date,
            "shields_remaining": self.shields_remaining,
            "shields_reset_month": self.shields_reset_month,
            "sessions_today":    self.sessions_today,
            "overtime_xp_today": self.overtime_xp_today,
        }
        return json.dumps(data, indent=2)

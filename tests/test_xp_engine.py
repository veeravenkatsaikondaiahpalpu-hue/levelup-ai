"""
Tests for the XP Engine.
Run with: python -m pytest tests/ -v
"""

import pytest
from datetime import date, timedelta

from xp_engine import (
    BuildType, Intensity, calculate_xp, get_streak_multiplier,
    get_daily_cap, calculate_level, is_primary_activity,
    get_xp_relevance_multiplier, get_badge_index_for_xp,
    BuildProfile, UserState, ActivityLog,
    calculate_gamer_xp, get_combo_multiplier, is_boss_day,
    get_gamer_daily_cap, qualifies_for_speedrun_bonus,
    get_gamer_relevance_multiplier,
    GAMER_OVERTIME_LIMIT, GAMER_BOSS_DAY_INTERVAL,
    GAMER_SPEEDRUN_BONUS_XP, GAMER_SPEEDRUN_THRESHOLD_MINS,
    GAMER_SIDE_QUEST_MULTIPLIER,
)


# ── calculate_level ───────────────────────────────────────────────────────────

class TestCalculateLevel:
    def test_level_1_at_zero_xp(self):
        assert calculate_level(0) == 1

    def test_level_2_at_500_xp(self):
        assert calculate_level(500) == 2

    def test_level_10_at_4999_xp(self):
        assert calculate_level(4_999) == 10

    def test_tier2_starts_at_5000(self):
        assert calculate_level(5_000) == 10

    def test_level_20_at_19999(self):
        assert calculate_level(19_999) == 19

    def test_level_20_at_20000(self):
        assert calculate_level(20_000) == 20

    def test_level_21_at_23000(self):
        assert calculate_level(23_000) == 21


# ── get_streak_multiplier ─────────────────────────────────────────────────────

class TestStreakMultiplier:
    def test_day_1_is_1x(self):
        mult, name = get_streak_multiplier(1)
        assert mult == 1.0
        assert name is None

    def test_day_4_is_1x(self):
        mult, _ = get_streak_multiplier(4)
        assert mult == 1.0

    def test_day_5_hits_powerup(self):
        mult, name = get_streak_multiplier(5)
        assert mult == 1.5
        assert name == "Power-Up"

    def test_day_6_is_still_1_5x_no_announcement(self):
        mult, name = get_streak_multiplier(6)
        assert mult == 1.5
        assert name is None  # announcement only on exact threshold day

    def test_day_10_hits_charged(self):
        mult, name = get_streak_multiplier(10)
        assert mult == 1.75
        assert name == "Charged"

    def test_day_20_hits_blazing(self):
        mult, name = get_streak_multiplier(20)
        assert mult == 2.0

    def test_day_30_hits_unstoppable(self):
        mult, name = get_streak_multiplier(30)
        assert mult == 2.25

    def test_day_60_hits_legendary(self):
        mult, name = get_streak_multiplier(60)
        assert mult == 2.5


# ── get_daily_cap ─────────────────────────────────────────────────────────────

class TestDailyCap:
    def test_level_1_cap_is_600(self):
        assert get_daily_cap(1) == 600

    def test_level_10_cap_is_600(self):
        assert get_daily_cap(10) == 600

    def test_level_11_cap_is_800(self):
        assert get_daily_cap(11) == 800

    def test_level_20_cap_is_800(self):
        assert get_daily_cap(20) == 800

    def test_level_21_cap_is_1000(self):
        assert get_daily_cap(21) == 1_000


# ── is_primary_activity ───────────────────────────────────────────────────────

class TestPrimaryActivity:
    def test_gym_is_primary_for_strength(self):
        assert is_primary_activity("gym_session", BuildType.STRENGTH) is True

    def test_gym_is_not_primary_for_intelligence(self):
        assert is_primary_activity("gym_session", BuildType.INTELLIGENCE) is False

    def test_study_is_primary_for_intelligence(self):
        assert is_primary_activity("study_session", BuildType.INTELLIGENCE) is True

    def test_meditation_is_primary_for_wellness(self):
        assert is_primary_activity("meditation", BuildType.WELLNESS) is True

    def test_networking_is_primary_for_entrepreneur(self):
        assert is_primary_activity("networking", BuildType.ENTREPRENEUR) is True

    def test_unknown_activity_is_not_primary(self):
        assert is_primary_activity("unknown_activity_xyz", BuildType.STRENGTH) is False

    def test_relevance_multiplier_primary_is_1(self):
        assert get_xp_relevance_multiplier("gym_session", BuildType.STRENGTH) == 1.0

    def test_relevance_multiplier_non_primary_is_0_5(self):
        assert get_xp_relevance_multiplier("gym_session", BuildType.INTELLIGENCE) == 0.5


# ── calculate_xp ─────────────────────────────────────────────────────────────

class TestCalculateXP:
    def test_basic_primary_activity(self):
        """60 min intense gym session, Strength build, Day 1, Level 1"""
        result = calculate_xp(
            activity_type    = "gym_session",
            duration_minutes = 60,
            intensity        = Intensity.INTENSE,
            primary_build    = BuildType.STRENGTH,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 0,
        )
        # 60 * 2.0 * 1.0 * 1.0 = 120
        assert result.final_xp == 120
        assert result.is_primary is True
        assert result.was_capped is False

    def test_non_primary_activity_gets_half_xp(self):
        """Same session but Intelligence build — gym is non-primary"""
        result = calculate_xp(
            activity_type    = "gym_session",
            duration_minutes = 60,
            intensity        = Intensity.INTENSE,
            primary_build    = BuildType.INTELLIGENCE,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 0,
        )
        # 60 * 2.0 * 0.5 * 1.0 = 60
        assert result.final_xp == 60
        assert result.is_primary is False

    def test_streak_multiplier_applied(self):
        """Day 5 streak = 1.5x multiplier"""
        result = calculate_xp(
            activity_type    = "gym_session",
            duration_minutes = 60,
            intensity        = Intensity.INTENSE,
            primary_build    = BuildType.STRENGTH,
            current_streak   = 5,
            current_level    = 1,
            daily_xp_so_far  = 0,
        )
        # 60 * 2.0 * 1.0 * 1.5 = 180
        assert result.final_xp == 180
        assert result.streak_multiplier == 1.5
        assert result.power_up_name == "Power-Up"

    def test_daily_cap_enforced(self):
        """Already at 560 XP today with cap 600 — only 40 XP available"""
        result = calculate_xp(
            activity_type    = "gym_session",
            duration_minutes = 60,
            intensity        = Intensity.INTENSE,
            primary_build    = BuildType.STRENGTH,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 560,
        )
        assert result.final_xp == 40
        assert result.was_capped is True
        assert result.daily_xp_remaining == 0

    def test_zero_xp_when_cap_already_reached(self):
        """Daily cap already at 600 — earns 0 XP"""
        result = calculate_xp(
            activity_type    = "gym_session",
            duration_minutes = 60,
            intensity        = Intensity.INTENSE,
            primary_build    = BuildType.STRENGTH,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 600,
        )
        assert result.final_xp == 0

    def test_light_intensity_multiplier(self):
        """30 min light walk, Wellness build"""
        result = calculate_xp(
            activity_type    = "walking",
            duration_minutes = 30,
            intensity        = Intensity.LIGHT,
            primary_build    = BuildType.WELLNESS,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 0,
        )
        # 30 * 1.0 * 0.5 * 1.0 = 15 (walking is not primary for wellness, score=0.6)
        assert result.final_xp == 15

    def test_moderate_intensity(self):
        """60 min moderate study session, Intelligence build"""
        result = calculate_xp(
            activity_type    = "study_session",
            duration_minutes = 60,
            intensity        = Intensity.MODERATE,
            primary_build    = BuildType.INTELLIGENCE,
            current_streak   = 1,
            current_level    = 1,
            daily_xp_so_far  = 0,
        )
        # 60 * 1.5 * 1.0 * 1.0 = 90
        assert result.final_xp == 90

    def test_summary_output_is_string(self):
        result = calculate_xp("gym_session", 60, Intensity.INTENSE,
                               BuildType.STRENGTH, 1, 1, 0)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "gym_session" in summary


# ── get_badge_index_for_xp ────────────────────────────────────────────────────

class TestBadgeIndex:
    def test_no_badge_below_1000(self):
        assert get_badge_index_for_xp(999) == -1

    def test_badge_0_at_1000(self):
        assert get_badge_index_for_xp(1_000) == 0

    def test_badge_1_at_5000(self):
        assert get_badge_index_for_xp(5_000) == 1

    def test_badge_5_at_150000(self):
        assert get_badge_index_for_xp(150_000) == 5


# ── BuildProfile ──────────────────────────────────────────────────────────────

class TestBuildProfile:
    def test_starts_at_level_1(self):
        profile = BuildProfile(BuildType.STRENGTH)
        assert profile.current_level == 1

    def test_badge_earned_on_xp_threshold(self):
        profile = BuildProfile(BuildType.STRENGTH, total_xp=999)
        new_badge = profile.add_xp(1)
        assert new_badge == "Iron"

    def test_legendary_flag(self):
        profile = BuildProfile(BuildType.STRENGTH, total_xp=150_000)
        assert profile.is_legendary is True

    def test_not_legendary_below_threshold(self):
        profile = BuildProfile(BuildType.STRENGTH, total_xp=149_999)
        assert profile.is_legendary is False

    def test_xp_to_next_badge(self):
        profile = BuildProfile(BuildType.STRENGTH, total_xp=0)
        assert profile.xp_to_next_badge == 1_000


# ── UserState ─────────────────────────────────────────────────────────────────

class TestUserState:
    def _make_user(self, build=BuildType.STRENGTH, xp=0):
        return UserState(
            user_id       = "test_user",
            username      = "Veera",
            primary_build = BuildProfile(build, total_xp=xp),
        )

    def test_streak_increments_on_consecutive_day(self):
        user = self._make_user()
        yesterday = date.today() - timedelta(days=1)
        user.last_activity_date = yesterday.isoformat()
        user.current_streak = 3

        result = user.update_streak(date.today())
        assert user.current_streak == 4
        assert result["streak_broken"] is False

    def test_streak_resets_after_two_missed_days_no_shield(self):
        user = self._make_user()
        user.shields_remaining = 0
        three_days_ago = date.today() - timedelta(days=3)
        user.last_activity_date = three_days_ago.isoformat()
        user.current_streak = 10

        result = user.update_streak(date.today())
        assert user.current_streak == 1
        assert result["streak_broken"] is True

    def test_shield_consumed_for_one_missed_day(self):
        user = self._make_user()
        user.shields_remaining = 3
        two_days_ago = date.today() - timedelta(days=2)
        user.last_activity_date = two_days_ago.isoformat()
        user.current_streak = 5

        result = user.update_streak(date.today())
        assert result["shield_used"] is True
        assert user.shields_remaining == 2
        assert user.current_streak == 6

    def test_daily_xp_cap_respected_in_add_xp(self):
        user = self._make_user()
        user.daily_xp_today = 550
        result = user.add_xp(200)
        assert result["xp_added"] == 50
        assert result["cap_reached"] is True

    def test_cannot_switch_build_without_legendary(self):
        user = self._make_user(xp=10_000)
        result = user.archive_primary_and_start_new(BuildType.INTELLIGENCE)
        assert result["success"] is False

    def test_can_switch_build_after_legendary(self):
        user = self._make_user(xp=150_000)
        result = user.archive_primary_and_start_new(BuildType.INTELLIGENCE)
        assert result["success"] is True
        assert user.primary_build.build_type == BuildType.INTELLIGENCE
        assert len(user.legacy_builds) == 1
        assert user.primary_build.total_xp == 500  # legacy bonus

    def test_context_dict_has_required_keys(self):
        user = self._make_user()
        ctx = user.to_context_dict()
        required = [
            "primary_build", "primary_badge", "primary_level", "primary_xp_total",
            "current_streak", "streak_multiplier", "shields_remaining",
            "daily_xp_today", "daily_xp_cap", "daily_xp_remaining",
        ]
        for key in required:
            assert key in ctx, f"Missing key: {key}"

    def test_context_dict_has_gamer_keys(self):
        user = self._make_user()
        ctx = user.to_context_dict()
        gamer_keys = [
            "gamer_sessions_today", "gamer_overtime_xp_today",
            "gamer_overtime_remaining", "gamer_is_boss_day",
        ]
        for key in gamer_keys:
            assert key in ctx, f"Missing gamer key: {key}"


# ── Gamer Mechanics ───────────────────────────────────────────────────────────

class TestGamerComboMultiplier:
    def test_first_session_no_combo(self):
        assert get_combo_multiplier(0) == 0.0

    def test_second_session_is_0_2x(self):
        assert get_combo_multiplier(1) == 0.20

    def test_third_session_is_0_3x(self):
        assert get_combo_multiplier(2) == 0.30

    def test_fourth_session_is_0_4x(self):
        assert get_combo_multiplier(3) == 0.40

    def test_fifth_session_is_still_0_4x(self):
        assert get_combo_multiplier(4) == 0.40


class TestGamerBossDay:
    def test_streak_7_is_boss_day(self):
        assert is_boss_day(7) is True

    def test_streak_14_is_boss_day(self):
        assert is_boss_day(14) is True

    def test_streak_6_is_not_boss_day(self):
        assert is_boss_day(6) is False

    def test_streak_0_is_not_boss_day(self):
        assert is_boss_day(0) is False

    def test_streak_21_is_boss_day(self):
        assert is_boss_day(21) is True


class TestGamerDailyCap:
    def test_normal_day_cap_unchanged(self):
        assert get_gamer_daily_cap(600, 5) == 600

    def test_boss_day_doubles_cap(self):
        assert get_gamer_daily_cap(600, 7) == 1200

    def test_boss_day_doubles_higher_cap(self):
        assert get_gamer_daily_cap(800, 14) == 1600


class TestSpeedrunBonus:
    def test_qualifies_intense_short(self):
        assert qualifies_for_speedrun_bonus(
            "competitive_gaming", GAMER_SPEEDRUN_THRESHOLD_MINS, Intensity.INTENSE
        ) is True

    def test_too_long_no_bonus(self):
        assert qualifies_for_speedrun_bonus(
            "competitive_gaming", GAMER_SPEEDRUN_THRESHOLD_MINS + 1, Intensity.INTENSE
        ) is False

    def test_moderate_no_bonus(self):
        assert qualifies_for_speedrun_bonus(
            "competitive_gaming", 20, Intensity.MODERATE
        ) is False

    def test_light_no_bonus(self):
        assert qualifies_for_speedrun_bonus(
            "competitive_gaming", 10, Intensity.LIGHT
        ) is False


class TestGamerRelevanceMultiplier:
    def test_primary_gamer_activity_is_1x(self):
        assert get_gamer_relevance_multiplier("competitive_gaming") == 1.0

    def test_non_gamer_activity_is_side_quest(self):
        mult = get_gamer_relevance_multiplier("gym_session")
        assert mult == GAMER_SIDE_QUEST_MULTIPLIER
        assert mult == 0.40

    def test_side_quest_stricter_than_default(self):
        # 0.4x is stricter than standard 0.5x non-primary
        assert get_gamer_relevance_multiplier("study_session") < 0.5


class TestCalculateGamerXP:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            activity_type          = "competitive_gaming",
            duration_minutes       = 60,
            intensity              = Intensity.INTENSE,
            current_streak         = 1,
            current_level          = 1,
            daily_xp_so_far        = 0,
            primary_sessions_today = 0,
            overtime_xp_today      = 0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_first_session_no_bonuses(self):
        """Day 1, session 1, no cap — base XP only"""
        r = calculate_gamer_xp(**self._base_kwargs())
        # base: 60 * 2.0 * 1.0 * 1.0 = 120
        assert r.base_final_xp == 120
        assert r.combo_bonus_xp == 0
        assert r.speedrun_bonus == 0
        assert r.overtime_xp == 0
        assert r.final_xp == 120

    def test_combo_bonus_on_second_session(self):
        """Second primary session today → +0.2x combo"""
        r = calculate_gamer_xp(**self._base_kwargs(primary_sessions_today=1))
        # base 120, combo = 120 * 0.2 = 24
        assert r.combo_multiplier == 0.20
        assert r.combo_bonus_xp == 24
        assert r.final_xp == 144

    def test_combo_bonus_on_fourth_session(self):
        """Fourth session → +0.4x combo"""
        r = calculate_gamer_xp(**self._base_kwargs(primary_sessions_today=3))
        assert r.combo_multiplier == 0.40
        assert r.combo_bonus_xp == 48
        assert r.final_xp == 168

    def test_speedrun_bonus_applied(self):
        """Intense session ≤25 min gets +15 XP"""
        r = calculate_gamer_xp(**self._base_kwargs(
            duration_minutes=GAMER_SPEEDRUN_THRESHOLD_MINS,
            intensity=Intensity.INTENSE,
        ))
        assert r.speedrun_bonus == GAMER_SPEEDRUN_BONUS_XP
        assert GAMER_SPEEDRUN_BONUS_XP in [r.final_xp - r.base_final_xp,
                                            r.final_xp - r.base_final_xp - r.combo_bonus_xp]

    def test_no_speedrun_bonus_for_moderate(self):
        r = calculate_gamer_xp(**self._base_kwargs(
            duration_minutes=20,
            intensity=Intensity.MODERATE,
        ))
        assert r.speedrun_bonus == 0

    def test_overtime_xp_after_cap(self):
        """Already at cap → Overtime pool absorbs overflow"""
        r = calculate_gamer_xp(**self._base_kwargs(daily_xp_so_far=600))
        # Base earn = 0 (cap hit), overtime = min(120, 150) = 120
        assert r.base_final_xp == 0
        assert r.overtime_xp > 0
        assert r.overtime_xp <= GAMER_OVERTIME_LIMIT

    def test_overtime_pool_depletes(self):
        """Overtime already partly spent — can't exceed remaining pool"""
        r = calculate_gamer_xp(**self._base_kwargs(
            daily_xp_so_far=600,
            overtime_xp_today=100,
        ))
        assert r.overtime_xp <= 50   # 150 - 100 = 50 remaining
        assert r.overtime_pool_remaining == max(0, 50 - r.overtime_xp)

    def test_boss_day_doubles_cap(self):
        """Streak=7 → Boss Day → cap 1200 instead of 600.
        Streak 7 also activates the 1.5x Power-Up streak bonus (day 5+),
        so raw XP = 60 * 2.0 * 1.0 * 1.5 = 180 (well within 1200 cap)."""
        r = calculate_gamer_xp(**self._base_kwargs(
            current_streak=7,
            daily_xp_so_far=0,
        ))
        assert r.is_boss_day is True
        # 60 * 2.0 * 1.0 * 1.5 = 180 (streak 1.5x at day 7, cap not exceeded)
        assert r.base_final_xp == 180

    def test_side_quest_penalty_applied(self):
        """Non-Gamer activity → 0.4x penalty"""
        r = calculate_gamer_xp(**self._base_kwargs(activity_type="gym_session"))
        assert r.side_quest_penalty is True
        # base: 60 * 2.0 * 0.4 * 1.0 = 48
        assert r.base_final_xp == 48

    def test_side_quest_no_combo_bonus(self):
        """Combo multiplier does NOT apply to side quest (non-primary) sessions"""
        r = calculate_gamer_xp(**self._base_kwargs(
            activity_type="gym_session",
            primary_sessions_today=2,
        ))
        assert r.combo_bonus_xp == 0

    def test_summary_is_string(self):
        r = calculate_gamer_xp(**self._base_kwargs())
        assert isinstance(r.summary(), str)
        assert "GAMER SESSION" in r.summary()


class TestGamerUserState:
    def _make_gamer(self):
        return UserState(
            user_id       = "test_gamer",
            username      = "Veera",
            primary_build = BuildProfile(BuildType.GAMER, total_xp=0),
        )

    def test_gamer_fields_initialized(self):
        user = self._make_gamer()
        assert user.sessions_today == 0
        assert user.overtime_xp_today == 0

    def test_gamer_fields_in_context(self):
        user = self._make_gamer()
        ctx = user.to_context_dict()
        assert "gamer_sessions_today" in ctx
        assert "gamer_overtime_remaining" in ctx
        assert ctx["gamer_overtime_remaining"] == GAMER_OVERTIME_LIMIT

    def test_boss_day_flag_in_context(self):
        user = self._make_gamer()
        user.current_streak = GAMER_BOSS_DAY_INTERVAL
        ctx = user.to_context_dict()
        assert ctx["gamer_is_boss_day"] is True

    def test_boss_day_doubles_daily_cap_in_add_xp(self):
        user = self._make_gamer()
        user.current_streak = GAMER_BOSS_DAY_INTERVAL  # Boss Day
        user.daily_xp_today = 550
        # Standard cap 600 → Boss cap 1200 → 650 XP still available
        result = user.add_xp(200)
        assert result["xp_added"] == 200   # not capped at 600
        assert result["daily_cap"] == 1200

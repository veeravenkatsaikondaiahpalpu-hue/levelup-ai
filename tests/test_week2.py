"""
Tests for Week 2 modules: system_prompt and anomaly data generator.
Run with: python -m pytest tests/test_week2.py -v
"""

import sys
import os
import csv
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.system_prompt import (
    build_system_prompt, detect_mode,
    BUILD_PERSONALITY,
)
from xp_engine import BuildType, UserState, BuildProfile
from data.generate_anomaly_data import (
    gen_normal, gen_anomaly_xp_grinding,
    gen_anomaly_impossible_streak, gen_anomaly_intensity_spoofing,
    generate_dataset, LogRecord,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ctx(build="strength", streak=1, daily_xp=0, level=1, xp=0):
    """Build a minimal context dict for system prompt tests."""
    return {
        "primary_build":        build,
        "primary_badge":        "Iron",
        "primary_level":        level,
        "primary_xp_total":     xp,
        "xp_to_next_badge":     1000,
        "next_badge":           "Bronze Warrior",
        "xp_to_next_level":     500,
        "current_streak":       streak,
        "longest_streak":       streak,
        "streak_multiplier":    1.0,
        "shields_remaining":    4,
        "daily_xp_today":       daily_xp,
        "daily_xp_cap":         600,
        "daily_xp_remaining":   600 - daily_xp,
        "daily_cap_reached":    daily_xp >= 600,
        "can_unlock_build":     xp >= 150_000,
        "gamer_sessions_today":     0,
        "gamer_overtime_xp_today":  0,
        "gamer_overtime_remaining": 150,
        "gamer_is_boss_day":        False,
        "recent_logs":          [],
    }


# ── system_prompt: build_system_prompt ───────────────────────────────────────

class TestBuildSystemPrompt:

    def test_returns_string(self):
        ctx = make_ctx()
        result = build_system_prompt(ctx)
        assert isinstance(result, str)

    def test_contains_build_name(self):
        ctx = make_ctx(build="strength")
        prompt = build_system_prompt(ctx)
        assert "STRENGTH" in prompt

    def test_contains_persona_name(self):
        ctx = make_ctx(build="strength")
        prompt = build_system_prompt(ctx)
        assert "TITAN" in prompt

    def test_gamer_persona_name(self):
        ctx = make_ctx(build="gamer")
        prompt = build_system_prompt(ctx)
        assert "GG" in prompt

    def test_contains_user_stats(self):
        ctx = make_ctx(build="intelligence", streak=10, level=5)
        prompt = build_system_prompt(ctx)
        assert "10 days" in prompt
        assert "Level" in prompt

    def test_gamer_extras_shown_for_gamer(self):
        ctx = make_ctx(build="gamer")
        ctx["gamer_is_boss_day"] = True
        prompt = build_system_prompt(ctx)
        assert "GAMER MECHANICS" in prompt
        assert "DOUBLED" in prompt

    def test_gamer_extras_hidden_for_other_builds(self):
        ctx = make_ctx(build="strength")
        prompt = build_system_prompt(ctx)
        assert "GAMER MECHANICS" not in prompt

    def test_struggling_sentiment_adds_alert(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx, sentiment="struggling")
        assert "STRUGGLING" in prompt

    def test_motivated_sentiment_adds_energy(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx, sentiment="motivated")
        assert "MOTIVATED" in prompt

    def test_xp_coach_mode_shows_formula(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx, mode="xp_coach")
        assert "Duration x Intensity x Relevance x Streak" in prompt

    def test_build_advisor_mode_mentions_legendary(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx, mode="build_advisor")
        assert "Legendary" in prompt or "legendary" in prompt.lower()

    def test_motivation_mode_is_present(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx, mode="motivation")
        assert "MOTIVATION" in prompt

    def test_rules_always_present(self):
        ctx = make_ctx()
        prompt = build_system_prompt(ctx)
        assert "Never claim to be ChatGPT" in prompt
        assert "Never give medical diagnoses" in prompt

    def test_recent_logs_shown(self):
        ctx = make_ctx()
        ctx["recent_logs"] = [
            {"activity": "gym_session", "duration": 60,
             "intensity": "intense", "xp": 120, "logged_at": "2026-04-19T10:00:00"}
        ]
        prompt = build_system_prompt(ctx)
        assert "gym_session" in prompt

    def test_all_builds_have_unique_persona_names(self):
        names = [p["name"] for p in BUILD_PERSONALITY.values()]
        assert len(names) == len(set(names)), "Duplicate persona names found"

    def test_legendary_unlocked_message_shown(self):
        ctx = make_ctx(xp=150_000)
        ctx["can_unlock_build"] = True
        prompt = build_system_prompt(ctx)
        assert "Legendary" in prompt or "YES" in prompt


# ── system_prompt: detect_mode ────────────────────────────────────────────────

class TestDetectMode:

    def test_xp_question_detected(self):
        assert detect_mode("how much xp do i earn?") == "xp_coach"

    def test_daily_cap_is_xp_coach(self):
        assert detect_mode("have i hit my daily cap?") == "xp_coach"

    def test_build_question_detected(self):
        assert detect_mode("can i switch my build?") == "build_advisor"

    def test_legendary_is_build_advisor(self):
        assert detect_mode("when do i reach legendary?") == "build_advisor"

    def test_struggling_sentiment_overrides(self):
        assert detect_mode("what should I do today", sentiment="struggling") == "motivation"

    def test_tired_keyword_detected(self):
        assert detect_mode("I'm so tired, I want to quit") == "motivation"

    def test_give_up_detected(self):
        assert detect_mode("I feel like giving up") == "motivation"

    def test_general_fallback(self):
        assert detect_mode("hey whats up") == "general"

    def test_case_insensitive(self):
        assert detect_mode("HOW MUCH XP?") == "xp_coach"


# ── Anomaly Data Generator ────────────────────────────────────────────────────

class TestNormalGenerator:

    def test_returns_log_record(self):
        r = gen_normal()
        assert isinstance(r, LogRecord)

    def test_is_not_anomaly(self):
        for _ in range(50):
            r = gen_normal()
            assert r.is_anomaly == 0
            assert r.anomaly_type == "none"

    def test_reasonable_session_count(self):
        counts = [gen_normal().activities_per_day for _ in range(200)]
        assert all(1 <= c <= 4 for c in counts)

    def test_reasonable_duration(self):
        for _ in range(50):
            r = gen_normal()
            assert 15 <= r.avg_session_duration <= 120

    def test_xp_within_cap(self):
        for _ in range(100):
            r = gen_normal()
            assert r.daily_xp_total <= 600


class TestXPGrindingGenerator:

    def test_is_anomaly(self):
        for _ in range(20):
            r = gen_anomaly_xp_grinding()
            assert r.is_anomaly == 1
            assert r.anomaly_type == "xp_grinding"

    def test_high_session_count_or_long_duration(self):
        for _ in range(100):
            r = gen_anomaly_xp_grinding()
            abnormal = r.activities_per_day >= 8 or r.max_session_duration > 300
            assert abnormal, f"Expected abnormal session: {r}"

    def test_always_at_cap(self):
        for _ in range(20):
            r = gen_anomaly_xp_grinding()
            assert r.sessions_at_cap_ratio == 1.0


class TestImpossibleStreakGenerator:

    def test_is_anomaly(self):
        for _ in range(20):
            r = gen_anomaly_impossible_streak()
            assert r.is_anomaly == 1
            assert r.anomaly_type == "impossible_streak"

    def test_large_streak_gap(self):
        for _ in range(50):
            r = gen_anomaly_impossible_streak()
            assert r.streak_gap_days >= 3, f"Expected gap >= 3, got {r.streak_gap_days}"

    def test_short_sessions(self):
        for _ in range(50):
            r = gen_anomaly_impossible_streak()
            assert r.avg_session_duration <= 20


class TestIntensitySpoofingGenerator:

    def test_is_anomaly(self):
        for _ in range(20):
            r = gen_anomaly_intensity_spoofing()
            assert r.is_anomaly == 1
            assert r.anomaly_type == "intensity_spoofing"

    def test_max_switch_rate(self):
        for _ in range(20):
            r = gen_anomaly_intensity_spoofing()
            assert r.intensity_switch_rate == 1.0


class TestGenerateDataset:

    def test_generates_correct_totals(self, tmp_path):
        full, train, test = generate_dataset(
            n_total=100, anomaly_ratio=0.20,
            output_dir=str(tmp_path)
        )
        with open(full) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 100

    def test_correct_anomaly_ratio(self, tmp_path):
        full, _, _ = generate_dataset(
            n_total=1000, anomaly_ratio=0.20,
            output_dir=str(tmp_path)
        )
        with open(full) as f:
            rows = list(csv.DictReader(f))
        n_anomaly = sum(1 for r in rows if r["is_anomaly"] == "1")
        ratio = n_anomaly / len(rows)
        assert abs(ratio - 0.20) < 0.02, f"Anomaly ratio {ratio:.2f} out of range"

    def test_train_test_split_sizes(self, tmp_path):
        _, train, test = generate_dataset(
            n_total=100, anomaly_ratio=0.20,
            output_dir=str(tmp_path)
        )
        with open(train) as f:
            n_train = sum(1 for _ in f) - 1   # minus header
        with open(test) as f:
            n_test = sum(1 for _ in f) - 1
        assert n_train + n_test == 100

    def test_all_required_columns_present(self, tmp_path):
        full, _, _ = generate_dataset(
            n_total=50, anomaly_ratio=0.20,
            output_dir=str(tmp_path)
        )
        with open(full) as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames)
        required = {
            "activities_per_day", "daily_xp_total", "streak_gap_days",
            "intensity_switch_rate", "avg_session_duration", "max_session_duration",
            "sessions_at_cap_ratio", "xp_per_minute", "is_anomaly", "anomaly_type"
        }
        assert required.issubset(cols)

    def test_three_anomaly_types_present(self, tmp_path):
        full, _, _ = generate_dataset(
            n_total=300, anomaly_ratio=0.30,
            output_dir=str(tmp_path)
        )
        with open(full) as f:
            rows = list(csv.DictReader(f))
        types = {r["anomaly_type"] for r in rows if r["is_anomaly"] == "1"}
        assert "xp_grinding" in types
        assert "impossible_streak" in types
        assert "intensity_spoofing" in types

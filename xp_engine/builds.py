"""
builds.py - Build types, badge tiers, and activity relevance scoring.

Every build has:
  - A badge ladder (6 tiers, Tier 6 = Legendary = unlocks new primary build)
  - A relevance score for each activity type (used by xp_calculator)
"""

from enum import Enum
from typing import Dict, Optional


# ── Build Types ───────────────────────────────────────────────────────────────

class BuildType(Enum):
    STRENGTH     = "strength"
    INTELLIGENCE = "intelligence"
    DEXTERITY    = "dexterity"
    WELLNESS     = "wellness"
    CREATIVE     = "creative"
    ENTREPRENEUR = "entrepreneur"


# ── Intensity Levels ──────────────────────────────────────────────────────────

class Intensity(Enum):
    LIGHT    = "light"
    MODERATE = "moderate"
    INTENSE  = "intense"

INTENSITY_MULTIPLIERS: Dict[Intensity, float] = {
    Intensity.LIGHT:    1.0,
    Intensity.MODERATE: 1.5,
    Intensity.INTENSE:  2.0,
}


# ── Badge Tiers ───────────────────────────────────────────────────────────────
# 6 tiers per build. XP thresholds are shared across all builds.
# Reaching Tier 6 (index 5) = Legendary = unlocks a new primary build slot.

BADGE_XP_THRESHOLDS = [1_000, 5_000, 15_000, 35_000, 75_000, 150_000]
LEGENDARY_BADGE_INDEX = 5

BADGE_TIERS: Dict[BuildType, list] = {
    BuildType.STRENGTH:     ["Iron",     "Bronze Warrior",  "Silver Gladiator", "Gold Champion", "Platinum Titan",  "Legendary Titan"],
    BuildType.INTELLIGENCE: ["Curious",  "Apprentice",      "Scholar",          "Sage",          "Mastermind",      "Oracle"],
    BuildType.DEXTERITY:    ["Nimble",   "Swift",           "Agile",            "Razor",         "Phantom",         "Ghost"],
    BuildType.WELLNESS:     ["Balanced", "Grounded",        "Centered",         "Resilient",     "Transcendent",    "Ascendant"],
    BuildType.CREATIVE:     ["Dabbler",  "Artisan",         "Creator",          "Visionary",     "Icon",            "Legend"],
    BuildType.ENTREPRENEUR: ["Hustler",  "Builder",         "Founder",          "Operator",      "Mogul",           "Empire"],
}

BUILD_DISPLAY_NAMES: Dict[BuildType, str] = {
    BuildType.STRENGTH:     "Strength",
    BuildType.INTELLIGENCE: "Intelligence",
    BuildType.DEXTERITY:    "Dexterity",
    BuildType.WELLNESS:     "Wellness",
    BuildType.CREATIVE:     "Creative",
    BuildType.ENTREPRENEUR: "Entrepreneur",
}

BUILD_DESCRIPTIONS: Dict[BuildType, str] = {
    BuildType.STRENGTH:     "Physical fitness, gym, sports, endurance",
    BuildType.INTELLIGENCE: "Study, reading, research, online courses",
    BuildType.DEXTERITY:    "Skills, instruments, martial arts, coding precision",
    BuildType.WELLNESS:     "Mental health, sleep, nutrition, meditation",
    BuildType.CREATIVE:     "Art, music, writing, design, photography",
    BuildType.ENTREPRENEUR: "Business, networking, productivity, leadership",
}


# ── Activity Relevance Table ──────────────────────────────────────────────────
# score >= 0.8 -> primary activity for that build -> 1.0x XP multiplier
# score <  0.8 -> non-primary                     -> 0.5x XP multiplier

BUILD_RELEVANCE: Dict[str, Dict[BuildType, float]] = {
    # Physical
    "gym_session":         {BuildType.STRENGTH: 1.0, BuildType.INTELLIGENCE: 0.15, BuildType.DEXTERITY: 0.4,  BuildType.WELLNESS: 0.3,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.1 },
    "running":             {BuildType.STRENGTH: 0.9, BuildType.INTELLIGENCE: 0.1,  BuildType.DEXTERITY: 0.6,  BuildType.WELLNESS: 0.5,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.1 },
    "walking":             {BuildType.STRENGTH: 0.4, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 0.3,  BuildType.WELLNESS: 0.6,  BuildType.CREATIVE: 0.2,  BuildType.ENTREPRENEUR: 0.15},
    "sports":              {BuildType.STRENGTH: 0.8, BuildType.INTELLIGENCE: 0.15, BuildType.DEXTERITY: 0.9,  BuildType.WELLNESS: 0.4,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.1 },
    "martial_arts":        {BuildType.STRENGTH: 0.8, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 1.0,  BuildType.WELLNESS: 0.3,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.1 },
    "yoga":                {BuildType.STRENGTH: 0.4, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 0.6,  BuildType.WELLNESS: 0.9,  BuildType.CREATIVE: 0.2,  BuildType.ENTREPRENEUR: 0.1 },
    "swimming":            {BuildType.STRENGTH: 0.9, BuildType.INTELLIGENCE: 0.1,  BuildType.DEXTERITY: 0.7,  BuildType.WELLNESS: 0.5,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.1 },
    # Mental / Study
    "study_session":       {BuildType.STRENGTH: 0.2, BuildType.INTELLIGENCE: 1.0,  BuildType.DEXTERITY: 0.25, BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 0.3,  BuildType.ENTREPRENEUR: 0.6 },
    "reading":             {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.9,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 0.3,  BuildType.CREATIVE: 0.4,  BuildType.ENTREPRENEUR: 0.5 },
    "online_course":       {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 1.0,  BuildType.DEXTERITY: 0.3,  BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 0.35, BuildType.ENTREPRENEUR: 0.65},
    "research":            {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 1.0,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 0.1,  BuildType.CREATIVE: 0.3,  BuildType.ENTREPRENEUR: 0.5 },
    # Skills
    "instrument_practice": {BuildType.STRENGTH: 0.15,BuildType.INTELLIGENCE: 0.4,  BuildType.DEXTERITY: 1.0,  BuildType.WELLNESS: 0.25, BuildType.CREATIVE: 0.8,  BuildType.ENTREPRENEUR: 0.15},
    "coding_sprint":       {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.85, BuildType.DEXTERITY: 0.9,  BuildType.WELLNESS: 0.1,  BuildType.CREATIVE: 0.5,  BuildType.ENTREPRENEUR: 0.7 },
    "typing_practice":     {BuildType.STRENGTH: 0.05,BuildType.INTELLIGENCE: 0.3,  BuildType.DEXTERITY: 0.9,  BuildType.WELLNESS: 0.05, BuildType.CREATIVE: 0.2,  BuildType.ENTREPRENEUR: 0.3 },
    # Wellness
    "meditation":          {BuildType.STRENGTH: 0.2, BuildType.INTELLIGENCE: 0.3,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 1.0,  BuildType.CREATIVE: 0.3,  BuildType.ENTREPRENEUR: 0.25},
    "journaling":          {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.5,  BuildType.DEXTERITY: 0.15, BuildType.WELLNESS: 0.8,  BuildType.CREATIVE: 0.7,  BuildType.ENTREPRENEUR: 0.4 },
    "sleep_tracking":      {BuildType.STRENGTH: 0.3, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 1.0,  BuildType.CREATIVE: 0.15, BuildType.ENTREPRENEUR: 0.15},
    "nutrition_tracking":  {BuildType.STRENGTH: 0.5, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 0.3,  BuildType.WELLNESS: 0.9,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.15},
    "cold_shower":         {BuildType.STRENGTH: 0.5, BuildType.INTELLIGENCE: 0.2,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 0.8,  BuildType.CREATIVE: 0.1,  BuildType.ENTREPRENEUR: 0.2 },
    # Creative
    "art_session":         {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.35, BuildType.DEXTERITY: 0.7,  BuildType.WELLNESS: 0.3,  BuildType.CREATIVE: 1.0,  BuildType.ENTREPRENEUR: 0.2 },
    "writing":             {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.6,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 1.0,  BuildType.ENTREPRENEUR: 0.7 },
    "music_production":    {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.4,  BuildType.DEXTERITY: 0.8,  BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 1.0,  BuildType.ENTREPRENEUR: 0.2 },
    "photography":         {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.3,  BuildType.DEXTERITY: 0.6,  BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 0.9,  BuildType.ENTREPRENEUR: 0.3 },
    "design_work":         {BuildType.STRENGTH: 0.05,BuildType.INTELLIGENCE: 0.4,  BuildType.DEXTERITY: 0.6,  BuildType.WELLNESS: 0.1,  BuildType.CREATIVE: 1.0,  BuildType.ENTREPRENEUR: 0.4 },
    # Entrepreneur
    "networking":          {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.3,  BuildType.DEXTERITY: 0.15, BuildType.WELLNESS: 0.2,  BuildType.CREATIVE: 0.2,  BuildType.ENTREPRENEUR: 1.0 },
    "business_task":       {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.4,  BuildType.DEXTERITY: 0.2,  BuildType.WELLNESS: 0.1,  BuildType.CREATIVE: 0.25, BuildType.ENTREPRENEUR: 1.0 },
    "cold_outreach":       {BuildType.STRENGTH: 0.05,BuildType.INTELLIGENCE: 0.25, BuildType.DEXTERITY: 0.1,  BuildType.WELLNESS: 0.05, BuildType.CREATIVE: 0.15, BuildType.ENTREPRENEUR: 0.9 },
    "pitch_practice":      {BuildType.STRENGTH: 0.1, BuildType.INTELLIGENCE: 0.4,  BuildType.DEXTERITY: 0.3,  BuildType.WELLNESS: 0.1,  BuildType.CREATIVE: 0.4,  BuildType.ENTREPRENEUR: 0.9 },
}

PRIMARY_THRESHOLD = 0.8  # score >= this = primary activity for that build


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_relevance_score(activity_type: str, primary_build: BuildType) -> float:
    """Raw relevance score (0.0-1.0). Defaults to 0.5 for unknown activities."""
    if activity_type not in BUILD_RELEVANCE:
        return 0.5
    return BUILD_RELEVANCE[activity_type].get(primary_build, 0.5)


def is_primary_activity(activity_type: str, build: BuildType) -> bool:
    """True if this activity counts as primary for the given build."""
    return get_relevance_score(activity_type, build) >= PRIMARY_THRESHOLD


def get_xp_relevance_multiplier(activity_type: str, primary_build: BuildType) -> float:
    """1.0x for primary activities, 0.5x for everything else."""
    return 1.0 if is_primary_activity(activity_type, primary_build) else 0.5


def get_badge_index_for_xp(total_xp: int) -> int:
    """Highest badge tier index earned for a given XP total. -1 = none yet."""
    earned = -1
    for i, threshold in enumerate(BADGE_XP_THRESHOLDS):
        if total_xp >= threshold:
            earned = i
    return earned


def get_badge_name(build: BuildType, badge_index: int) -> Optional[str]:
    """Badge name at a tier index. None if not yet earned."""
    if badge_index < 0 or badge_index >= len(BADGE_TIERS[build]):
        return None
    return BADGE_TIERS[build][badge_index]


def calculate_level(total_xp: int) -> int:
    """
    Level progression:
      Level 1-10  :     0 -  4,999 XP  (500 XP per level)
      Level 10-20 : 5,000 - 19,999 XP  (1,500 XP per level)
      Level 20+   : 20,000+        XP  (3,000 XP per level)
    """
    if total_xp < 5_000:
        return max(1, total_xp // 500 + 1)
    elif total_xp < 20_000:
        return 10 + (total_xp - 5_000) // 1_500
    else:
        return 20 + (total_xp - 20_000) // 3_000

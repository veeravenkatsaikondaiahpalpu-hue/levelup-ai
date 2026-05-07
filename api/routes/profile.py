"""
api/routes/profile.py
======================
User profile, XP logging, and build management endpoints.

POST /api/profile                       -- create a new user profile
GET  /api/profile/{user_id}             -- get profile summary
GET  /api/profile/{user_id}/context     -- full context dict for LLM system prompt
POST /api/profile/{user_id}/log         -- log an activity → earn XP + streak update
GET  /api/profile/{user_id}/xp/preview  -- preview XP without logging
POST /api/profile/{user_id}/build/switch -- unlock secondary or archive+restart

Storage: JSON files in data/profiles/{user_id}.json  (no database needed for dev/exam)
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from xp_engine import (
    BuildType, Intensity,
    UserState, BuildProfile, ActivityLog as EngineActivityLog,
    calculate_xp, best_activity_for_xp,
    BUILD_DISPLAY_NAMES, BUILD_DESCRIPTIONS,
    BADGE_TIERS,
)

router = APIRouter(prefix="/api/profile", tags=["profile"])

# ── Storage path ──────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[2]
PROFILES_DIR = ROOT / "data" / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers: persistence ──────────────────────────────────────────────────────

def _profile_path(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return PROFILES_DIR / f"{safe}.json"


def _save(state: UserState, logs: list[dict]) -> None:
    data = json.loads(state.to_json())
    data["activity_history"] = logs
    _profile_path(state.user_id).write_text(json.dumps(data, indent=2))


def _load(user_id: str) -> tuple[UserState, list[dict]]:
    """Load UserState + activity history from disk. Raises 404 if not found."""
    path = _profile_path(user_id)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{user_id}' not found. Create it first via POST /api/profile",
        )
    data = json.loads(path.read_text())

    # Rebuild primary build
    pb_data = data["primary_build"]
    primary = BuildProfile(
        build_type=BuildType(pb_data["build"]),
        total_xp=pb_data["total_xp"],
        is_legacy=pb_data.get("is_legacy", False),
    )

    # Secondary build (optional)
    secondary = None
    if data.get("secondary_build"):
        sb = data["secondary_build"]
        secondary = BuildProfile(
            build_type=BuildType(sb["build"]),
            total_xp=sb["total_xp"],
            is_legacy=sb.get("is_legacy", False),
        )

    # Legacy builds
    legacy = [
        BuildProfile(
            build_type=BuildType(b["build"]),
            total_xp=b["total_xp"],
            is_legacy=True,
        )
        for b in data.get("legacy_builds", [])
    ]

    state = UserState(
        user_id         = data["user_id"],
        username        = data["username"],
        primary_build   = primary,
        secondary_build = secondary,
        legacy_builds   = legacy,
        current_streak  = data.get("current_streak", 0),
        longest_streak  = data.get("longest_streak", 0),
        last_activity_date = data.get("last_activity_date"),
        daily_xp_today  = data.get("daily_xp_today", 0),
        last_reset_date = data.get("last_reset_date", date.today().isoformat()),
        shields_remaining = data.get("shields_remaining", 4),
        shields_reset_month = data.get("shields_reset_month", date.today().month),
        sessions_today  = data.get("sessions_today", 0),
        overtime_xp_today = data.get("overtime_xp_today", 0),
    )

    logs = data.get("activity_history", [])
    return state, logs


# ── Request / Response schemas ────────────────────────────────────────────────

class CreateProfileRequest(BaseModel):
    user_id:  str = Field(..., min_length=3, max_length=32, description="Unique user ID")
    username: str = Field(..., min_length=1, max_length=50,  description="Display name")
    build:    str = Field(..., description="strength | intelligence | dexterity | wellness | creative | entrepreneur | gamer")

    model_config = {"json_schema_extra": {"example": {
        "user_id":  "veera_42",
        "username": "Veera",
        "build":    "intelligence",
    }}}


class LogActivityRequest(BaseModel):
    activity_type:    str       = Field(..., description="Activity key (e.g. gym_session, reading, coding)")
    duration_minutes: int       = Field(..., ge=1, le=480, description="Session duration in minutes (1-480)")
    intensity:        str       = Field(..., description="light | moderate | intense")
    notes:            Optional[str] = Field(None, max_length=200, description="Optional session notes")

    model_config = {"json_schema_extra": {"example": {
        "activity_type":    "reading",
        "duration_minutes": 45,
        "intensity":        "moderate",
        "notes":            "Finished chapter 3 of Deep Work",
    }}}


class BuildSwitchRequest(BaseModel):
    new_build: str  = Field(..., description="Target build type")
    mode:      str  = Field(..., description="'secondary' (unlock slot) | 'archive' (archive+restart)")

    model_config = {"json_schema_extra": {"example": {
        "new_build": "strength",
        "mode":      "secondary",
    }}}


class XPPreviewRequest(BaseModel):
    activity_type:    str = Field(..., description="Activity key")
    duration_minutes: int = Field(..., ge=1, le=480)
    intensity:        str = Field(..., description="light | moderate | intense")


# ── Response helpers ──────────────────────────────────────────────────────────

def _profile_summary(state: UserState, logs: list[dict]) -> dict:
    """Flat summary dict for the GET /profile/{id} response."""
    pb = state.primary_build
    return {
        "user_id":    state.user_id,
        "username":   state.username,
        "primary_build": {
            "build":       pb.build_type.value,
            "display":     BUILD_DISPLAY_NAMES[pb.build_type],
            "badge":       pb.badge_name or "Unranked",
            "next_badge":  pb.next_badge_name,
            "level":       pb.current_level,
            "total_xp":    pb.total_xp,
            "xp_to_badge": pb.xp_to_next_badge,
            "xp_to_level": pb.xp_to_next_level,
            "is_legendary": pb.is_legendary,
        },
        "secondary_build": state.secondary_build.to_dict() if state.secondary_build else None,
        "legacy_builds":   [b.to_dict() for b in state.legacy_builds],
        "streak": {
            "current":    state.current_streak,
            "longest":    state.longest_streak,
            "shields":    state.shields_remaining,
            "last_active": state.last_activity_date,
        },
        "daily": {
            "xp_today":    state.daily_xp_today,
            "xp_cap":      600,
            "xp_remaining": max(0, 600 - state.daily_xp_today),
            "cap_reached": state.daily_xp_today >= 600,
        },
        "recent_logs": logs[-5:],
        "total_activities": len(logs),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new user profile")
async def create_profile(body: CreateProfileRequest):
    """
    Create a new LevelUp user profile.

    Picks a build from:
    `strength | intelligence | dexterity | wellness | creative | entrepreneur | gamer`

    The profile is persisted to `data/profiles/{user_id}.json`.
    """
    # Validate build
    try:
        build_type = BuildType(body.build.lower())
    except ValueError:
        valid = [b.value for b in BuildType]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown build '{body.build}'. Choose from: {valid}",
        )

    # Check if profile already exists
    if _profile_path(body.user_id).exists():
        raise HTTPException(
            status_code=409,
            detail=f"Profile '{body.user_id}' already exists.",
        )

    state = UserState(
        user_id       = body.user_id,
        username      = body.username,
        primary_build = BuildProfile(build_type=build_type),
    )

    _save(state, [])

    return {
        "created":  True,
        "user_id":  state.user_id,
        "username": state.username,
        "build":    build_type.value,
        "display":  BUILD_DISPLAY_NAMES[build_type],
        "description": BUILD_DESCRIPTIONS[build_type],
        "badge_ladder": BADGE_TIERS[build_type],
        "message": f"Welcome to LevelUp, {state.username}! Your {BUILD_DISPLAY_NAMES[build_type]} build is ready. Time to grind.",
    }


@router.get("/{user_id}", summary="Get user profile")
async def get_profile(user_id: str):
    """Return the full profile summary including XP, streak, and recent activity."""
    state, logs = _load(user_id)
    return _profile_summary(state, logs)


@router.get("/{user_id}/context", summary="Get chatbot context dict")
async def get_context(user_id: str):
    """
    Return the full context dict that gets injected into the LLM system prompt.
    Call this before every chat message to give the AI up-to-date user stats.
    """
    state, logs = _load(user_id)
    recent = [
        EngineActivityLog(
            activity_type    = log["activity"],
            duration_minutes = log["duration"],
            intensity        = log["intensity"],
            xp_earned        = log["xp"],
            logged_at        = log.get("logged_at", ""),
        )
        for log in logs[-5:]
    ]
    return state.to_context_dict(recent_logs=recent)


@router.post("/{user_id}/log", summary="Log an activity and earn XP")
async def log_activity(user_id: str, body: LogActivityRequest):
    """
    Log a completed activity for the user.

    - Calculates XP using the full formula (duration × intensity × relevance × streak)
    - Updates the daily XP counter and enforces the daily cap
    - Advances or repairs the streak (auto-consumes shields for 1-day gaps)
    - Triggers badge check — notifies if a new badge was earned

    Returns the XP breakdown, streak status, and updated profile summary.
    """
    state, logs = _load(user_id)

    # Validate intensity
    try:
        intensity = Intensity(body.intensity.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid intensity '{body.intensity}'. Use: light | moderate | intense",
        )

    # Calculate XP
    xp_result = calculate_xp(
        activity_type    = body.activity_type,
        duration_minutes = body.duration_minutes,
        intensity        = intensity,
        primary_build    = state.primary_build.build_type,
        current_streak   = state.current_streak,
        current_level    = state.primary_build.current_level,
        daily_xp_so_far  = state.daily_xp_today,
    )

    # Update streak
    streak_update = state.update_streak()

    # Add XP to state
    xp_update = state.add_xp(xp_result.final_xp)

    # Append to activity history
    log_entry = {
        "activity":  body.activity_type,
        "duration":  body.duration_minutes,
        "intensity": body.intensity.lower(),
        "xp":        xp_result.final_xp,
        "notes":     body.notes,
        "logged_at": datetime.now().isoformat(),
    }
    logs.append(log_entry)

    # Persist
    _save(state, logs)

    # Build response
    response = {
        "logged":   True,
        "activity": body.activity_type,
        "xp": {
            "earned":          xp_result.final_xp,
            "base":            int(xp_result.base_xp),
            "intensity_mult":  xp_result.intensity_multiplier,
            "relevance_mult":  xp_result.relevance_multiplier,
            "streak_mult":     xp_result.streak_multiplier,
            "was_capped":      xp_result.was_capped,
            "daily_total":     xp_result.daily_xp_after,
            "daily_cap":       xp_result.daily_cap,
            "daily_remaining": xp_result.daily_xp_remaining,
        },
        "streak": {
            "current":      streak_update["current_streak"],
            "shield_used":  streak_update["shield_used"],
            "broken":       streak_update["streak_broken"],
            "new_power_up": xp_result.power_up_name,
        },
        "profile": _profile_summary(state, logs),
    }

    # Badge notification
    if xp_update.get("badge_earned"):
        response["badge_earned"] = xp_update["badge_earned"]

    # Motivational flag for chatbot tone (fire-and-forget — no model needed)
    if streak_update["streak_broken"]:
        response["chatbot_hint"] = "support_mode"
    elif xp_update.get("badge_earned") or xp_result.power_up_name:
        response["chatbot_hint"] = "hype_mode"

    return response


@router.get("/{user_id}/xp/preview", summary="Preview XP for an activity without logging")
async def preview_xp(user_id: str, activity_type: str, duration_minutes: int, intensity: str = "moderate"):
    """
    Calculate the XP a user would earn for an activity WITHOUT logging it.
    Useful for the UI to show projected XP before the user submits.
    """
    state, _ = _load(user_id)

    try:
        intensity_enum = Intensity(intensity.lower())
    except ValueError:
        raise HTTPException(400, detail="intensity must be light | moderate | intense")

    result = calculate_xp(
        activity_type    = activity_type,
        duration_minutes = duration_minutes,
        intensity        = intensity_enum,
        primary_build    = state.primary_build.build_type,
        current_streak   = state.current_streak,
        current_level    = state.primary_build.current_level,
        daily_xp_so_far  = state.daily_xp_today,
    )

    return {
        "activity":       activity_type,
        "duration":       duration_minutes,
        "intensity":      intensity,
        "projected_xp":   result.final_xp,
        "breakdown": {
            "base":           int(result.base_xp),
            "intensity_mult": result.intensity_multiplier,
            "relevance_mult": result.relevance_multiplier,
            "streak_mult":    result.streak_multiplier,
            "raw_xp":         round(result.raw_xp, 1),
            "is_primary":     result.is_primary,
            "would_cap":      result.was_capped,
        },
        "daily_remaining": result.daily_xp_remaining + result.final_xp,  # before this activity
    }


@router.get("/{user_id}/xp/best", summary="Get highest-XP activities for this user")
async def best_activities(user_id: str, duration: int = 60):
    """
    Return the top activities ranked by XP earned for a given duration.
    Uses the user's current streak multiplier and daily remaining cap.
    Useful for the 'What should I do today?' chatbot feature.
    """
    state, _ = _load(user_id)

    ranked = best_activity_for_xp(
        primary_build    = state.primary_build.build_type,
        current_streak   = state.current_streak,
        current_level    = state.primary_build.current_level,
        daily_xp_so_far  = state.daily_xp_today,
        duration_minutes = duration,
    )

    return {
        "build":    state.primary_build.build_type.value,
        "duration": duration,
        "streak":   state.current_streak,
        "ranked_activities": [
            {"activity": act, "projected_xp": xp}
            for act, xp in ranked[:10]
        ],
    }


@router.post("/{user_id}/build/switch", summary="Switch or unlock a secondary build")
async def switch_build(user_id: str, body: BuildSwitchRequest):
    """
    Unlock a new build (requires Legendary badge on the primary build).

    **mode = 'secondary'** — Keep your primary build active, unlock a second build slot.
    XP on secondary activities earns at 0.75× the normal rate.

    **mode = 'archive'** — Archive your current build (badge + XP permanently preserved),
    and start a fresh primary build with a 500 XP legacy bonus.
    """
    state, logs = _load(user_id)

    try:
        new_build = BuildType(body.new_build.lower())
    except ValueError:
        raise HTTPException(400, detail=f"Unknown build '{body.new_build}'")

    if body.mode == "secondary":
        result = state.unlock_secondary_build(new_build)
    elif body.mode == "archive":
        result = state.archive_primary_and_start_new(new_build)
    else:
        raise HTTPException(400, detail="mode must be 'secondary' or 'archive'")

    if not result["success"]:
        raise HTTPException(403, detail=result["reason"])

    _save(state, logs)
    return {**result, "profile": _profile_summary(state, logs)}


@router.get("/builds/list", summary="List all available builds")
async def list_builds():
    """Return all 7 builds with display names, descriptions, and badge ladders."""
    return {
        b.value: {
            "display":      BUILD_DISPLAY_NAMES[b],
            "description":  BUILD_DESCRIPTIONS[b],
            "badge_ladder": BADGE_TIERS[b],
        }
        for b in BuildType
    }

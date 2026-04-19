"""
generate_anomaly_data.py - Synthetic activity log dataset for anomaly detection.

Generates 10,000 labelled activity log records (80% normal, 20% anomalous).
Three anomaly types are injected to train the classifiers to detect XP cheating:

  Type A — XP Grinding        : 8+ sessions per day or single session > 300 min
  Type B — Impossible Streaks : streak advanced without matching log dates
  Type C — Intensity Spoofing : rapid intensity switching on same activity

Output:
  data/raw/anomaly_logs.csv   — full dataset
  data/raw/anomaly_train.csv  — 80% train split
  data/raw/anomaly_test.csv   — 20% test split

Features (8 numeric, all normalised for model training):
  activities_per_day        — how many sessions logged in that day
  daily_xp_total            — total XP earned that day
  streak_gap_days           — days since last activity (0 = same day, 1 = consecutive)
  intensity_switch_rate     — fraction of sessions that switched intensity vs previous
  avg_session_duration      — mean duration across all sessions that day
  max_session_duration      — longest single session that day
  sessions_at_cap_ratio     — fraction of days where daily XP hit the cap
  xp_per_minute             — daily_xp_total / total active minutes

Label:
  is_anomaly  — 0 (normal) or 1 (anomalous)
  anomaly_type — "none" | "xp_grinding" | "impossible_streak" | "intensity_spoofing"
"""

import random
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Literal

SEED = 42
random.seed(SEED)

ACTIVITIES = [
    "gym_session", "running", "walking", "sports", "martial_arts",
    "study_session", "reading", "online_course", "research",
    "instrument_practice", "coding_sprint", "typing_practice",
    "meditation", "journaling", "sleep_tracking", "nutrition_tracking",
    "art_session", "writing", "music_production",
    "networking", "business_task",
    "competitive_gaming", "speedrunning", "esports_practice", "apm_training",
]

INTENSITIES = ["light", "moderate", "intense"]
INTENSITY_MULT = {"light": 1.0, "moderate": 1.5, "intense": 2.0}
DAILY_CAP = 600


# ── Record Structure ──────────────────────────────────────────────────────────

@dataclass
class LogRecord:
    activities_per_day:    int
    daily_xp_total:        float
    streak_gap_days:       int
    intensity_switch_rate: float
    avg_session_duration:  float
    max_session_duration:  int
    sessions_at_cap_ratio: float
    xp_per_minute:         float
    is_anomaly:            int   # 0 or 1
    anomaly_type:          str   # "none" | "xp_grinding" | "impossible_streak" | "intensity_spoofing"


# ── XP Calculator (simplified, no streak mult for generation) ─────────────────

def calc_xp(duration: int, intensity: str) -> float:
    return duration * INTENSITY_MULT[intensity]


# ── Normal Log Generator ──────────────────────────────────────────────────────

def gen_normal() -> LogRecord:
    """Realistic user: 1-4 sessions/day, normal durations, consistent streak."""
    num_sessions = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
    durations    = [random.randint(15, 120) for _ in range(num_sessions)]
    intensities  = random.choices(INTENSITIES, weights=[50, 35, 15], k=num_sessions)

    daily_xp = min(sum(calc_xp(d, i) for d, i in zip(durations, intensities)), DAILY_CAP)

    # Intensity switch rate: consecutive sessions compared
    switches = sum(
        1 for j in range(1, num_sessions) if intensities[j] != intensities[j-1]
    )
    switch_rate = switches / max(1, num_sessions - 1)

    # Streak gap: mostly 0 (same day) or 1 (consecutive), occasional 2 (shield used)
    gap = random.choices([0, 1, 2], weights=[20, 70, 10])[0]

    at_cap = 1.0 if daily_xp >= DAILY_CAP * 0.98 else 0.0
    total_mins = sum(durations)

    return LogRecord(
        activities_per_day    = num_sessions,
        daily_xp_total        = round(daily_xp, 1),
        streak_gap_days       = gap,
        intensity_switch_rate = round(switch_rate, 3),
        avg_session_duration  = round(total_mins / num_sessions, 1),
        max_session_duration  = max(durations),
        sessions_at_cap_ratio = at_cap,
        xp_per_minute         = round(daily_xp / total_mins, 3),
        is_anomaly            = 0,
        anomaly_type          = "none",
    )


# ── Anomaly Type A: XP Grinding ───────────────────────────────────────────────

def gen_anomaly_xp_grinding() -> LogRecord:
    """
    8-15 sessions per day OR single session > 300 min.
    Daily XP suspiciously near or exceeding cap every day.
    """
    if random.random() < 0.5:
        # Too many sessions
        num_sessions = random.randint(8, 15)
        durations    = [random.randint(15, 60) for _ in range(num_sessions)]
    else:
        # One impossibly long session
        num_sessions = 1
        durations    = [random.randint(301, 480)]

    intensities  = random.choices(INTENSITIES, weights=[10, 30, 60], k=num_sessions)
    daily_xp     = sum(calc_xp(d, i) for d, i in zip(durations, intensities))
    # Grinder always hits cap
    capped_xp    = max(daily_xp, DAILY_CAP * random.uniform(0.98, 1.3))

    switches = sum(
        1 for j in range(1, num_sessions) if intensities[j] != intensities[j-1]
    )
    switch_rate  = switches / max(1, num_sessions - 1)
    total_mins   = sum(durations)

    return LogRecord(
        activities_per_day    = num_sessions,
        daily_xp_total        = round(capped_xp, 1),
        streak_gap_days       = 0,    # always consecutive — never misses a day
        intensity_switch_rate = round(switch_rate, 3),
        avg_session_duration  = round(total_mins / num_sessions, 1),
        max_session_duration  = max(durations),
        sessions_at_cap_ratio = 1.0,
        xp_per_minute         = round(capped_xp / total_mins, 3),
        is_anomaly            = 1,
        anomaly_type          = "xp_grinding",
    )


# ── Anomaly Type B: Impossible Streak ────────────────────────────────────────

def gen_anomaly_impossible_streak() -> LogRecord:
    """
    Streak counter advanced without plausible activity log.
    Gap of 3-7 days but streak not reset — impossible without manual edit.
    Also: XP earned per minute is suspiciously low (activity logged but not real).
    """
    num_sessions = random.randint(1, 2)
    durations    = [random.randint(5, 20) for _ in range(num_sessions)]   # suspiciously short
    intensities  = random.choices(INTENSITIES, weights=[70, 25, 5], k=num_sessions)
    daily_xp     = sum(calc_xp(d, i) for d, i in zip(durations, intensities))
    total_mins   = sum(durations)

    return LogRecord(
        activities_per_day    = num_sessions,
        daily_xp_total        = round(daily_xp, 1),
        streak_gap_days       = random.randint(3, 7),   # ANOMALY: large gap but streak not reset
        intensity_switch_rate = 0.0,
        avg_session_duration  = round(total_mins / num_sessions, 1),
        max_session_duration  = max(durations),
        sessions_at_cap_ratio = 0.0,
        xp_per_minute         = round(daily_xp / total_mins, 3),
        is_anomaly            = 1,
        anomaly_type          = "impossible_streak",
    )


# ── Anomaly Type C: Intensity Spoofing ───────────────────────────────────────

def gen_anomaly_intensity_spoofing() -> LogRecord:
    """
    Rapid intensity switching on same activity to maximise multiplier.
    Very high switch rate (> 0.85) with suspiciously high XP per minute.
    """
    num_sessions = random.randint(4, 8)
    durations    = [random.randint(10, 30) for _ in range(num_sessions)]

    # Alternates light → intense → light → intense... to game the multiplier
    intensities  = ["light" if j % 2 == 0 else "intense" for j in range(num_sessions)]
    # XP calculated as if all intense
    daily_xp     = sum(calc_xp(d, "intense") for d in durations)
    capped_xp    = min(daily_xp, DAILY_CAP)

    switch_rate  = 1.0   # switches every session — definitive spoof pattern
    total_mins   = sum(durations)

    return LogRecord(
        activities_per_day    = num_sessions,
        daily_xp_total        = round(capped_xp, 1),
        streak_gap_days       = random.choices([0, 1], weights=[60, 40])[0],
        intensity_switch_rate = switch_rate,
        avg_session_duration  = round(total_mins / num_sessions, 1),
        max_session_duration  = max(durations),
        sessions_at_cap_ratio = 1.0 if capped_xp >= DAILY_CAP * 0.98 else 0.5,
        xp_per_minute         = round(capped_xp / total_mins, 3),
        is_anomaly            = 1,
        anomaly_type          = "intensity_spoofing",
    )


# ── Dataset Builder ───────────────────────────────────────────────────────────

ANOMALY_GENERATORS = [
    gen_anomaly_xp_grinding,
    gen_anomaly_impossible_streak,
    gen_anomaly_intensity_spoofing,
]


def generate_dataset(
    n_total: int = 10_000,
    anomaly_ratio: float = 0.20,
    output_dir: str = "data/raw",
) -> tuple[str, str, str]:
    """
    Generate the full anomaly detection dataset and split into train/test.

    Returns paths: (full_csv, train_csv, test_csv)
    """
    os.makedirs(output_dir, exist_ok=True)

    n_anomaly = int(n_total * anomaly_ratio)
    n_normal  = n_total - n_anomaly

    records: list[LogRecord] = []

    # Normal records
    for _ in range(n_normal):
        records.append(gen_normal())

    # Anomaly records — equally split across 3 types
    per_type = n_anomaly // 3
    remainder = n_anomaly - per_type * 3
    for gen_fn, count in zip(ANOMALY_GENERATORS, [per_type, per_type, per_type + remainder]):
        for _ in range(count):
            records.append(gen_fn())

    # Shuffle
    random.shuffle(records)

    # Write full CSV
    fieldnames = list(asdict(records[0]).keys())
    full_path  = os.path.join(output_dir, "anomaly_logs.csv")
    with open(full_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    # Train / test split (80/20)
    split_idx  = int(len(records) * 0.8)
    train_recs = records[:split_idx]
    test_recs  = records[split_idx:]

    train_path = os.path.join(output_dir, "anomaly_train.csv")
    test_path  = os.path.join(output_dir, "anomaly_test.csv")

    for path, recs in [(train_path, train_recs), (test_path, test_recs)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(r) for r in recs)

    return full_path, train_path, test_path


# ── Stats Printer ─────────────────────────────────────────────────────────────

def print_stats(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    total    = len(rows)
    normal   = sum(1 for r in rows if r["is_anomaly"] == "0")
    anomalous = total - normal
    types    = {}
    for r in rows:
        t = r["anomaly_type"]
        types[t] = types.get(t, 0) + 1

    print(f"\n  Total records : {total:,}")
    print(f"  Normal        : {normal:,}  ({100*normal/total:.1f}%)")
    print(f"  Anomalous     : {anomalous:,}  ({100*anomalous/total:.1f}%)")
    print("  Breakdown:")
    for k, v in sorted(types.items()):
        print(f"    {k:<30} {v:,}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating anomaly detection dataset...")
    full, train, test = generate_dataset(n_total=10_000, anomaly_ratio=0.20)
    print(f"\nFiles saved:")
    print(f"  Full  : {full}")
    print(f"  Train : {train}")
    print(f"  Test  : {test}")
    print("\nDataset stats (full):")
    print_stats(full)
    print("\nTrain split stats:")
    print_stats(train)
    print("\nTest split stats:")
    print_stats(test)
    print("\nDone. Ready for scikit-learn training.")

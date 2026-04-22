"""
boost_all_builds_to_4000.py
============================
Ensures every build has 4000+ training samples by:

  1. Loading all existing labeled samples per build
  2. Loading the 20,806 GENERIC samples and re-tagging them
     using keyword classification to the right build
  3. Loading HuggingFace datasets for any build still under 4000
  4. Writing the final balanced finetune_train.jsonl

Target: 4000+ samples per build (7 builds = 28,000+ total)

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/boost_all_builds_to_4000.py
"""

import os, json, random
from collections import defaultdict
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_JSONL    = os.path.join("data", "raw", "finetune_train.jsonl")
VAL_JSONL      = os.path.join("data", "raw", "finetune_val.jsonl")
MISSING_DIR    = os.path.join("data", "raw", "missing_builds")
OUT_TRAIN      = os.path.join("data", "raw", "finetune_train_v2.jsonl")
OUT_VAL        = os.path.join("data", "raw", "finetune_val_v2.jsonl")
TARGET         = 4000

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "TITAN": (
        "You are TITAN, the AI companion for LevelUp's STRENGTH build — an RPG "
        "self-improvement app. You coach users on weightlifting, muscle building, "
        "progressive overload, nutrition for strength, and physical resilience. "
        "You speak with intensity, directness, and iron discipline. "
        "The body is the foundation. Help the user level up their physical strength."
    ),
    "ORACLE": (
        "You are ORACLE, the AI companion for LevelUp's INTELLIGENCE build — an RPG "
        "self-improvement app. You help users with learning, programming, mathematics, "
        "science, critical thinking, research skills, and mental sharpness. "
        "You speak with precision, curiosity, and intellectual depth. "
        "Knowledge is power. Help the user master their mind."
    ),
    "PHANTOM": (
        "You are PHANTOM, the AI companion for LevelUp's DEXTERITY build — an RPG "
        "self-improvement app. You coach users on parkour, martial arts, gymnastics, "
        "agility training, reflexes, body control, climbing, and movement skills. "
        "You speak with precision and fluid energy. "
        "Mastery of movement is mastery of self. Help the user become unstoppable."
    ),
    "SAGE": (
        "You are SAGE, the AI companion for LevelUp's WELLNESS build — an RPG "
        "self-improvement app. You guide users on mindfulness, meditation, sleep, "
        "mental health, stress management, yoga, and emotional wellbeing. "
        "You speak with calm wisdom and grounded clarity. "
        "Stillness is the highest power. Help the user build deep wellbeing."
    ),
    "MUSE": (
        "You are MUSE, the AI companion for LevelUp's CREATIVE build — an RPG "
        "self-improvement app. You guide users on writing, music, visual art, "
        "filmmaking, game design, creative thinking, and building a creative practice. "
        "You speak with warmth and imagination. "
        "Creativity is a muscle, not a gift. Help the user find their creative voice."
    ),
    "EMPIRE": (
        "You are EMPIRE, the AI companion for LevelUp's ENTREPRENEUR build — an RPG "
        "self-improvement app. You coach users on building businesses, startups, "
        "personal finance, productivity, leadership, sales, and entrepreneurial mindset. "
        "You speak with strategic clarity and hard-won wisdom. "
        "Execution beats ideas. Help the user build something that lasts."
    ),
    "GG": (
        "You are GG, the AI companion for LevelUp's GAMER build — an RPG "
        "self-improvement app where competitive gaming, speedrunning, D&D, and "
        "esports culture meet real-world self-improvement. "
        "You coach users on game strategy, D&D worldbuilding, esports performance, "
        "tilt management, and gaming-driven personal growth. "
        "The game was always real. Help the user level up in every sense."
    ),
}

# ── Keyword classifiers ────────────────────────────────────────────────────────
# Each build has primary and secondary keywords
# Primary keywords = strong signal (score 3)
# Secondary keywords = moderate signal (score 1)

BUILD_KEYWORDS = {
    "TITAN": {
        "primary": [
            "weightlift", "squat", "deadlift", "bench press", "muscle", "bodybuilding",
            "progressive overload", "1rm", "one rep max", "bulking", "cutting",
            "hypertrophy", "powerlifting", "strength training", "gym routine",
            "protein intake", "creatine", "pre-workout", "post-workout", "gains",
        ],
        "secondary": [
            "workout", "exercise", "gym", "fitness", "lift", "training", "strength",
            "protein", "diet", "body", "physical", "cardio", "weight", "rep", "set",
            "nutrition", "athlete", "muscle mass", "fat loss", "testosterone",
        ],
    },
    "ORACLE": {
        "primary": [
            "learn programming", "study math", "algorithm", "data structure",
            "machine learning", "deep learning", "neural network", "calculus",
            "linear algebra", "statistics", "python", "javascript", "code review",
            "learn faster", "speed reading", "memory technique", "feynman",
        ],
        "secondary": [
            "learn", "study", "math", "code", "programming", "science", "knowledge",
            "education", "research", "logic", "problem", "think", "intelligence",
            "book", "understand", "explain", "concept", "theory", "formula", "brain",
        ],
    },
    "PHANTOM": {
        "primary": [
            "parkour", "free running", "bjj", "brazilian jiu jitsu", "judo", "karate",
            "muay thai", "martial arts", "gymnastics", "agility ladder", "flexibility",
            "splits", "rock climbing", "bouldering", "calisthenics", "handstand",
            "backflip", "reaction time", "reflexes", "body control",
        ],
        "secondary": [
            "agile", "nimble", "movement", "balance", "coordination", "dexterity",
            "stretch", "climb", "jump", "sprint", "run", "dance", "sport", "combat",
            "fight", "kick", "punch", "roll", "flip", "speed", "athletic",
        ],
    },
    "SAGE": {
        "primary": [
            "meditation", "mindfulness", "anxiety", "depression", "mental health",
            "stress relief", "sleep quality", "insomnia", "yoga", "breathing exercise",
            "panic attack", "therapy", "counseling", "gratitude journal",
            "emotional regulation", "burnout recovery", "self-compassion",
        ],
        "secondary": [
            "stress", "calm", "peace", "relax", "breathe", "sleep", "rest",
            "mental", "emotional", "wellbeing", "wellness", "health", "heal",
            "mindset", "positive", "journal", "self-care", "balance", "energy",
            "overwhelm", "fear", "worry", "cope", "grounded", "present",
        ],
    },
    "MUSE": {
        "primary": [
            "creative writing", "fiction writing", "novel", "screenplay", "poetry",
            "music production", "songwriting", "music theory", "drawing technique",
            "digital art", "illustration", "graphic design", "photography",
            "filmmaking", "game design", "world building", "storytelling",
        ],
        "secondary": [
            "write", "art", "music", "draw", "paint", "creative", "design",
            "story", "poem", "song", "compose", "fiction", "craft", "create",
            "photograph", "film", "animate", "character", "plot", "narrative",
            "inspiration", "style", "voice", "imagination", "invent",
        ],
    },
    "EMPIRE": {
        "primary": [
            "startup", "entrepreneur", "business model", "revenue model", "MVP",
            "product market fit", "venture capital", "pitch deck", "saas",
            "personal finance", "investing", "passive income", "freelancing",
            "leadership", "management", "sales funnel", "marketing strategy",
            "cash flow", "valuation", "equity",
        ],
        "secondary": [
            "business", "company", "money", "income", "profit", "customer",
            "market", "product", "brand", "growth", "scale", "strategy",
            "finance", "invest", "sell", "buy", "deal", "negotiate", "client",
            "team", "hire", "manage", "goal", "success", "launch",
        ],
    },
    "GG": {
        "primary": [
            "d&d", "dungeons and dragons", "tabletop rpg", "dnd", "dungeon master",
            "league of legends", "valorant", "esports", "speedrun", "competitive gaming",
            "game theory", "meta", "build path", "skill tree", "boss fight",
            "lore", "worldbuilding rpg", "game mechanic", "npc",
        ],
        "secondary": [
            "game", "gaming", "gamer", "play", "player", "quest", "level", "xp",
            "rpg", "fantasy", "character", "dungeon", "raid", "guild", "loot",
            "skill", "ability", "strategy", "win", "rank", "tournament", "stream",
        ],
    },
}

def score_sample(text, build):
    """Score how relevant a text is to a given build."""
    text_lower = text.lower()
    score = 0
    for kw in BUILD_KEYWORDS[build]["primary"]:
        if kw in text_lower:
            score += 3
    for kw in BUILD_KEYWORDS[build]["secondary"]:
        if kw in text_lower:
            score += 1
    return score

def classify_sample(sample):
    """Return the best-matching build for a sample, or None if weak signal."""
    text = (sample.get("user","") + " " + sample.get("assistant","")).lower()
    scores = {build: score_sample(text, build) for build in SYSTEM_PROMPTS}
    best_build = max(scores, key=scores.get)
    best_score = scores[best_build]
    if best_score < 3:
        return None  # Weak signal — keep as generic
    return best_build

def detect_build_from_system(system_prompt):
    """Detect which build a sample already belongs to."""
    for build in SYSTEM_PROMPTS:
        if build in system_prompt:
            return build
    return "GENERIC"

def make_sample(build, user, assistant):
    return {
        "system": SYSTEM_PROMPTS[build],
        "user": user.strip(),
        "assistant": assistant.strip(),
    }

# ── Step 1: Load all existing samples ─────────────────────────────────────────

print("=" * 60)
print("LevelUp AI -- Boost All Builds to 4000+")
print("=" * 60)

print("\n[1/5] Loading existing training data ...")
build_pools = defaultdict(list)

def load_jsonl(path):
    samples = []
    if not os.path.exists(path): return samples
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    pass
    return samples

# Load main train file
train_samples = load_jsonl(TRAIN_JSONL)
print(f"  Main train: {len(train_samples):,} samples")
generic_pool = []
for s in train_samples:
    build = detect_build_from_system(s.get("system",""))
    if build == "GENERIC":
        generic_pool.append(s)
    else:
        build_pools[build].append(s)

print(f"  Generic pool: {len(generic_pool):,} samples available for re-tagging")
for b, pool in build_pools.items():
    print(f"  {b}: {len(pool):,} existing labeled samples")

# Load missing_builds directory
missing_files = {
    "TITAN":   "titan_samples.jsonl",
    "SAGE":    "sage_samples.jsonl",
    "PHANTOM": "phantom_samples.jsonl",
    "EMPIRE":  "empire_samples.jsonl",
    "MUSE":    "muse_samples.jsonl",
}
for build, fname in missing_files.items():
    path = os.path.join(MISSING_DIR, fname)
    loaded = load_jsonl(path)
    if loaded:
        build_pools[build].extend(loaded)
        print(f"  + {build} missing_builds: +{len(loaded):,} samples")

print("\nAfter loading all sources:")
for b in SYSTEM_PROMPTS:
    print(f"  {b}: {len(build_pools[b]):,} samples")

# ── Step 2: Re-tag GENERIC samples ────────────────────────────────────────────

print(f"\n[2/5] Re-tagging {len(generic_pool):,} GENERIC samples ...")
re_tagged = defaultdict(list)
unclassified = []

for s in generic_pool:
    build = classify_sample(s)
    if build:
        re_tagged[build].append(s)
    else:
        unclassified.append(s)

print(f"  Successfully re-tagged: {sum(len(v) for v in re_tagged.values()):,}")
print(f"  Remained unclassified:  {len(unclassified):,}")
for b, pool in re_tagged.items():
    print(f"    -> {b}: {len(pool):,} samples")

# Add re-tagged samples to pools (with corrected system prompts)
for build, samples in re_tagged.items():
    for s in samples:
        new_s = {
            "system":    SYSTEM_PROMPTS[build],
            "user":      s.get("user","").strip(),
            "assistant": s.get("assistant","").strip(),
        }
        if new_s["user"] and new_s["assistant"]:
            build_pools[build].append(new_s)

print("\nAfter re-tagging:")
for b in SYSTEM_PROMPTS:
    print(f"  {b}: {len(build_pools[b]):,} samples  (need {max(0, TARGET - len(build_pools[b])):,} more)")

# ── Step 3: HuggingFace top-ups for any build still under 4000 ────────────────

print(f"\n[3/5] HuggingFace top-ups for builds under {TARGET:,} ...")

def hf_topup(build, ds_name, split, extractor, keywords=None, max_samples=2000):
    needed = TARGET - len(build_pools[build])
    if needed <= 0:
        print(f"  {build}: already at {len(build_pools[build]):,} -- skipping HF")
        return
    print(f"  {build}: needs {needed:,} more -- loading {ds_name} ...")
    try:
        ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            if count >= min(needed + 200, max_samples): break
            try:
                q, a = extractor(row)
            except:
                continue
            q, a = str(q).strip(), str(a).strip()
            if not q or not a or len(a) < 60: continue
            if keywords and not any(k in (q+a).lower() for k in keywords): continue
            build_pools[build].append(make_sample(build, q, a))
            count += 1
        print(f"    Added {count:,} samples from HF")
    except Exception as e:
        print(f"    Failed: {e}")

# TITAN top-up — HealthCareMagic fitness filter
if len(build_pools["TITAN"]) < TARGET:
    hf_topup(
        "TITAN",
        "lavita/ChatDoctor-HealthCareMagic-100k", "train",
        lambda r: (r.get("input",""), r.get("output","")),
        keywords=["exercise","workout","gym","muscle","weight","fitness",
                  "protein","strength","training","lift","cardio","body"],
        max_samples=3000,
    )

# ORACLE top-up — education + programming Q&A
if len(build_pools["ORACLE"]) < TARGET:
    hf_topup(
        "ORACLE",
        "iamtarun/python_code_instructions_18k_alpaca", "train",
        lambda r: (r.get("instruction","") + " " + r.get("input",""),
                   r.get("output","")),
        max_samples=3000,
    )

# SAGE top-up — mental health counseling
if len(build_pools["SAGE"]) < TARGET:
    hf_topup(
        "SAGE",
        "Amod/mental_health_counseling_conversations", "train",
        lambda r: (r.get("Context",""), r.get("Response","")),
        max_samples=3000,
    )
if len(build_pools["SAGE"]) < TARGET:
    hf_topup(
        "SAGE",
        "heliosbrahma/mental_health_conversational_data", "train",
        lambda r: (r.get("Context",""), r.get("Response","")),
        max_samples=2000,
    )

# PHANTOM top-up — fitness/sports (closest available)
if len(build_pools["PHANTOM"]) < TARGET:
    hf_topup(
        "PHANTOM",
        "lavita/ChatDoctor-HealthCareMagic-100k", "train",
        lambda r: (r.get("input",""), r.get("output","")),
        keywords=["sport","run","climb","martial","gymnast","agility","flex",
                  "parkour","movement","balance","coordination","athletic",
                  "dance","yoga","stretch","cardio","sprint","jump","train"],
        max_samples=3000,
    )

# MUSE top-up — writing + creative
if len(build_pools["MUSE"]) < TARGET:
    hf_topup(
        "MUSE",
        "teknium/GPT4-LLM-Cleaned", "train",
        lambda r: (r.get("instruction",""), r.get("output","")),
        keywords=["write","story","poem","song","music","art","draw","paint",
                  "creative","design","fiction","character","plot","compose",
                  "craft","imagine","novel","screenplay","lyric","voice"],
        max_samples=3000,
    )

# EMPIRE top-up — business / finance
if len(build_pools["EMPIRE"]) < TARGET:
    hf_topup(
        "EMPIRE",
        "gbharti/finance-alpaca", "train",
        lambda r: (r.get("instruction","") + " " + r.get("input",""),
                   r.get("output","")),
        max_samples=3000,
    )
if len(build_pools["EMPIRE"]) < TARGET:
    hf_topup(
        "EMPIRE",
        "teknium/GPT4-LLM-Cleaned", "train",
        lambda r: (r.get("instruction",""), r.get("output","")),
        keywords=["business","startup","entrepreneur","revenue","profit","product",
                  "market","customer","sales","invest","fund","company","strategy",
                  "leadership","productivity","finance","budget","income","brand"],
        max_samples=3000,
    )

# GG top-up if needed
if len(build_pools["GG"]) < TARGET:
    hf_topup(
        "GG",
        "teknium/GPT4-LLM-Cleaned", "train",
        lambda r: (r.get("instruction",""), r.get("output","")),
        keywords=["game","gaming","dnd","rpg","dungeon","fantasy","esport",
                  "player","quest","character","lore","strategy","boss","level"],
        max_samples=2000,
    )

# ── Step 4: Cap each build at 5000, shuffle, and split ────────────────────────

print(f"\n[4/5] Final counts and train/val split ...")

CAP = 5000
all_train, all_val = [], []
final_counts = {}

for build in SYSTEM_PROMPTS:
    pool = build_pools[build]
    # Deduplicate by user text
    seen = set()
    deduped = []
    for s in pool:
        key = s.get("user","")[:100]
        if key not in seen and s.get("user") and s.get("assistant"):
            seen.add(key)
            deduped.append(s)
    random.seed(42)
    random.shuffle(deduped)
    deduped = deduped[:CAP]
    split_idx = max(1, int(len(deduped) * 0.9))
    all_train.extend(deduped[:split_idx])
    all_val.extend(deduped[split_idx:])
    final_counts[build] = len(deduped)
    status = "OK" if len(deduped) >= TARGET else "LOW"
    print(f"  [{status}] {build}: {len(deduped):,} total  "
          f"(train: {split_idx:,}  val: {len(deduped)-split_idx:,})")

# Shuffle final sets
random.shuffle(all_train)
random.shuffle(all_val)

# ── Step 5: Save ───────────────────────────────────────────────────────────────

print(f"\n[5/5] Saving new dataset files ...")

def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {path}: {len(samples):,} samples  ({size_mb:.1f} MB)")

save_jsonl(OUT_TRAIN, all_train)
save_jsonl(OUT_VAL, all_val)

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DONE -- Final Dataset Summary")
print("=" * 60)
print(f"{'Build':<12} {'Samples':>8}  {'Status'}")
print("-" * 35)
for build, count in final_counts.items():
    status = "OK" if count >= TARGET else f"!! needs {TARGET-count} more"
    print(f"{build:<12} {count:>8,}  {status}")

print("-" * 35)
print(f"{'TOTAL':<12} {sum(final_counts.values()):>8,}")
print(f"\nTrain file : {OUT_TRAIN}  ({len(all_train):,} samples)")
print(f"Val file   : {OUT_VAL}  ({len(all_val):,} samples)")
print(f"\nNext step  : rename v2 files to replace originals, then run training.")
print("=" * 60)

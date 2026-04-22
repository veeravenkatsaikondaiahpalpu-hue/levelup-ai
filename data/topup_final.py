"""
topup_final.py
==============
Final top-up for the 4 builds still under 4000:
  SAGE    : needs 2,412 more
  PHANTOM : needs   912 more
  TITAN   : needs   202 more
  EMPIRE  : needs   384 more

Appends directly to finetune_train_v2.jsonl / finetune_val_v2.jsonl

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/topup_final.py
"""

import os, json, random
from datasets import load_dataset

TRAIN_V2 = os.path.join("data", "raw", "finetune_train_v2.jsonl")
VAL_V2   = os.path.join("data", "raw", "finetune_val_v2.jsonl")

SYSTEM_PROMPTS = {
    "TITAN": (
        "You are TITAN, the AI companion for LevelUp's STRENGTH build — an RPG "
        "self-improvement app. You coach users on weightlifting, muscle building, "
        "progressive overload, nutrition for strength, and physical resilience. "
        "You speak with intensity, directness, and iron discipline. "
        "The body is the foundation. Help the user level up their physical strength."
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
    "EMPIRE": (
        "You are EMPIRE, the AI companion for LevelUp's ENTREPRENEUR build — an RPG "
        "self-improvement app. You coach users on building businesses, startups, "
        "personal finance, productivity, leadership, sales, and entrepreneurial mindset. "
        "You speak with strategic clarity and hard-won wisdom. "
        "Execution beats ideas. Help the user build something that lasts."
    ),
}

def load_existing_keys(path):
    """Load first 120 chars of each user message as dedup keys."""
    keys = set()
    if not os.path.exists(path): return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
                keys.add(s.get("user","")[:120])
            except:
                pass
    return keys

def make_sample(build, q, a):
    return {"system": SYSTEM_PROMPTS[build], "user": q.strip(), "assistant": a.strip()}

def append_samples(train_path, val_path, samples):
    """Append samples to existing files, 90/10 split."""
    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.9))
    with open(train_path, "a", encoding="utf-8") as f:
        for s in samples[:split]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_path, "a", encoding="utf-8") as f:
        for s in samples[split:]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Appended: {split} train + {len(samples)-split} val = {len(samples)} total")

print("=" * 60)
print("LevelUp AI -- Final Top-Up Pass")
print("=" * 60)

# Load all existing keys to avoid duplicates
print("\nLoading existing keys for deduplication ...")
existing_keys = load_existing_keys(TRAIN_V2)
existing_keys |= load_existing_keys(VAL_V2)
print(f"  {len(existing_keys):,} existing unique samples")

targets = {"SAGE": 2500, "PHANTOM": 1000, "TITAN": 300, "EMPIRE": 500}

# ═══════════════════════════════════════════════════════════════════════════════
# SAGE — use 3 more diverse wellness/psychology datasets
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- SAGE (need ~2,500 more) ---")
sage_samples = []

sage_hf_sources = [
    ("vibhorag23/psychology_therapist_data", "train",
     lambda r: (r.get("input","") or r.get("question",""),
                r.get("output","") or r.get("answer",""))),
    ("mpingale/mental-health-chat-dataset", "train",
     lambda r: (r.get("questionTitle","") + " " + r.get("questionText",""),
                r.get("answerText",""))),
    ("samhog/psychology-10k", "train",
     lambda r: (r.get("input",""), r.get("output",""))),
    ("lavita/MedQuAD", "train",
     lambda r: (r.get("Question",""), r.get("Answer",""))),
]

wellness_keywords = [
    "stress","anxiety","depression","mental","emotion","therapy","counsel",
    "mindful","meditat","sleep","wellbeing","wellness","heal","cope","griev",
    "trauma","burnout","self-care","breath","calm","peace","anger","sad",
    "lonely","relation","self-esteem","confidence","habit","routine","mood"
]

for ds_name, split, extractor in sage_hf_sources:
    if len(sage_samples) >= targets["SAGE"]: break
    print(f"  Loading {ds_name} ...")
    try:
        ds = load_dataset(ds_name, split=split, streaming=True)
        count = 0
        for row in ds:
            if len(sage_samples) >= targets["SAGE"]: break
            try:
                q, a = extractor(row)
            except: continue
            q, a = str(q).strip(), str(a).strip()
            if not q or not a or len(a) < 60: continue
            # Use full text as key for less aggressive dedup
            key = q[:120]
            if key in existing_keys: continue
            if any(k in (q+a).lower() for k in wellness_keywords):
                sage_samples.append(make_sample("SAGE", q, a))
                existing_keys.add(key)
                count += 1
        print(f"    Added {count} samples")
    except Exception as e:
        print(f"    Failed: {e}")

# Handcrafted additional SAGE (diverse phrasings)
sage_extra = [
    ("My mind races at night and I can't fall asleep. What actually helps?",
     "Racing thoughts at bedtime are your brain's way of processing unfinished business from the day. Two techniques that have strong evidence: First, try 4-7-8 breathing — inhale for 4 counts, hold for 7, exhale slowly for 8. Do this 4 cycles. It physically activates your parasympathetic nervous system. Second, 'cognitive shuffle' — imagine random, unconnected images in sequence (a banana, then a lighthouse, then a red shoe). This mimics the brain's natural sleep-onset process and interrupts anxious thought loops. Keep your room cool (17-19°C), phone face-down outside the room, and same wake time every day regardless of when you fell asleep."),
    ("How do I stop being so hard on myself all the time?",
     "Self-criticism feels productive but it's actually just a habit — and habits can be changed. When the inner critic speaks, ask: 'Would I say this to a friend in the same situation?' Almost always the answer is no. Treat yourself with the same basic decency you'd extend to anyone else. Practically: when you catch a self-critical thought, write it down and then write a kinder reframe. Not toxic positivity — just fairness. 'I failed' becomes 'I tried something difficult and it didn't work out this time.' Over weeks, you're retraining the default narrative. Self-compassion isn't weakness — research consistently shows it leads to higher resilience and better performance than self-criticism."),
    ("I feel disconnected from people even in social situations. What's wrong with me?",
     "Nothing is wrong with you — disconnection in social situations is incredibly common and has several possible roots: anxiety that creates a layer between you and the moment, a mismatch between surface conversation and your need for depth, or simply being depleted. Start by noticing whether the disconnection has a pattern: specific situations, specific people, certain times of day. The fix is usually more presence, not more performance. Try anchoring to the physical: feel your feet on the ground, notice the sound of the voice you're listening to. Presence is a skill, not a personality trait. If it's persistent and painful, talking to a therapist — even just a few sessions — can give you a map of what's driving it."),
    ("What's the best morning routine for mental health?",
     "The most evidence-backed morning routine for mental health has five elements, none of which need to take long: (1) No phone for the first 20 minutes. Your first waking moments set the neurological tone of the day — don't hand them to social media. (2) Natural light within 30 minutes of waking. This anchors your circadian rhythm and regulates mood hormones. (3) Move your body — even 10 minutes of walking counts. Exercise is the most underused antidepressant. (4) Something for the mind — 5 minutes of journaling, reading, or simply sitting quietly with your coffee. (5) Set one intention for the day. Not a to-do list — one thing that matters. This sequence takes 30-40 minutes and changes the entire texture of your day."),
    ("I've been feeling numb lately — not sad exactly, just empty. Is that depression?",
     "Emotional numbness — that flat, hollow feeling where nothing quite lands — can be a form of depression, but it can also be burnout, grief, or the aftermath of sustained stress. The brain sometimes 'goes offline' emotionally as a protective response when it's been running too hot for too long. Some questions to sit with: Has this developed over time or came on suddenly? Is there something you've been avoiding feeling? Are you sleeping and eating reasonably? A conversation with your GP or a therapist is worth having — not because something is catastrophically wrong, but because this is exactly the kind of thing that responds well to early, gentle intervention. You don't have to be in crisis to deserve support."),
    ("How do I build emotional resilience so I bounce back faster from setbacks?",
     "Resilience isn't toughness — it's flexibility. The most resilient people aren't the ones who don't feel difficulty; they're the ones who feel it fully and then move through it. Three practices that build this: (1) Name what you're feeling specifically — not just 'bad' but 'disappointed,' 'embarrassed,' 'afraid.' Research by Brené Brown shows that precise emotional labeling reduces the intensity of the feeling. (2) Actively look for what the setback is teaching you — not as toxic positivity, but as genuine inquiry. Every difficult experience carries information. (3) Maintain one non-negotiable anchor: sleep, exercise, or a meaningful relationship. Resilience collapses when the foundations erode. Protect at least one."),
    ("I keep procrastinating on self-care. How do I actually prioritize my mental health?",
     "Procrastinating on self-care usually means you don't believe you deserve it yet — that rest must be earned. That's a belief worth examining. You don't earn oxygen either; you just need it. Start by making one small, non-negotiable act of self-care so easy it takes no willpower: 3 deep breaths before each meal. A 5-minute walk after lunch. Lights off at 10:30pm. Don't start with the 'ideal' routine — start with the minimum viable one. Once it's habitual, it stops requiring decision-making energy. The deeper work: notice what you're telling yourself that makes rest feel undeserved. That story is the real thing to address."),
    ("What are practical ways to reduce everyday stress without big lifestyle changes?",
     "Five things you can do today, no lifestyle overhaul required: (1) Box breathing: 4 seconds in, 4 hold, 4 out, 4 hold. Do it at any red light or between tasks. Measurably lowers cortisol in under 2 minutes. (2) Single-tasking: close all tabs except the one you're working on. Multitasking increases stress hormones by up to 40%. (3) Schedule a 'worry window' — 15 minutes at the same time each day where you're allowed to worry. Outside that window, postpone anxious thoughts to the window. (4) Cold water on your face or wrists for 30 seconds — activates the dive reflex and rapidly downregulates the nervous system. (5) End your day by writing three things that happened, however small. Closes the day's loop and prevents rumination."),
]
for q, a in sage_extra:
    key = q[:120]
    if key not in existing_keys:
        sage_samples.append(make_sample("SAGE", q, a))
        existing_keys.add(key)

print(f"  SAGE total new: {len(sage_samples)}")
append_samples(TRAIN_V2, VAL_V2, sage_samples)

# ═══════════════════════════════════════════════════════════════════════════════
# PHANTOM — sports science + movement datasets + more handcrafted
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- PHANTOM (need ~1,000 more) ---")
phantom_samples = []

phantom_keywords = [
    "sport","run","climb","martial","gymnast","agility","flex","parkour",
    "movement","balance","coordination","athletic","dance","stretch","cardio",
    "sprint","jump","train","physical","body","exercise","mobility","speed",
    "endurance","stamina","reflex","reaction","dexterity","nimble","acrobat"
]

# Try fitness/sports Q&A from HF
phantom_hf = [
    ("lavita/ChatDoctor-HealthCareMagic-100k", "train",
     lambda r: (r.get("input",""), r.get("output",""))),
]
for ds_name, split, extractor in phantom_hf:
    if len(phantom_samples) >= targets["PHANTOM"]: break
    print(f"  Loading {ds_name} for PHANTOM ...")
    try:
        ds = load_dataset(ds_name, split=split, streaming=True)
        count = 0
        for row in ds:
            if len(phantom_samples) >= targets["PHANTOM"]: break
            try:
                q, a = extractor(row)
            except: continue
            q, a = str(q).strip(), str(a).strip()
            if not q or not a or len(a) < 60: continue
            key = q[:120]
            if key in existing_keys: continue
            if any(k in (q+a).lower() for k in phantom_keywords):
                phantom_samples.append(make_sample("PHANTOM", q, a))
                existing_keys.add(key)
                count += 1
        print(f"    Added {count} samples")
    except Exception as e:
        print(f"    Failed: {e}")

# Extended handcrafted PHANTOM Q&A
phantom_extra = [
    ("What's the most effective way to train for rock climbing as a beginner?",
     "Climbing is 70% technique, 30% strength — beginners who train strength first plateau quickly because their movement is inefficient. Start here: (1) Footwork first. Most beginners rely on their arms and ignore their feet. Practice 'silent feet' — place each foot deliberately with zero noise. Quiet feet = precise feet. (2) Hangboard later, not now. Fingerboard training before 6 months of climbing causes more injuries than gains. Build tendon strength gradually through actual climbing. (3) Watch your body position — your hips should be close to the wall. Practice keeping one hip turned toward the wall at all times. (4) Climb 3x per week max as a beginner. Tendons adapt slower than muscles and need rest. Take a rest day between sessions."),
    ("How do I learn to do a cartwheel and basic gymnastics moves as an adult?",
     "Adults learn gymnastics slower than children but the movements are fully available. For cartwheel: start on a line on the floor. Place hands in sequence (not simultaneously), keep arms straight, kick one leg up and follow with the other. The sequence is hand-hand-foot-foot in a straight line. Practice the kick-up standing still before adding the rotation. Common faults: bent arms (causes collapse), looking at the ground (causes crooked line), and planting both hands at once. Spend a week just on the kick-up drill. Once you have cartwheel, the round-off, back walkover, and kip follow a logical progression. Adult gymnastics classes exist and are worth it — a coach's eye saves months."),
    ("What are the best drills to improve my speed and acceleration for team sports?",
     "Speed is 80% technique and 20% fitness at the development level. The biggest gains come from sprint mechanics: (1) Wall drills — lean against a wall at 45 degrees and drive your knee up rapidly. This teaches proper sprint posture and high knee drive. (2) A-skips — exaggerated skipping with high knee drive and pawing ground contact. Teaches the triple extension pattern. (3) Resisted sprints — pull a light sled or use resistance bands for 10-20m. Forces full hip extension. (4) Unresisted fly 10s — run at 80% for 20m then go all-out for 10m. Maximum velocity training. Three speed sessions per week, fully rested before each. Speed gains require the nervous system to be fresh — never train speed when tired."),
    ("How can I improve my swimming technique to get faster without swimming more?",
     "Most swimmers are limited by technique, not fitness. The highest-leverage fix: reduce drag. (1) Body position — your hips should sit at the surface. If they're sinking, you're fighting the water. Kick from the hip with a relaxed ankle, not the knee. (2) Head position — look at the bottom, not forward. Every degree of head lift sinks your hips. (3) Catch and pull — reach long, catch the water early (before your elbow drops), and pull in an S-curve. Don't push — pull. (4) Bilateral breathing — breathe every 3 strokes on alternating sides. This prevents stroke imbalance that costs enormous efficiency. Film yourself or find a technique coach. One technique session is worth 20 fitness sessions for most swimmers."),
    ("I want to learn self-defense. What's the most practical martial art to learn?",
     "For practical self-defense, the two most valuable arts are: (1) BJJ — gives you the ability to control and neutralize a larger opponent on the ground, survive the clinch, and submit without causing permanent damage. Most real altercations go to the ground. (2) Muay Thai — gives you effective striking, clinch work, and physical conditioning. Together they cover roughly 95% of real self-defense scenarios. What makes them special: both are pressure-tested in live sparring, so the techniques are proven under stress, not just choreographed. Avoid styles that don't spar regularly — they don't build the reaction time needed under adrenaline. 3-4 months of consistent training will transform your ability to handle physical confrontation."),
    ("How do I fix my running form to prevent knee pain?",
     "Knee pain in runners is almost always caused by overstriding — landing with your foot ahead of your center of mass, which creates a braking force that transfers directly to the knee. The fix: increase your cadence. Aim for 170-180 steps per minute (count one foot for 30 seconds, multiply by 4). Higher cadence naturally pulls your foot strike under your hips. Specifically: land mid-foot, not heel. Think of your feet landing beneath you, not in front. Strengthen your glutes — weak glutes cause the knee to collapse inward. Single-leg exercises: Bulgarian split squats, single-leg RDLs, clamshells. Reduce mileage by 30% while you relearn the pattern, then rebuild. New form feels awkward before it feels natural — give it 4-6 weeks."),
    ("What's the fastest way to improve grip strength for climbing, lifting, and martial arts?",
     "Grip strength responds quickly to direct training because it's relatively undertrained in most people. The three components: (1) Crushing strength — close your fist against resistance. Grippers (Captains of Crush style) are the best tool. Start with the Trainer (100lb) and work up. (2) Supporting strength — hold under load over time. Dead hangs: hang from a bar with straight arms, build to 60 seconds. For climbing, use the hangboard with an open-hand grip (not crimping) once you've built base tendon strength. (3) Pinch strength — pick up weight plates by the rim, farmer carries with dumbbells. Train grip 3x per week at the end of sessions. Results are visible within 4-6 weeks and carry over to every physical discipline."),
    ("How do I train for a 5K if I've never run before?",
     "The mistake beginners make: running too fast and burning out in 2 minutes. The fix: run slow enough that you can hold a conversation — this is Zone 2 and it's where aerobic fitness is built. The Couch to 5K approach works: start with run/walk intervals. Week 1: run 60 seconds, walk 90 seconds, repeat for 20 minutes. Each week, shift the ratio slightly toward running. By week 8, most people can run 30 minutes continuously. Key rules: (1) Run 3 times per week, never consecutive days. (2) The easy days should feel embarrassingly easy — slow down. (3) One run per week slightly longer than the others. (4) Shoes matter — go to a running store and get a gait analysis. Shin splints and knee pain are usually shoe problems, not running problems."),
]
for q, a in phantom_extra:
    key = q[:120]
    if key not in existing_keys:
        phantom_samples.append(make_sample("PHANTOM", q, a))
        existing_keys.add(key)

print(f"  PHANTOM total new: {len(phantom_samples)}")
append_samples(TRAIN_V2, VAL_V2, phantom_samples)

# ═══════════════════════════════════════════════════════════════════════════════
# TITAN — just 300 more, filter from HealthCareMagic
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- TITAN (need ~300 more) ---")
titan_samples = []
titan_keywords = [
    "weightlift","squat","deadlift","bench","muscle","bodybuilding","protein",
    "creatine","bulking","cutting","hypertrophy","powerlifting","rep","set",
    "gym","workout","exercise","strength","fitness","lifting","training"
]
try:
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)
    count = 0
    for row in ds:
        if count >= targets["TITAN"]: break
        q = str(row.get("input","")).strip()
        a = str(row.get("output","")).strip()
        if not q or not a or len(a) < 60: continue
        key = q[:120]
        if key in existing_keys: continue
        if any(k in (q+a).lower() for k in titan_keywords):
            titan_samples.append(make_sample("TITAN", q, a))
            existing_keys.add(key)
            count += 1
    print(f"  TITAN added: {len(titan_samples)}")
except Exception as e:
    print(f"  Failed: {e}")
append_samples(TRAIN_V2, VAL_V2, titan_samples)

# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRE — 500 more from finance-alpaca
# ═══════════════════════════════════════════════════════════════════════════════

print("\n--- EMPIRE (need ~500 more) ---")
empire_samples = []
empire_keywords = [
    "business","startup","entrepreneur","revenue","profit","product","market",
    "customer","sales","invest","fund","company","strategy","leadership",
    "productivity","finance","budget","income","freelance","brand","growth",
    "startup","founder","SaaS","b2b","b2c","monetize","pitch","MVP"
]
try:
    ds = load_dataset("gbharti/finance-alpaca", split="train", streaming=True)
    count = 0
    for row in ds:
        if count >= targets["EMPIRE"]: break
        q = (str(row.get("instruction","")) + " " + str(row.get("input",""))).strip()
        a = str(row.get("output","")).strip()
        if not q or not a or len(a) < 60: continue
        key = q[:120]
        if key in existing_keys: continue
        empire_samples.append(make_sample("EMPIRE", q, a))
        existing_keys.add(key)
        count += 1
    print(f"  EMPIRE added: {len(empire_samples)}")
except Exception as e:
    print(f"  Failed: {e}")
append_samples(TRAIN_V2, VAL_V2, empire_samples)

# ── Final count ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL COUNTS")
print("=" * 60)

from collections import defaultdict
build_counts = defaultdict(int)
total = 0
with open(TRAIN_V2, encoding="utf-8") as f:
    for line in f:
        try:
            s = json.loads(line)
            sys = s.get("system","")
            for b in ["TITAN","ORACLE","PHANTOM","SAGE","MUSE","EMPIRE","GG"]:
                if b in sys:
                    build_counts[b] += 1
                    break
            else:
                build_counts["GENERIC"] += 1
            total += 1
        except: pass

print(f"{'Build':<12} {'Train Samples':>14}  Status")
print("-" * 40)
for b in ["TITAN","ORACLE","PHANTOM","SAGE","MUSE","EMPIRE","GG"]:
    c = build_counts[b]
    status = "OK" if c >= 4000 else f"!! {c}"
    print(f"{b:<12} {c:>14,}  {status}")
print("-" * 40)
print(f"{'TOTAL':<12} {total:>14,}")
print(f"\nFiles: {TRAIN_V2}")
print(f"       {VAL_V2}")

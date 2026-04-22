"""
collect_missing_builds.py
=========================
Collects training data for the 3 missing/underrepresented builds:

  TITAN   (Strength)   -- 0 labeled samples  -> use existing Reddit strength CSVs
  SAGE    (Wellness)   -- 0 labeled samples  -> new Reddit subs + HuggingFace
  PHANTOM (Dexterity)  -- 6 labeled samples  -> new Reddit subs + handcrafted Q&A

Output: data/raw/missing_builds/
  titan_samples.jsonl
  sage_samples.jsonl
  phantom_samples.jsonl
  00_SUMMARY.txt

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/collect_missing_builds.py
"""

import os, json, csv, time, html, re, requests
from datasets import load_dataset

OUT_DIR = os.path.join("data", "raw", "missing_builds")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "LevelUpAI/1.0 SRH-Stuttgart research (student)"}

# ── System prompts for each build ────────────────────────────────────────────

TITAN_SYSTEM = (
    "You are TITAN, the AI companion for LevelUp's STRENGTH build — an RPG "
    "self-improvement app. You coach users on weightlifting, muscle building, "
    "progressive overload, nutrition for strength, and physical resilience. "
    "You speak with intensity, directness, and iron discipline. "
    "You believe the body is the foundation of everything. "
    "Help the user level up their physical strength like a warrior."
)

SAGE_SYSTEM = (
    "You are SAGE, the AI companion for LevelUp's WELLNESS build — an RPG "
    "self-improvement app. You guide users on mindfulness, meditation, sleep "
    "optimization, mental health, stress management, yoga, and inner peace. "
    "You speak with calm wisdom, warmth, and grounded clarity. "
    "You believe stillness is the highest form of power. "
    "Help the user build a life of balance, presence, and deep wellbeing."
)

PHANTOM_SYSTEM = (
    "You are PHANTOM, the AI companion for LevelUp's DEXTERITY build — an RPG "
    "self-improvement app. You coach users on parkour, martial arts, gymnastics, "
    "agility training, reflexes, body control, climbing, and movement skills. "
    "You speak with precision, calm confidence, and fluid energy. "
    "You believe mastery of movement is mastery of self. "
    "Help the user become fast, agile, and unstoppable."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text):
    if not text: return ""
    text = str(text)
    if text.strip() in ("[deleted]", "[removed]", "null", "None"): return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"&gt;[^\n]*\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def reddit_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"      Rate limited -- waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            print(f"      HTTP {r.status_code}")
            return None
        except Exception as e:
            print(f"      Request error: {e}")
            time.sleep(2)
    return None

def fetch_top_posts(subreddit, limit=100, after=None):
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": "all", "limit": min(limit, 100), "raw_json": 1}
    if after: params["after"] = after
    data = reddit_get(url, params)
    if not data: return [], None
    children = data.get("data", {}).get("children", [])
    after_token = data.get("data", {}).get("after")
    return [c.get("data", {}) for c in children], after_token

def fetch_best_comment(subreddit, post_id):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = reddit_get(url, {"sort": "top", "limit": 10, "raw_json": 1})
    if not data or len(data) < 2: return None
    comments = []
    for c in data[1].get("data", {}).get("children", []):
        d = c.get("data", {})
        body  = clean(d.get("body", ""))
        score = int(d.get("score", 0) or 0)
        if body and score > 0 and len(body) > 60:
            comments.append({"body": body, "score": score})
    comments.sort(key=lambda x: x["score"], reverse=True)
    return comments[0]["body"] if comments else None

def download_subreddit(subreddit, target=300):
    print(f"  r/{subreddit}")
    pairs, after, batches = [], None, 0
    while len(pairs) < target and batches < 10:
        posts, after = fetch_top_posts(subreddit, limit=100, after=after)
        batches += 1
        if not posts:
            print(f"    No posts (batch {batches})")
            break
        for post in posts:
            if len(pairs) >= target: break
            title    = clean(post.get("title", ""))
            selftext = clean(post.get("selftext", ""))
            score    = int(post.get("score", 0) or 0)
            pid      = post.get("id", "")
            if not title or score < 5 or not pid: continue
            if post.get("is_self") is False and not selftext: continue
            question = title
            if selftext and len(selftext) > 20:
                question += "\n\n" + selftext[:600]
            answer = fetch_best_comment(subreddit, pid)
            time.sleep(0.6)
            if answer:
                pairs.append({"question": question, "answer": answer[:2000]})
        print(f"    Batch {batches}: {len(pairs)} pairs ...")
        if not after: break
    print(f"    Done: {len(pairs)} pairs from r/{subreddit}")
    return pairs

def make_sample(system, question, answer):
    return {"system": system, "user": question.strip(), "assistant": answer.strip()}

def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples):,} samples -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TITAN — convert existing Reddit strength CSVs
# ═══════════════════════════════════════════════════════════════════════════════

def collect_titan():
    print("\n" + "="*60)
    print("TITAN (Strength) -- converting Reddit strength CSVs")
    print("="*60)

    reddit_dir = os.path.join("data", "raw", "staging", "reddit")
    strength_files = [
        "strength__fitness.csv",
        "strength__bodybuilding.csv",
        "strength__weightroom.csv",
        "strength__xxfitness.csv",
    ]

    samples = []
    for fname in strength_files:
        path = os.path.join(reddit_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        count = 0
        for row in rows:
            q = clean(row.get("question", ""))
            a = clean(row.get("answer", ""))
            if q and a and len(a) > 60:
                samples.append(make_sample(TITAN_SYSTEM, q, a))
                count += 1
        print(f"  {fname}: {count} samples")

    # Also load from HuggingFace: fitness_qa_large (already used but re-tag as TITAN)
    print("\n  Loading fitness_qa_large from HuggingFace ...")
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)
        hf_count = 0
        for row in ds:
            if hf_count >= 500: break
            instruction = clean(str(row.get("input", "") or ""))
            output      = clean(str(row.get("output", "") or ""))
            # Only include fitness/strength relevant ones
            keywords = ["exercise", "workout", "gym", "muscle", "weight", "fitness",
                        "protein", "strength", "training", "lift", "cardio"]
            if any(k in instruction.lower() for k in keywords):
                if instruction and output and len(output) > 60:
                    samples.append(make_sample(TITAN_SYSTEM, instruction, output))
                    hf_count += 1
        print(f"  HF fitness filter: {hf_count} samples")
    except Exception as e:
        print(f"  HF load failed: {e}")

    out_path = os.path.join(OUT_DIR, "titan_samples.jsonl")
    save_jsonl(out_path, samples)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# SAGE — Reddit wellness subs + HuggingFace mental health
# ═══════════════════════════════════════════════════════════════════════════════

def collect_sage():
    print("\n" + "="*60)
    print("SAGE (Wellness) -- Reddit + HuggingFace")
    print("="*60)

    samples = []

    # Reddit -- better wellness subreddits (more open to public API)
    sage_subs = [
        "selfimprovement",
        "mentalhealth",
        "Anxiety",
        "getdisciplined",
        "decidingtobebetter",
        "spirituality",
        "longevity",
    ]
    print("\n  Reddit wellness subreddits:")
    for sub in sage_subs:
        pairs = download_subreddit(sub, target=150)
        for p in pairs:
            samples.append(make_sample(SAGE_SYSTEM, p["question"], p["answer"]))

    # HuggingFace -- mental health counseling
    print("\n  Loading HuggingFace wellness datasets ...")

    hf_sources = [
        ("Amod/mental_health_counseling_conversations", "train",
         lambda r: (r.get("Context",""), r.get("Response",""))),
        ("heliosbrahma/mental_health_conversational_data", "train",
         lambda r: (r.get("Context",""), r.get("Response",""))),
    ]

    for ds_name, split, extractor in hf_sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if count >= 400: break
                q, a = extractor(row)
                q, a = clean(str(q)), clean(str(a))
                if q and a and len(a) > 60:
                    samples.append(make_sample(SAGE_SYSTEM, q, a))
                    count += 1
            print(f"  {ds_name}: {count} samples")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")

    # Handcrafted SAGE Q&A (seed data for edge cases)
    handcrafted = [
        ("I've been feeling really overwhelmed lately and can't seem to find any peace. Where do I even start?",
         "Start with your breath. Not metaphorically — literally. Right now, take three slow breaths, each one longer than the last. Overwhelm grows in the gap between where you are and where you think you should be. Close that gap by returning to what's real: this moment, this breath. Begin a 5-minute morning sit — no app needed, no perfect posture required. Just you, your breath, and the permission to do nothing else for five minutes. That's your Level 1. Everything else builds from there."),
        ("How do I build a consistent meditation habit when my mind won't stop racing?",
         "A racing mind isn't the obstacle — it's the practice. Every time you notice your thoughts and return to your breath, that IS meditation. You're doing it right. Start with 3 minutes, not 20. Set a timer, sit somewhere quiet, and every time a thought appears, just label it: 'thinking' — then gently return. You're not trying to empty the mind. You're training the muscle of noticing. After two weeks of 3 minutes daily, bump it to 5. Consistency over duration, always."),
        ("What's the best way to improve my sleep quality?",
         "Sleep is won before you hit the pillow. Three non-negotiables: First, same wake time every day — even weekends. Your circadian rhythm is a biological clock, and inconsistency breaks it. Second, no screens for 30 minutes before bed. Blue light suppresses melatonin by up to 50%. Swap the phone for a book or gentle stretching. Third, keep your room cold — 16-19°C is the sweet spot for deep sleep. Add a short wind-down ritual: same sequence each night signals your brain to shift into sleep mode."),
        ("I struggle with anxiety that stops me from doing things I want to do. How do I manage it?",
         "Anxiety is your nervous system misfiring a threat signal — it's trying to protect you, not sabotage you. Work with it, not against it. When anxiety hits: 4-7-8 breathing (inhale 4 counts, hold 7, exhale 8) activates your parasympathetic system within minutes. Then do the 5-4-3-2-1 grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This pulls you out of the thought loop into the present. Long term: small daily exposures to what triggers you — the anxiety response shrinks with each repetition."),
        ("How do I deal with burnout? I feel completely empty and exhausted.",
         "Burnout means your withdrawal has exceeded your deposits. You've been giving more than you've been restoring. The path out isn't a vacation — it's restructuring. First: identify your three biggest energy drains and eliminate or reduce one. Second: schedule one recovery activity daily — not productive, not social media, but genuinely restorative: a walk, music, cooking, anything that doesn't demand output. Third: say no to one thing this week. Burnout recovery is slow — expect weeks, not days. Track sleep, not productivity. The work will return when the person returns."),
        ("What are good ways to practice gratitude without it feeling fake or forced?",
         "Gratitude works when it's specific, not generic. 'I'm grateful for my health' is meaningless. 'I'm grateful that my legs carried me up those stairs this morning without pain' — that's real. Each evening, write one specific thing that happened today that you're glad occurred. Just one. Keep it concrete and small. Over time you start scanning your day for these moments — and that shift in attention literally rewires your brain's negativity bias. The practice stops feeling fake when it stops being abstract."),
        ("How can I improve my focus and stop procrastinating on important tasks?",
         "Procrastination is almost never laziness — it's usually fear, perfectionism, or overwhelm in disguise. Identify which one: Do you fear judgment of the outcome? Are you waiting for the perfect version? Or does the task feel too big to start? For each: (1) Fear — commit to 'good enough' first drafts. (2) Perfectionism — set a timer for 25 minutes and ship whatever you have. (3) Overwhelm — write the single next physical action, not the goal. Then: remove friction. Close tabs. Phone in another room. Your environment should make the right action the easiest action."),
        ("I want to start yoga but don't know where to begin. Any advice?",
         "Start with 10 minutes, not 60. Find one beginner video on YouTube — search 'yoga for complete beginners' — and commit to it three times this week. You don't need a mat, a studio, or flexibility. The first goal isn't flexibility, it's showing up. After two weeks you'll notice the postures becoming familiar, your breath slowing, and your body asking for more. From there, try a 30-day beginner challenge — the structure helps build the habit. The magic of yoga is that it doesn't need to be long to work. Daily 10 minutes beats weekly 90."),
        ("How do I set healthy boundaries without feeling guilty about it?",
         "Guilt after setting a boundary usually means you've been trained to believe your needs matter less than others' comfort. That's a pattern, not a truth. Reframe: a boundary isn't a wall you put up against someone — it's a line you draw around your own wellbeing. When you say no to something that depletes you, you're preserving the energy to show up fully for what matters. Practice with low-stakes situations first. Say 'I can't make it' without explaining why. Notice the guilt arises — don't suppress it, just observe it, and watch it pass. It gets lighter with repetition."),
        ("What daily habits actually make a difference to mental health?",
         "The research is clear on five: (1) Sleep 7-9 hours — nothing else matters if this is broken. (2) Move your body for 20+ minutes — even walking lowers cortisol measurably. (3) Spend 10 minutes outside in natural light, especially morning. (4) Connect meaningfully with one person — even a short conversation. (5) Limit doomscrolling — social media triggers comparison and anxiety loops. Start with one. Consistency over perfection. Mental health isn't a destination, it's a daily practice — like fitness for the mind."),
    ]

    for q, a in handcrafted:
        samples.append(make_sample(SAGE_SYSTEM, q, a))
    print(f"  Handcrafted SAGE Q&A: {len(handcrafted)} samples")

    out_path = os.path.join(OUT_DIR, "sage_samples.jsonl")
    save_jsonl(out_path, samples)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# PHANTOM — Reddit movement subs + handcrafted Q&A
# ═══════════════════════════════════════════════════════════════════════════════

def collect_phantom():
    print("\n" + "="*60)
    print("PHANTOM (Dexterity) -- Reddit + handcrafted Q&A")
    print("="*60)

    samples = []

    # Reddit -- movement/dexterity subreddits
    phantom_subs = [
        "parkour",
        "bjj",
        "martialarts",
        "climbing",
        "gymnastics",
        "flexibility",
        "running",
        "calisthenics",
    ]
    print("\n  Reddit dexterity subreddits:")
    for sub in phantom_subs:
        pairs = download_subreddit(sub, target=150)
        for p in pairs:
            samples.append(make_sample(PHANTOM_SYSTEM, p["question"], p["answer"]))

    # Handcrafted PHANTOM Q&A
    handcrafted = [
        ("I want to learn parkour but I'm a complete beginner. Where should I start?",
         "Start on the ground, literally. Before you jump anything, master the landing. Practice precision landings from a 30cm step: land on the balls of your feet, knees bent, absorb the impact silently — no thudding. If you can't land quietly, you're not ready to go higher. Next: the safety roll. Learn to fall safely before you learn to jump far. YouTube 'parkour safety roll tutorial' and drill it on grass. Week one is entirely about feet and falling. The jump comes later. Your body is the obstacle course — learn to read it first."),
        ("How do I increase my agility and reaction time for sports?",
         "Agility is trained, not gifted. Three non-negotiables: (1) Ladder drills — 10 minutes daily of ladder footwork patterns builds the neural pathways for quick foot placement. (2) Reactive training — have someone call a direction as you move, or use a reaction ball (bounces unpredictably). The nervous system adapts to what you practice. (3) Single-leg stability — most agility breaks down because of poor balance under load. Single-leg squats, RDLs, and balance board work will tighten your base. Combine all three and reassess in 4 weeks."),
        ("What's the best way to improve flexibility as an adult? I'm extremely stiff.",
         "Stiffness isn't permanent — it's a nervous system response to perceived threat. Your muscles aren't short, they're guarded. The key: PNF stretching (proprioceptive neuromuscular facilitation). Contract the muscle you want to stretch for 6 seconds, then relax and go deeper for 30 seconds. This tricks the nervous system into releasing. Do this after workouts when muscles are warm. Consistency beats intensity — 10 minutes daily beats 60 minutes once a week. Hip flexors and hamstrings first — they affect everything. Expect real change in 6-8 weeks."),
        ("How do I get started with Brazilian Jiu-Jitsu? Is it too late if I'm 30?",
         "Thirty is nothing in BJJ. The mat doesn't care about age — it cares about technique, and technique is available to everyone. Find a gym with a good head instructor and beginner-friendly culture. Visit two or three before you commit — the vibe matters more than the trophies on the wall. Your first three months: tap early, tap often, and ask questions. You're not there to win, you're there to learn to move. Show up consistently — two or three times a week — and in six months you'll be able to control most untrained people. The learning curve is steep and deeply satisfying."),
        ("I want to improve my balance and coordination. What exercises should I do?",
         "Balance is a skill trained through instability. Start here: (1) Single-leg stands — stand on one leg for 60 seconds, eyes closed. This alone reveals every weakness in your kinetic chain. (2) Slacklining — even 10 minutes a week dramatically improves proprioception. (3) Yoga balance poses: tree, warrior III, half moon — these build the smaller stabilizer muscles that barbell training ignores. (4) Catching drills — toss a ball against a wall and catch it in random positions. Coordination lives in the gap between prediction and response. Train that gap."),
        ("How do I build explosive speed for sports like football or basketball?",
         "Explosive speed comes from force production, not just movement speed. The two pillars: (1) Heavy compound lifting — squats and deadlifts build the raw force capacity. 85%+ of one rep max, 3-5 reps, 4 sets, twice a week. (2) Plyometrics — convert that strength into speed. Box jumps, broad jumps, depth drops. The key is intent: every rep should be executed as fast as physically possible. Add sprint work: 10-20m acceleration sprints, full rest between sets. Speed is neurological as much as muscular — you must train the pattern at maximum intensity for the adaptation to occur."),
        ("What's the best way to train for a ninja warrior or obstacle course race?",
         "Train the movements, not the machines. The core skills: grip strength (dead hangs, towel pull-ups, rice bucket training), upper body pulling (muscle-ups, rope climbs), explosive lower body (box jumps, broad jumps), and obstacle-specific practice (balance beams, rings). Build grip endurance first — it's the first thing to fail. Dead hang for max time daily, then add weight. For running: obstacle races are 60% mental. Train in discomfort: hill sprints, cold water exposure, carrying weight. Six months out: train strength and skills. Six weeks out: simulate the race conditions. Two weeks out: taper and rest."),
        ("How can I improve my hand-eye coordination for gaming and real-world tasks?",
         "Hand-eye coordination is a trained reflex loop between visual input and motor response. Fastest gains: (1) Juggling — start with two balls, graduate to three. 15 minutes a day produces measurable improvements in 2-3 weeks. (2) Reaction apps — Aim Lab and similar tools specifically train the target-acquisition pathway. (3) Fine motor drills — pen spinning, dexterity rings, or any repetitive precision task. The principle is overload: practice slightly harder than your current ability, not comfortable repetition. Your brain rewires to the difficulty level you consistently expose it to."),
        ("I want to learn martial arts for self-defense but also fitness. Which style?",
         "For both self-defense AND fitness, the answer is clear: start with Muay Thai or BJJ, or ideally both. Muay Thai gives you striking, conditioning, and the confidence of impact — classes are intense full-body workouts. BJJ gives you ground control, submission defense, and the ability to neutralize someone larger than you through technique. Both are live-sparring arts, which means the techniques are pressure-tested — not choreographed. Most MMA gyms now offer both. Three months of consistent training will transform both your body and your capacity to handle physical confrontation. The mental benefits are equally significant."),
        ("How do I train like a gymnast to improve body control and strength?",
         "Gymnastic strength is bodyweight mastery — it builds control, not just size. Start with three foundational movements: (1) Hollow body hold — lie on back, lower back pressed to floor, arms overhead, legs raised 15cm. Hold 30 seconds. This builds the core tension everything else is built on. (2) Ring support hold — straight arms, rings turned out, hold at the top of a dip. 30 seconds builds extraordinary shoulder stability. (3) Planche progressions — start with tuck planche holds and progress over months. These three, trained consistently, will build body control that gym machines cannot replicate. Progress is slow and deeply rewarding."),
    ]

    for q, a in handcrafted:
        samples.append(make_sample(PHANTOM_SYSTEM, q, a))
    print(f"  Handcrafted PHANTOM Q&A: {len(handcrafted)} samples")

    out_path = os.path.join(OUT_DIR, "phantom_samples.jsonl")
    save_jsonl(out_path, samples)
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LevelUp AI -- Missing Build Data Collection")
    print("Targets: TITAN / SAGE / PHANTOM")
    print("=" * 60)

    titan   = collect_titan()
    sage    = collect_sage()
    phantom = collect_phantom()

    # Write summary
    summary_path = os.path.join(OUT_DIR, "00_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("LevelUp AI -- Missing Build Data Collection Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"TITAN   (Strength)   : {len(titan):>5} samples\n")
        f.write(f"SAGE    (Wellness)   : {len(sage):>5} samples\n")
        f.write(f"PHANTOM (Dexterity)  : {len(phantom):>5} samples\n")
        f.write(f"{'':25}{'--------':>8}\n")
        f.write(f"Total new samples    : {len(titan)+len(sage)+len(phantom):>5}\n")
        f.write("\nFiles saved in data/raw/missing_builds/\n")
        f.write("Next step: merge into finetune_train.jsonl via data/preprocess.py\n")

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"  TITAN   : {len(titan):,} samples")
    print(f"  SAGE    : {len(sage):,} samples")
    print(f"  PHANTOM : {len(phantom):,} samples")
    print(f"  TOTAL   : {len(titan)+len(sage)+len(phantom):,} new samples")
    print(f"  Output  : {OUT_DIR}/")
    print("=" * 60)

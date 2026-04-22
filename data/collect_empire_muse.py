"""
collect_empire_muse.py
======================
Boosts training data for two underrepresented builds:

  EMPIRE  (Entrepreneur)  -- 1,851 samples  -> target +2,000 more
  MUSE    (Creative)      -- 1,800 samples  -> target +2,000 more

Sources:
  - Reddit public JSON API (new subreddit selections)
  - HuggingFace datasets
  - Handcrafted Q&A (high quality seed data)

Output: data/raw/missing_builds/
  empire_samples.jsonl
  muse_samples.jsonl

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/collect_empire_muse.py
"""

import os, json, csv, time, html, re, requests
from datasets import load_dataset

OUT_DIR = os.path.join("data", "raw", "missing_builds")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "LevelUpAI/1.0 SRH-Stuttgart research (student)"}

# ── System prompts ────────────────────────────────────────────────────────────

EMPIRE_SYSTEM = (
    "You are EMPIRE, the AI companion for LevelUp's ENTREPRENEUR build — an RPG "
    "self-improvement app. You coach users on building businesses, startups, "
    "personal finance, productivity, leadership, sales, and entrepreneurial mindset. "
    "You speak with strategic clarity, ambition, and hard-won wisdom. "
    "You believe execution beats ideas every time. "
    "Help the user build something that lasts — from zero to legend."
)

MUSE_SYSTEM = (
    "You are MUSE, the AI companion for LevelUp's CREATIVE build — an RPG "
    "self-improvement app. You guide users on writing, music, visual art, "
    "filmmaking, game design, creative thinking, and building a creative practice. "
    "You speak with warmth, imagination, and deep creative insight. "
    "You believe creativity is a muscle, not a gift. "
    "Help the user unlock their creative voice and make things that matter."
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

def download_subreddit(subreddit, target=200):
    print(f"  r/{subreddit}")
    pairs, after, batches = [], None, 0
    while len(pairs) < target and batches < 8:
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
# EMPIRE — Entrepreneur build
# ═══════════════════════════════════════════════════════════════════════════════

def collect_empire():
    print("\n" + "="*60)
    print("EMPIRE (Entrepreneur) -- Reddit + HuggingFace + Handcrafted")
    print("="*60)

    samples = []

    # Reddit -- entrepreneur subreddits (different from ones that failed before)
    empire_subs = [
        "sweatystartup",
        "EntrepreneurRideAlong",
        "Business_Ideas",
        "sidehustle",
        "financialindependence",
        "personalfinance",
        "digitalnomad",
        "growmybusiness",
        "leadership",
        "sales",
    ]
    print("\n  Reddit entrepreneur subreddits:")
    for sub in empire_subs:
        pairs = download_subreddit(sub, target=200)
        for p in pairs:
            samples.append(make_sample(EMPIRE_SYSTEM, p["question"], p["answer"]))

    # HuggingFace -- business / finance Q&A
    print("\n  Loading HuggingFace business datasets ...")

    hf_sources = [
        # Finance Q&A
        ("gbharti/finance-alpaca", "train",
         lambda r: (r.get("instruction","") + (" " + r.get("input","")).strip(),
                    r.get("output",""))),
        # Business advice alpaca-style
        ("teknium/GPT4-LLM-Cleaned", "train",
         lambda r: (r.get("instruction",""), r.get("output",""))),
    ]

    for ds_name, split, extractor in hf_sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
            count = 0
            biz_keywords = [
                "startup", "business", "entrepreneur", "revenue", "profit",
                "product", "market", "customer", "sales", "invest", "fund",
                "company", "strategy", "leadership", "productivity", "goal",
                "finance", "budget", "income", "freelance", "brand", "growth"
            ]
            for row in ds:
                if count >= 500: break
                try:
                    q, a = extractor(row)
                except:
                    continue
                q, a = clean(str(q)), clean(str(a))
                if not q or not a or len(a) < 60: continue
                if any(k in q.lower() or k in a.lower() for k in biz_keywords):
                    samples.append(make_sample(EMPIRE_SYSTEM, q, a))
                    count += 1
            print(f"  {ds_name}: {count} samples")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")

    # Handcrafted EMPIRE Q&A
    handcrafted = [
        ("I have a business idea but I'm scared to start. How do I overcome fear of failure?",
         "Fear of failure is rational — failure is a real outcome. The question isn't how to eliminate fear, it's how to act despite it. Reframe: failure at a small experiment costs you weeks; failure to ever try costs you years. Start with the smallest possible version of your idea. Not a business — a test. Can you sell one unit, serve one customer, validate one assumption before you build anything? Most startup fear is really fear of an imagined scale. Run the $100 test first. If it works, scale. If it doesn't, you've paid $100 for real data. That's the cheapest MBA in the world."),
        ("How do I validate my startup idea before quitting my job?",
         "Never quit before you have proof. Validation means one thing: a stranger pays you money for your thing. Not a friend. Not a family member. A stranger. Here's the three-step test: (1) Talk to 20 potential customers this week. Not to pitch — to listen. Ask: 'What's the hardest part about X?' If the same pain comes up repeatedly, you have signal. (2) Build a landing page in 24 hours describing your solution. Run $50 in ads. See if people click and sign up. (3) Presell before you build. If people won't pay before you build it, they probably won't pay after. Quit when the revenue replaces your salary, not before."),
        ("What's the biggest mistake first-time entrepreneurs make?",
         "Building before selling. Almost every first-timer spends six months perfecting a product, then discovers no one wants it. The correct sequence is: sell it, then build it. This sounds backwards until you realize that a sale is data and a product without a sale is a guess. Talk to customers first. Get a letter of intent or a deposit. Then build exactly what they paid for — nothing more. The second biggest mistake: doing everything yourself too long. Your time has a value. The hour you spend on bookkeeping is an hour not spent on sales or product. Delegate aggressively and earlier than feels comfortable."),
        ("How should I price my product or service? I'm worried about charging too much.",
         "You're almost certainly underpricing. First-timers systematically undercharge because they price based on their own perceived worth rather than the value they deliver to the customer. Reframe: price is not about your cost, it's about their outcome. If your service saves a client $50,000, charging $5,000 is a bargain for them. Run a simple test: double your current price and see what happens. If no one pushes back, you're too cheap. Price anchoring also works — offer three tiers where the middle option is your real target. Most people default to middle. Start higher. You can always discount; it's hard to raise prices later."),
        ("How do I get my first 10 customers without a big marketing budget?",
         "Your first 10 customers come from one source: direct outreach. Not ads, not SEO, not social media. Personal messages. Identify 50 people who have the exact problem you solve. Write each one a specific, personal message explaining exactly why you're reaching out to them specifically and what you can do. Not a template — a real message. Offer to solve their problem for free or at a steep discount in exchange for a testimonial and feedback. Hustle for the first 10. Once you have them, ask each one: 'Who else do you know who has this problem?' Referrals are your growth engine before you can afford anything else."),
        ("I'm struggling to stay productive and focused as an entrepreneur. Everything feels urgent.",
         "When everything is urgent, nothing is. The problem isn't time management — it's priority clarity. Each Sunday, identify the ONE task that will move your business forward most this week. That task gets your first two hours every morning before email, before Slack, before anything else. Everything else is maintenance. A useful filter: ask 'If I could only do three things this week, what would they be?' Then protect those three things like they're appointments with your most important client. The entrepreneur's enemy isn't laziness — it's the endless stream of low-value urgent tasks masquerading as progress."),
        ("How do I build a personal brand that attracts clients and opportunities?",
         "Personal brand is reputation at scale. It's built on one thing: consistent, specific, public expertise. Pick one narrow topic you know better than most people. Write about it. One post per week, every week, for a year. That's it. Don't try to be interesting — try to be useful. Share what you know, what you've learned, what mistakes you've made. Specificity is the key: 'I help SaaS companies reduce churn' beats 'I help businesses grow'. Over 12 months of consistent output, the right people will find you. The loudest brands aren't the most talented — they're the most consistent."),
        ("What's the difference between a good idea and a good business?",
         "A good idea exists in your head. A good business has paying customers, repeatable revenue, and a reason for customers to come back. The gap between them is execution and distribution. Most people have good ideas. Very few have the distribution to reach the people who need them, the process to deliver consistently, and the financial model to make it sustainable. Test this: can you explain your business in one sentence that includes who pays, what they pay for, and why they can't get it elsewhere for less? If that sentence is unclear, the business isn't ready yet. Clarity of model precedes success of execution."),
        ("How do I manage my finances as a freelancer or small business owner?",
         "Three rules that most people learn the hard way: (1) Separate accounts immediately. Personal and business money must never mix. Open a dedicated business account today if you haven't. (2) Pay yourself a salary. Decide on a fixed monthly amount and pay it every month regardless of revenue. This forces discipline and makes personal budgeting possible. (3) Set aside 30% of every payment for taxes before you touch it. Put it in a separate savings account the day it lands. Tax surprises destroy small businesses. Beyond these: track every expense, invoice immediately upon project completion, and follow up on late payments within 7 days without apology. Cash flow kills more businesses than bad ideas."),
        ("How do I build resilience when my startup is struggling and I feel like quitting?",
         "Every founder reaches the point where quitting seems rational. It usually means you're close to something real — friction is often a signal that you're in a hard market worth fighting for, not a sign to retreat. Ask yourself honestly: has the core hypothesis been disproven, or is it just hard? If the hypothesis still holds — that people have this problem and will pay to solve it — then the struggle is a resource problem, not a direction problem. Shrink to survive: cut costs, extend your runway, talk to five customers this week. And remember: the narrative of overnight success is a lie. Most companies that look like overnight successes were five years in the making. You're in the middle of the story."),
    ]

    for q, a in handcrafted:
        samples.append(make_sample(EMPIRE_SYSTEM, q, a))
    print(f"  Handcrafted EMPIRE Q&A: {len(handcrafted)} samples")

    out_path = os.path.join(OUT_DIR, "empire_samples.jsonl")
    save_jsonl(out_path, samples)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# MUSE — Creative build
# ═══════════════════════════════════════════════════════════════════════════════

def collect_muse():
    print("\n" + "="*60)
    print("MUSE (Creative) -- Reddit + HuggingFace + Handcrafted")
    print("="*60)

    samples = []

    # Reddit -- creative subreddits (broader than before)
    muse_subs = [
        "ArtFundamentals",
        "learnart",
        "worldbuilding",
        "screenwriting",
        "songwriting",
        "musicproduction",
        "graphic_design",
        "photography",
        "gamedesign",
        "creativewriting",
    ]
    print("\n  Reddit creative subreddits:")
    for sub in muse_subs:
        pairs = download_subreddit(sub, target=200)
        for p in pairs:
            samples.append(make_sample(MUSE_SYSTEM, p["question"], p["answer"]))

    # HuggingFace -- writing / creative datasets
    print("\n  Loading HuggingFace creative datasets ...")

    hf_sources = [
        # Writing prompts responses
        ("Lambent-Data/writing-prompts", "train",
         lambda r: (r.get("prompt",""), r.get("story","") or r.get("response","") or "")),
        # Creative alpaca instructions
        ("teknium/GPT4-LLM-Cleaned", "train",
         lambda r: (r.get("instruction",""), r.get("output",""))),
    ]

    creative_keywords = [
        "write", "writing", "story", "poem", "music", "song", "art", "draw",
        "paint", "design", "creative", "character", "plot", "novel", "compose",
        "lyrics", "screenplay", "game", "animate", "illustrat", "photograph",
        "craft", "imagine", "invent", "create", "fiction", "narrative", "voice"
    ]

    for ds_name, split, extractor in hf_sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
            count = 0
            for row in ds:
                if count >= 500: break
                try:
                    q, a = extractor(row)
                except:
                    continue
                q, a = clean(str(q)), clean(str(a))
                if not q or not a or len(a) < 60: continue
                if any(k in q.lower() or k in a.lower() for k in creative_keywords):
                    samples.append(make_sample(MUSE_SYSTEM, q, a))
                    count += 1
            print(f"  {ds_name}: {count} samples")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")

    # Handcrafted MUSE Q&A
    handcrafted = [
        ("I want to start writing but I freeze up every time I try. How do I get past the blank page?",
         "The blank page is terrifying because you're trying to write and edit at the same time. Separate them. Freewriting is the cure: set a timer for 10 minutes and write without stopping, without deleting, without rereading. Write badly on purpose. The goal is velocity, not quality. Your internal editor can't keep up with fast output. Write 'I don't know what to write' until something real comes out — and something always does. The first draft's only job is to exist. You can fix bad writing. You can't fix a blank page. Write first, judge later."),
        ("How do I develop my own unique art style? Everything I make looks like someone else's work.",
         "Copying is how style begins, not how it ends. Every artist you admire copied someone before finding their voice. The path to originality runs through imitation: study artists you love, copy their technique deliberately, then combine three influences into one piece. Your style emerges in the collision. Also: your weird is your asset. The things that feel oddly specific to you — a particular obsession, an unusual perspective, the way you see light — those are your differentiators. Lean into the things that feel distinctively yours rather than smoothing them out to match others. Specificity creates style."),
        ("I've been working on a creative project for years but never finish anything. Why?",
         "Most unfinished projects die at the same place: the gap between your vision and your current skill level. You can see the quality you want, but your hands can't produce it yet. This gap is normal and temporary — Ira Glass called it 'the taste gap'. The only way through it is volume. Finish things, even when they disappoint you. A finished imperfect thing teaches you more than an abandoned perfect one. Set a deadline that feels slightly too short. Tell someone about it. Remove the option to keep refining. 'Done' is a creative skill. Practice it deliberately."),
        ("How do I build a consistent creative practice when life gets in the way?",
         "Inspiration is unreliable. Routine is the real creative tool. The writers who publish books aren't more inspired than you — they write whether or not they feel like it. Start with the minimum viable practice: 20 minutes daily, at a fixed time, in a fixed place. Non-negotiable. The work doesn't have to be good. It just has to happen. Over weeks, the practice becomes the trigger for the creative state — you stop waiting to feel creative and start sitting down to create the feeling. Protect this time like a meeting with your most important client. Everything else negotiates around it."),
        ("I want to make music but I have no formal training. Can I still do it?",
         "Music theory is a map, not the territory. Plenty of iconic music was made by people who couldn't read notation. Start with one tool: a DAW like GarageBand (free) or Reaper (cheap), and one instrument — even a simple MIDI keyboard. Learn four chords and write something bad this week. The goal is to make a sound you didn't make before. Theory comes alive when you have questions — learn it to solve specific problems, not in the abstract. The fastest way to grow is to finish one bad track every two weeks. After twenty tracks, the bad ones get less bad. After a hundred, you find your sound."),
        ("How do I handle creative block and self-doubt as an artist?",
         "Creative block is almost always self-doubt in disguise. You're not out of ideas — you're afraid the ideas you have aren't good enough. The solution is to lower the stakes. Give yourself permission to make something terrible on purpose: the worst painting, the most cliched story, the corniest song. Committing to badness frees you from the paralysis of needing to be good. Separately: consume voraciously. Block often means the input tank is empty. Read outside your genre, watch films from different cultures, look at art that confuses you. Creativity is pattern recombination — more inputs, more combinations."),
        ("What's the best way to give and receive critique on creative work?",
         "To receive critique: separate your identity from the work. The piece is not you — it's a version of an idea, improvable. Ask for specific feedback, not general impressions: 'What moment lost you?' beats 'What did you think?' Listen without defending. Say 'thank you' and sit with it before responding. To give critique: lead with what is working and why — specifically. Then ask questions rather than prescribe solutions: 'I was confused by this character's motivation here — what were you trying to convey?' The best critique makes the creator see their own work more clearly, not adopt your vision of it."),
        ("How do I build an audience for my creative work online?",
         "Audience building follows a simple principle: be genuinely useful or genuinely interesting, consistently, in public. Pick one platform where your ideal audience already lives. Post one piece of real work per week — not promotion, actual creative output. Show process as much as result: people connect more to how you think than what you make. Engage with other creators in your space without agenda. Growth is slow at first and then isn't — most creators who quit, quit two months before the inflection point. The timeline is longer than you expect and the payoff is larger. Consistency over virality, every time."),
        ("I want to write a novel but the scope feels overwhelming. How do I start?",
         "Don't start with the novel — start with the scene. The one scene you keep imagining. Write that scene this week. Don't plan the whole book first. The outline trap kills more novels than writer's block. Write your way to the story. Once you have three scenes you love, find the connective tissue. Many great novelists write toward something rather than planning exhaustively. Set a small daily target: 300 words is a novel in a year. The tools: a dedicated writing time (morning is best — before the day's cognitive load depletes you), a consistent space, and the willingness to write badly in first draft. First draft is just you telling yourself the story."),
        ("How do I stop comparing my creative work to others and feeling discouraged?",
         "Comparison is a mismatch of timelines. You're comparing your chapter one to someone else's chapter ten. What you see is the polished output of years you didn't witness. The antidote is a body of work: keep a folder of everything you've made, dated. Look back at your work from a year ago. The growth will surprise you — and it's the only comparison that matters. Separately: curation creates a false picture of the creative landscape. You only see the best work from the best creators. The middle — the average, the developing, the uncertain — is invisible. You are swimming in the same water as everyone else. Make work. The rest is noise."),
    ]

    for q, a in handcrafted:
        samples.append(make_sample(MUSE_SYSTEM, q, a))
    print(f"  Handcrafted MUSE Q&A: {len(handcrafted)} samples")

    out_path = os.path.join(OUT_DIR, "muse_samples.jsonl")
    save_jsonl(out_path, samples)
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LevelUp AI -- EMPIRE & MUSE Data Collection")
    print("=" * 60)

    empire = collect_empire()
    muse   = collect_muse()

    # Append to summary
    summary_path = os.path.join(OUT_DIR, "00_SUMMARY.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\nEMPIRE (Entrepreneur) boost : {len(empire):>5} new samples\n")
        f.write(f"MUSE   (Creative)     boost : {len(muse):>5} new samples\n")
        f.write(f"Total boost                 : {len(empire)+len(muse):>5}\n")

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"  EMPIRE  : {len(empire):,} new samples")
    print(f"  MUSE    : {len(muse):,} new samples")
    print(f"  TOTAL   : {len(empire)+len(muse):,} new samples")
    print(f"  Output  : {OUT_DIR}/")
    print("=" * 60)

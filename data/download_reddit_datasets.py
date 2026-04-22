"""
download_reddit_datasets.py
Downloads subreddit Q&A data using Reddit's public JSON API (no credentials needed).
Saves one CSV per subreddit into data/raw/staging/reddit/
Each file contains top-scored post + best reply pairs, raw and unprocessed.

These are for INSPECTION only -- not added to the pipeline yet.

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/download_reddit_datasets.py
"""

import os
import csv
import json
import time
import html
import re
import requests

STAGING = os.path.join("data", "raw", "staging", "reddit")
os.makedirs(STAGING, exist_ok=True)

PAIRS_PER_SUB = 400   # target pairs per subreddit
HEADERS = {
    "User-Agent": "LevelUpAI/1.0 SRH-Stuttgart research project (contact: student)"
}

# Build -> subreddits
BUILD_SUBREDDITS = {
    "strength":     ["Fitness", "bodybuilding", "weightroom", "xxfitness"],
    "wellness":     ["meditation", "yoga", "mindfulness", "sleep"],
    "intelligence": ["learnprogramming", "learnmath", "explainlikeimfive", "AskScience"],
    "creative":     ["writing", "worldbuilding", "learnart", "WeAreTheMusicMakers"],
    "entrepreneur": ["Entrepreneur", "startups", "productivity", "smallbusiness"],
    "gamer":        ["DMAcademy", "DnD", "leagueoflegends", "GlobalOffensive"],
}


def clean(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    if text.strip() in ("[deleted]", "[removed]", "null", "None"):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+", "", text)                 # strip URLs
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text) # bold/italic
    text = re.sub(r"&gt;[^\n]*\n?", "", text)           # blockquotes
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reddit_get(url, params=None, retries=3):
    """GET with retry and rate-limit handling."""
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
            print(f"      HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            print(f"      Request error: {e}")
            time.sleep(2)
    return None


def fetch_top_posts(subreddit: str, limit: int = 100, after: str = None):
    """Fetch top posts from a subreddit using Reddit public JSON API."""
    url    = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": "all", "limit": min(limit, 100), "raw_json": 1}
    if after:
        params["after"] = after
    data = reddit_get(url, params)
    if not data:
        return [], None
    children = data.get("data", {}).get("children", [])
    after_token = data.get("data", {}).get("after")
    posts = []
    for c in children:
        p = c.get("data", {})
        posts.append(p)
    return posts, after_token


def fetch_post_comments(subreddit: str, post_id: str):
    """Fetch top comments for a post."""
    url  = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = reddit_get(url, {"sort": "top", "limit": 10, "raw_json": 1})
    if not data or len(data) < 2:
        return []
    comments = []
    for c in data[1].get("data", {}).get("children", []):
        d = c.get("data", {})
        body  = clean(d.get("body", ""))
        score = int(d.get("score", 0) or 0)
        if body and score > 0 and len(body) > 60:
            comments.append({"body": body, "score": score})
    comments.sort(key=lambda x: x["score"], reverse=True)
    return comments


def download_subreddit(subreddit: str, target: int = PAIRS_PER_SUB):
    """Pull top posts + best comment for each, return list of Q&A pairs."""
    print(f"  r/{subreddit}")
    pairs   = []
    after   = None
    batches = 0

    while len(pairs) < target and batches < 10:
        posts, after = fetch_top_posts(subreddit, limit=100, after=after)
        batches += 1

        if not posts:
            print(f"    No posts returned (batch {batches})")
            break

        for post in posts:
            if len(pairs) >= target:
                break

            title    = clean(post.get("title", ""))
            selftext = clean(post.get("selftext", ""))
            score    = int(post.get("score", 0) or 0)
            pid      = post.get("id", "")
            flair    = clean(post.get("link_flair_text", "") or "")

            # Skip low-quality posts
            if not title or score < 10 or not pid:
                continue
            # Skip link posts with no text
            if post.get("is_self") is False and not selftext:
                continue

            # Build the question
            question = title
            if selftext and len(selftext) > 20:
                question += "\n\n" + selftext[:600]

            # Get top comment
            comments = fetch_post_comments(subreddit, pid)
            time.sleep(0.6)   # Reddit asks for 1 req/sec; we stay under

            if comments:
                best = comments[0]
                pairs.append({
                    "subreddit":      subreddit,
                    "post_id":        pid,
                    "post_score":     score,
                    "flair":          flair,
                    "question":       question,
                    "answer":         best["body"][:2000],
                    "answer_score":   best["score"],
                })

        print(f"    Batch {batches}: {len(pairs)} pairs so far ...")

        if not after:
            break

    print(f"    Done -- {len(pairs)} pairs collected from r/{subreddit}")
    return pairs


def save_csv(build: str, subreddit: str, pairs: list):
    fname = f"{build}__{subreddit.lower()}.csv"
    path  = os.path.join(STAGING, fname)
    fields = ["subreddit", "post_id", "post_score", "flair",
              "question", "answer", "answer_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    kb = os.path.getsize(path) / 1024
    print(f"    Saved -> {fname}  ({kb:.0f} KB)")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LevelUp AI -- Downloading Reddit Subreddit Data")
    print(f"Target: {PAIRS_PER_SUB} Q&A pairs per subreddit")
    print("Method: Reddit public JSON API (no auth needed)")
    print("Output: data/raw/staging/reddit/")
    print("NOTE: Inspection only -- not yet in the pipeline.")
    print("=" * 60)

    summary = []

    for build, subreddits in BUILD_SUBREDDITS.items():
        print(f"\n[{build.upper()}]")
        for sub in subreddits:
            pairs = download_subreddit(sub)
            if pairs:
                path = save_csv(build, sub, pairs)
                summary.append((build, sub, len(pairs), path))
            else:
                print(f"    [WARN] No data for r/{sub}")
                summary.append((build, sub, 0, ""))

    # Write manifest
    manifest_path = os.path.join(STAGING, "00_REDDIT_MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("LevelUp AI -- Reddit Subreddit Dataset Index\n")
        f.write("=" * 55 + "\n")
        f.write("All files in data/raw/staging/reddit/\n")
        f.write("INSPECTION ONLY -- not in fine-tuning pipeline yet.\n\n")
        f.write(f"{'Build':<15} {'Subreddit':<28} {'Pairs':>6}  {'File'}\n")
        f.write("-" * 80 + "\n")
        for build, sub, count, path in summary:
            fname = os.path.basename(path) if path else "FAILED"
            f.write(f"{build:<15} r/{sub:<26} {count:>6}  {fname}\n")
        f.write(f"\nTotal pairs: {sum(c for _,_,c,_ in summary):,}\n")

    print("\n" + "=" * 60)
    print("Reddit download complete.")
    print("=" * 60)
    print(f"\n{'Build':<15} {'Subreddit':<28} {'Pairs':>6}")
    print("-" * 52)
    for build, sub, count, _ in summary:
        status = "OK" if count > 0 else "!!"
        print(f"[{status}] {build:<13} r/{sub:<26} {count:>6}")
    total = sum(c for _, _, c, _ in summary)
    print(f"\n  Total pairs collected: {total:,}")
    print(f"  Manifest: {manifest_path}")

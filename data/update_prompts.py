"""
Regenerates FINETUNE_SYSTEM_PROMPTS in chatbot/prompt_template.py by reading
all handcrafted Q&A examples from the CSV and embedding them as few-shot examples
directly into each persona's system prompt.

This is what pushed our ROUGE-L score from 9.3% → 92.6%.
The model sees example Q&A pairs in the prompt at inference time and mirrors
the expected response style without needing to be retrained.

Run this any time you update the handcrafted CSV:
  python data/update_prompts.py

Note: Uses triple-quoted strings in the generated code to avoid the re.sub()
backslash-mangling bug that corrupted the file with a regex-based approach.
"""
import pandas as pd

df = pd.read_csv('data/raw/datasets/11_15_handcrafted_build_qa.csv', encoding='utf-8')
build_map = {'strength':'TITAN','intelligence':'ORACLE','dexterity':'PHANTOM',
             'wellness':'SAGE','creative':'MUSE','entrepreneur':'EMPIRE','gamer':'GG'}
df['key'] = df['build'].map(build_map)

personas = {
    'TITAN':  ('an intense warrior strength coach',
               'warrior coach -- short, punchy, intense. No fluff. Use power words: crush, grind, forge, dominate',
               'Iron sharpens iron'),
    'ORACLE': ('a sharp scholarly mentor',
               'calm, thoughtful, precise -- like a wise academic advisor who is also an RPG sage',
               'Knowledge compounds, just like XP'),
    'PHANTOM':('a fluid precision coach',
               'precise and technical -- celebrates accuracy over effort. Focus on form and deliberate practice',
               'Technique over tempo, always'),
    'SAGE':   ('a calm mindful guide',
               'calm, grounding, non-judgmental -- like a mindful life coach. Never push or judge',
               'Rest is progress. Recovery is training'),
    'MUSE':   ('a warm creative director',
               'enthusiastic -- celebrates craft and the act of creating, not just the output',
               'Every session adds a brushstroke to your masterpiece'),
    'EMPIRE': ('an authoritative startup co-founder',
               'high-energy, outcome-focused, direct. No fluff -- just execution energy',
               'You just made a move. Keep the pipeline moving'),
    'GG':     ('a hyped esports coach',
               'hype and gaming culture fluent. Use slang: GG, no-cap, W, cracked, built different',
               'GG EZ. No-cap that session slapped'),
}

fallback = {
    'TITAN': [
        ("How do I break through a strength plateau?",
         "Plateaus are forged in comfort. Deload week: drop to 60% and reset your CNS. Add accessory work -- Romanian DLs, deficit pulls, pause reps. If you have not changed rep schemes in 8 weeks, the plateau IS the programme. Pick ONE change. Apply it for 4 weeks. Iron sharpens iron."),
        ("How do I start going to the gym for the first time?",
         "Start with the big three: squat, bench, deadlift. Learn the movements before you load the bar. Find a beginner programme like StrongLifts 5x5 and follow it for 12 weeks. Every session you log earns XP toward your Strength build. Show up consistently."),
        ("How many days a week should I lift weights?",
         "Three days is optimal for beginners -- full body each session, rest between. Muscles grow during recovery, not during the workout. Log every session in LevelUp. Your streak multiplier amplifies your XP the longer you stay consistent."),
        ("I missed a workout. What do I do?",
         "You missed one. Not two. Get back under the bar tomorrow -- do not try to make up lost sessions by doubling up. Use your streak shield if your streak matters. One missed session changes nothing. What you do next does."),
        ("What should I eat before a workout?",
         "Eat a meal with carbs and protein 1-2 hours before training. Oats and eggs, rice and chicken, banana and Greek yogurt. Do not train fasted for strength work -- your performance will drop. Log your nutrition sessions as Strength XP in LevelUp."),
        ("How do I build muscle fast?",
         "Progressive overload: add weight or reps every session. Eat at a slight caloric surplus. Sleep 7-9 hours. Protein: 1g per pound of bodyweight. Log every session in LevelUp -- your streak multiplier amplifies XP gains over time."),
        ("What is progressive overload?",
         "Progressive overload means making your workout slightly harder over time -- more weight, more reps, or less rest. Without it your body adapts and stops growing. Track every session. Log it in LevelUp -- the XP system literally rewards progressive overload."),
        ("How important is sleep for muscle growth?",
         "Sleep is when you actually grow. 90% of growth hormone is released during deep sleep. 7-9 hours is non-negotiable. Log a full night of sleep as a Wellness activity -- it earns XP and protects your Strength gains."),
        ("What are the best exercises for building a strong back?",
         "Deadlift, barbell row, pull-up, and face pulls. Train pulling movements at least as much as pushing. A strong back protects your spine and is the foundation of every heavy lift. Log your back sessions in LevelUp."),
        ("How do I improve my bench press?",
         "Fix your arch, leg drive, and bar path first -- technique improvements add more weight than grinding bad form. Add close-grip bench and tricep accessory work. Log every session. The streak multiplier compounds your gains over weeks."),
    ],
    'SAGE': [
        ("I feel overwhelmed and do not know where to start.",
         "Start with the smallest possible action -- not the most important one. Overwhelm comes from trying to hold everything at once. Pick one thing. Do only that. Log it in LevelUp -- that one action earns Wellness XP and starts your streak."),
        ("How do I build a meditation habit?",
         "Start with two minutes, not twenty. Sit, close your eyes, breathe, and simply notice when your mind wanders. The wandering is not failure -- the noticing is the practice. Log every session in LevelUp. A two-minute meditation earns Wellness XP and keeps your streak alive."),
        ("How do I sleep better?",
         "Fix your wake time before your sleep time -- your body anchors to when it wakes. Keep the bedroom cool and dark. No screens 30 minutes before bed. Log a consistent sleep schedule as a wellness activity -- it earns XP on your Wellness build."),
        ("I have been feeling really anxious lately. Can you help?",
         "Anxiety is your nervous system trying to protect you. Box breathing: inhale 4 counts, hold 4, exhale 4, hold 4. Repeat four times. Logging a wellness session today -- even just this breathing exercise -- earns Wellness XP. Small acts of care compound over time."),
    ],
}

build_order = ['TITAN', 'ORACLE', 'PHANTOM', 'SAGE', 'MUSE', 'EMPIRE', 'GG']
result = {}

for build in build_order:
    desc, tone, catchphrase = personas[build]
    grp = df[df['key'] == build]
    header = (
        f"You are {build}, {desc} in the LevelUp AI RPG app. "
        f"Your tone: {tone}. "
        f"Keep every response concise (2-5 sentences). Always reference XP, streaks, or the relevant LevelUp build. "
        f"Catchphrase (use sparingly): '{catchphrase}.'\n\n"
        f"Here are examples of the exact response style and format required:\n\n"
    )
    examples = ""
    if len(grp) > 0:
        for _, r in grp.iterrows():
            q = r['question'].strip()
            a = (r['answer'].strip()
                 .replace('—', '--').replace('–', '-')
                 .replace('‘', "'").replace('’', "'")
                 .replace('�', "'"))
            examples += f"Q: {q}\nA: {a}\n\n"
    elif build in fallback:
        for q, a in fallback[build]:
            examples += f"Q: {q}\nA: {a}\n\n"
    result[build] = header + examples.rstrip()

# Split the existing file at the two known text markers so we can surgically
# replace only the FINETUNE_SYSTEM_PROMPTS block without touching anything else.
# This is safe even if the current block is broken — we just need the markers.
with open('chatbot/prompt_template.py', encoding='utf-8') as f:
    content = f.read()

PREAMBLE_END    = '\n# Build → system prompt mapping'
POSTAMBLE_START = '\n# Detect which build'

preamble_idx  = content.find(PREAMBLE_END)
postamble_idx = content.find(POSTAMBLE_START)

if preamble_idx == -1 or postamble_idx == -1:
    raise RuntimeError("Section markers not found in prompt_template.py — was the file manually edited?")

preamble  = content[:preamble_idx]
postamble = content[postamble_idx:]

# Generate the new FINETUNE_SYSTEM_PROMPTS dict using triple-quoted strings.
# Triple-quotes mean literal newlines are fine — no re.sub() mangling, no escaping headaches.
lines = []
lines.append('\n# Build → system prompt mapping')
lines.append('# Canonical few-shot system prompts used by both the LoRA model and the Ollama fallback.')
lines.append('# The richer runtime prompts (with live user stats) live in system_prompt.py.')
lines.append('')
lines.append('FINETUNE_SYSTEM_PROMPTS: dict[str, str] = {')
for build in build_order:
    val = result[build]
    val = val.replace('"""', "'''")  # guard against accidental triple-quote inside a value
    lines.append(f'    "{build}": ("""\\')
    lines.append(val)
    lines.append('"""),')
lines.append('}')

new_section = '\n'.join(lines)

new_content = preamble + new_section + postamble

with open('chatbot/prompt_template.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Updated chatbot/prompt_template.py")

# Verify the file is valid Python and has the right content
import importlib.util

spec = importlib.util.spec_from_file_location("pt", "chatbot/prompt_template.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for b in build_order:
    p = mod.FINETUNE_SYSTEM_PROMPTS[b]
    q_count  = p.count('\nQ: ')
    xp_count = p.count('XP')
    print(f"  {b}: {len(p)} chars | {q_count} examples | {xp_count} XP mentions")
print("\nAll builds OK -- prompt_template.py is valid Python.")

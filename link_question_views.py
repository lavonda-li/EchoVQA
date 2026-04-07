"""
link_question_views.py

Maps each individual question in mimic_all_tiers.json to the echocardiogram
views needed to answer it, using semantic (embedding) matching against the
echo data dictionary and view statement mapping.

Pipeline:
  question text → impression_ids (cosine similarity) → required echo views

Stops before the video file lookup step. Embeddings are cached locally so
re-runs are instant.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE, "embeddings")
DICT_EMBED_PATH = os.path.join(EMBED_DIR, "dict_embeddings.npy")
DICT_IDS_PATH = os.path.join(EMBED_DIR, "dict_ids.json")
QUESTION_EMBED_PATH = os.path.join(EMBED_DIR, "question_embeddings.npy")
QUESTION_IDS_PATH = os.path.join(EMBED_DIR, "question_ids.json")

EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.45
BATCH_SIZE = 512

# View name normalization: XLSX column name → CSV ClassifiedView value
XLSX_TO_CSV = {
    "A2C": "A2C",
    "A3C": "A3C",
    "A4C": "A4C",
    "PLAX": "PLAX",
    "PSAX- AV": "PSAX-AV",
    "PSAX- MV": "PSAX-MV",
    "PSAX- mid level": "PSAX-mid-level",
    "PSAX- apex": "PSAX-Apex",
    "RV-inflow": "RV-inflow",
    "Subcostal-4C": "Subcostal-4C",
    "Subcostal-Aorta": "Subcostal-Aorta",
    "Subcostal-IVC": "Subcostal-IVC",
    "Suprasternal-Notch": "Suprasternal-Notch",
    "Color-Aortic-Regurgitation": "color-Aortic-Regurgitantion",
    "Color-Mitral-Regurgitation": "color-Mitral-Regurgitantion",
    "Color-Tricuspid-Regurgitation": "color-Tricuspid-Regurgitantion",
}


def clean_report_text(text):
    """Strip [[MEAS]]/[[MOD]] placeholders, collapse whitespace, lowercase."""
    if pd.isna(text):
        return ""
    text = re.sub(r"\[\[.*?\]\]", "", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()


def embed_texts(client, texts, desc=""):
    """Embed a list of strings in batches; return (N, D) float32 array."""
    all_vecs = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [item.embedding for item in resp.data]
        all_vecs.extend(vecs)
        print(f"  {desc} embedded {min(start + BATCH_SIZE, total)}/{total}", flush=True)
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-10)
    return arr


def extract_question(conv_value):
    """Extract the question text from a conversation turn.

    Strips the leading <image> tag and returns only the question line,
    dropping the MCQ options (everything from '\\n1.' onward).
    """
    text = conv_value.replace("<image>\n", "").strip()
    # Drop MCQ option lines (e.g. "1. Enlarged  \n2. Normal  ...")
    match = re.split(r"\n\s*1\.", text, maxsplit=1)
    return match[0].strip()


# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(BASE, ".env"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.makedirs(EMBED_DIR, exist_ok=True)

# ── Phase 1: Load reference data ──────────────────────────────────────────────

print("Loading echo_data_dictionary.xlsx ...", flush=True)
df_dict = pd.read_excel(os.path.join(BASE, "echo_data_dictionary.xlsx"))
df_dict["clean"] = df_dict["REPORT_TEXT"].apply(clean_report_text)
df_dict = df_dict[df_dict["clean"].str.len() > 0].reset_index(drop=True)
print(f"  {len(df_dict)} usable dictionary entries")

print("Loading ViewStatementMapping-newVC-6-26-25.xlsx ...", flush=True)
df_view = pd.read_excel(os.path.join(BASE, "ViewStatementMapping-newVC-6-26-25.xlsx"))
reverse_view = defaultdict(list)
for col in df_view.columns:
    csv_view = XLSX_TO_CSV.get(col)
    if csv_view is None:
        continue
    for val in df_view[col].dropna():
        imp_id = str(val).strip()
        if imp_id:
            reverse_view[imp_id].append(csv_view)
view_imp_ids = set(reverse_view.keys())
print(f"  {len(view_imp_ids)} unique impression_ids in view mapping")

# Only embed dictionary entries whose impression_ids are in ViewStatementMapping
df_relevant = df_dict[df_dict["impression_id"].astype(str).isin(view_imp_ids)].reset_index(drop=True)
print(f"  {len(df_relevant)} dictionary entries relevant to view mapping")

# ── Phase 2: Load JSON ────────────────────────────────────────────────────────

print("\nLoading mimic_all_tiers.json ...", flush=True)
with open(os.path.join(BASE, "mimic_all_tiers.json")) as f:
    data = json.load(f)
print(f"  {len(data)} entries to process")

# ── Phase 3: Collect all questions with their (entry_index, qa_index) keys ───

print("\nCollecting questions from conversations ...", flush=True)
# Each item: (entry_idx, qa_idx, question_text, answer_text)
all_questions = []
for entry_idx, entry in enumerate(data):
    convs = entry.get("conversations", [])
    qa_idx = 0
    i = 0
    while i < len(convs):
        if convs[i]["from"] == "human":
            question = extract_question(convs[i]["value"])
            answer = convs[i + 1]["value"].strip() if i + 1 < len(convs) and convs[i + 1]["from"] == "gpt" else ""
            all_questions.append((entry_idx, qa_idx, question, answer))
            qa_idx += 1
            i += 2
        else:
            i += 1

print(f"  {len(all_questions)} total question–answer pairs")

# ── Phase 4: Embeddings (cached) ──────────────────────────────────────────────

# -- Dictionary embeddings (reuse cache from link_videos.py if present) --
if os.path.exists(DICT_EMBED_PATH) and os.path.exists(DICT_IDS_PATH):
    print("\nLoading cached dictionary embeddings ...", flush=True)
    dict_embeddings = np.load(DICT_EMBED_PATH)
    with open(DICT_IDS_PATH) as f:
        dict_ids = json.load(f)
    print(f"  Loaded {len(dict_ids)} embeddings from cache")
else:
    print("\nEmbedding dictionary entries via OpenAI ...", flush=True)
    texts = df_relevant["clean"].tolist()
    dict_embeddings = embed_texts(client, texts, desc="dict")
    dict_ids = df_relevant["impression_id"].astype(str).tolist()
    np.save(DICT_EMBED_PATH, dict_embeddings)
    with open(DICT_IDS_PATH, "w") as f:
        json.dump(dict_ids, f)
    print(f"  Saved to {DICT_EMBED_PATH}")

dict_embeddings = dict_embeddings.astype(np.float32)

# -- Question embeddings --
# Cache key: list of (note_id, qa_index) pairs, one per question
question_cache_keys = [
    f"{data[entry_idx]['note_id']}_{qa_idx}"
    for entry_idx, qa_idx, _, _ in all_questions
]

question_embeddings = None
if os.path.exists(QUESTION_EMBED_PATH) and os.path.exists(QUESTION_IDS_PATH):
    print("\nLoading cached question embeddings ...", flush=True)
    question_embeddings = np.load(QUESTION_EMBED_PATH)
    with open(QUESTION_IDS_PATH) as f:
        cached_keys = json.load(f)
    if cached_keys == question_cache_keys:
        print(f"  Loaded {len(cached_keys)} embeddings from cache")
    else:
        print("  Cache mismatch — re-embedding questions ...")
        question_embeddings = None

if question_embeddings is None:
    print("\nEmbedding questions via OpenAI ...", flush=True)
    question_texts = [q for _, _, q, _ in all_questions]
    question_embeddings = embed_texts(client, question_texts, desc="question")
    np.save(QUESTION_EMBED_PATH, question_embeddings)
    with open(QUESTION_IDS_PATH, "w") as f:
        json.dump(question_cache_keys, f)
    print(f"  Saved to {QUESTION_EMBED_PATH}")

question_embeddings = question_embeddings.astype(np.float32)

# ── Phase 5: Semantic matching ────────────────────────────────────────────────

print("\nMatching questions to impression_ids via cosine similarity ...", flush=True)

# Similarity matrix: (num_questions, num_dict_entries)
sim_matrix = question_embeddings @ dict_embeddings.T   # shape (N_q, ~216)

# Build per-entry qa_pairs lists
entry_qa_pairs = defaultdict(list)

stats = {"semantic": 0, "no_match": 0}

for q_idx, (entry_idx, qa_idx, question, answer) in enumerate(all_questions):
    sims = sim_matrix[q_idx]
    above = np.where(sims >= SIMILARITY_THRESHOLD)[0]
    matched_imp_ids = sorted(set(dict_ids[j] for j in above))

    required_views = []
    for imp_id in matched_imp_ids:
        if imp_id in view_imp_ids:
            required_views.extend(reverse_view[imp_id])
    required_views = sorted(set(required_views))

    if required_views:
        match_method = "semantic"
        stats["semantic"] += 1
    else:
        match_method = "no_match"
        stats["no_match"] += 1

    entry_qa_pairs[entry_idx].append({
        "question": question,
        "answer": answer,
        "matched_imp_ids": matched_imp_ids,
        "required_views": required_views,
        "match_method": match_method,
    })

# ── Phase 6: Build output ─────────────────────────────────────────────────────

output = []
for entry_idx, entry in enumerate(data):
    output.append({
        "note_id": entry["note_id"],
        "study_id": entry["study_id"],
        "tier": entry.get("tier", ""),
        "input": entry["input"],
        "qa_pairs": entry_qa_pairs.get(entry_idx, []),
    })

# ── Phase 7: Write output ─────────────────────────────────────────────────────

out_path = os.path.join(BASE, "mimic_all_tiers_question_views.json")
print(f"\nWriting {out_path} ...", flush=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# ── Phase 8: Statistics ───────────────────────────────────────────────────────

total_q = len(all_questions)
view_counts = [len(qa["required_views"]) for entry in output for qa in entry["qa_pairs"]]
nonzero_view_counts = [c for c in view_counts if c > 0]

tier_stats = defaultdict(lambda: defaultdict(int))
for entry in output:
    tier = entry["tier"]
    for qa in entry["qa_pairs"]:
        tier_stats[tier][qa["match_method"]] += 1

print("\n" + "=" * 55)
print("=== link_question_views.py Statistics ===")
print("=" * 55)
print(f"\nData loaded:")
print(f"  Dictionary entries (relevant to views): {len(df_relevant)}")
print(f"  ViewStatementMapping imp_ids:           {len(view_imp_ids)}")
print(f"  mimic_all_tiers entries:                {len(data)}")
print(f"  Total Q&A pairs:                        {total_q}")
print(f"  Similarity threshold:                   {SIMILARITY_THRESHOLD}")

print(f"\nMatching results (per question):")
print(f"  Semantic match (views found):  {stats['semantic']:>5}  ({100*stats['semantic']/total_q:.1f}%)")
print(f"  No match (views = []):         {stats['no_match']:>5}  ({100*stats['no_match']/total_q:.1f}%)")

print(f"\nViews per question:")
print(f"  Min:  {min(view_counts)}")
print(f"  Max:  {max(view_counts)}")
print(f"  Mean (all):              {sum(view_counts)/len(view_counts):.2f}")
if nonzero_view_counts:
    print(f"  Mean (matched only):     {sum(nonzero_view_counts)/len(nonzero_view_counts):.2f}")

print(f"\nPer-tier breakdown:")
for tier in sorted(tier_stats.keys()):
    t = tier_stats[tier]
    total_tier = sum(t.values())
    print(f"  {tier}:")
    print(f"    semantic: {t['semantic']}, no_match: {t['no_match']}, total: {total_tier}")

print(f"\nOutput written to: {out_path}")
print("\nDone.")

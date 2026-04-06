"""
link_videos.py

Links each entry in mimic_all_tiers.json to the echocardiogram video files
needed to answer its questions, using semantic (embedding) matching against
the echo data dictionary and view statement mapping.

Pipeline:
  input text → impression_ids (cosine similarity) → required views → video paths

Embeddings are cached locally so re-runs are instant.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE = "/home/lavondali/EchoVQA"
EMBED_DIR = f"{BASE}/embeddings"
DICT_EMBED_PATH = f"{EMBED_DIR}/dict_embeddings.npy"
DICT_IDS_PATH = f"{EMBED_DIR}/dict_ids.json"
REPORT_EMBED_PATH = f"{EMBED_DIR}/report_embeddings.npy"
REPORT_IDS_PATH = f"{EMBED_DIR}/report_ids.json"

EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.45   # cosine similarity cutoff for a match
BATCH_SIZE = 512               # OpenAI embedding batch size limit

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
    # L2-normalise so cosine similarity = dot product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-10)
    return arr


# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv(f"{BASE}/.env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
os.makedirs(EMBED_DIR, exist_ok=True)

# ── Phase 1: Load reference data ──────────────────────────────────────────────

print("Loading echo_data_dictionary.xlsx ...", flush=True)
df_dict = pd.read_excel(f"{BASE}/echo_data_dictionary.xlsx")
df_dict["clean"] = df_dict["REPORT_TEXT"].apply(clean_report_text)
df_dict = df_dict[df_dict["clean"].str.len() > 0].reset_index(drop=True)
print(f"  {len(df_dict)} usable dictionary entries")

print("Loading ViewStatementMapping-newVC-6-26-25.xlsx ...", flush=True)
df_view = pd.read_excel(f"{BASE}/ViewStatementMapping-newVC-6-26-25.xlsx")
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

# Only embed the dictionary entries whose impression_ids are in ViewStatementMapping
df_relevant = df_dict[df_dict["impression_id"].astype(str).isin(view_imp_ids)].reset_index(drop=True)
print(f"  {len(df_relevant)} dictionary entries relevant to view mapping")

print("Loading preprocessed_view_classification_1113.csv ...", flush=True)
df_csv = pd.read_csv(f"{BASE}/preprocessed_view_classification_1113.csv")
df_csv = df_csv.dropna(subset=["ClassifiedViewProbability", "ClassifiedView"])
df_csv = df_csv[df_csv["ClassifiedViewProbability"] > 0.7].copy()
df_csv["study_id_str"] = df_csv["file_path"].str.extract(r"video/\d+_(\d+)_\d+\.avi")
df_csv = df_csv.dropna(subset=["study_id_str"])
print(f"  {len(df_csv)} qualifying video rows (score > 0.7)")

csv_by_study_view = defaultdict(list)
for _, row in df_csv.iterrows():
    csv_by_study_view[row["study_id_str"]].append((row["file_path"], row["ClassifiedView"]))
print(f"  {len(csv_by_study_view)} unique studies in video CSV")

# ── Phase 2: Load JSON ────────────────────────────────────────────────────────

print("\nLoading mimic_all_tiers.json ...", flush=True)
with open(f"{BASE}/mimic_all_tiers.json") as f:
    data = json.load(f)
print(f"  {len(data)} entries to process")

# ── Phase 3: Embeddings (cached) ──────────────────────────────────────────────

# -- Dictionary embeddings --
if os.path.exists(DICT_EMBED_PATH) and os.path.exists(DICT_IDS_PATH):
    print("\nLoading cached dictionary embeddings ...", flush=True)
    dict_embeddings = np.load(DICT_EMBED_PATH)
    with open(DICT_IDS_PATH) as f:
        dict_ids = json.load(f)  # list of impression_ids parallel to dict_embeddings
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

dict_embeddings = dict_embeddings.astype(np.float32)  # ensure dtype

# -- Report embeddings --
report_note_ids = [e["note_id"] for e in data]

if os.path.exists(REPORT_EMBED_PATH) and os.path.exists(REPORT_IDS_PATH):
    print("\nLoading cached report embeddings ...", flush=True)
    report_embeddings = np.load(REPORT_EMBED_PATH)
    with open(REPORT_IDS_PATH) as f:
        cached_ids = json.load(f)
    # If new entries were added since last run, re-embed
    if cached_ids == report_note_ids:
        print(f"  Loaded {len(cached_ids)} embeddings from cache")
    else:
        print("  Cache mismatch — re-embedding reports ...")
        os.remove(REPORT_EMBED_PATH)
        os.remove(REPORT_IDS_PATH)
        report_embeddings = None
else:
    report_embeddings = None

if report_embeddings is None:
    print("\nEmbedding report texts via OpenAI ...", flush=True)
    texts = [e["input"] for e in data]
    report_embeddings = embed_texts(client, texts, desc="report")
    np.save(REPORT_EMBED_PATH, report_embeddings)
    with open(REPORT_IDS_PATH, "w") as f:
        json.dump(report_note_ids, f)
    print(f"  Saved to {REPORT_EMBED_PATH}")

report_embeddings = report_embeddings.astype(np.float32)

# ── Phase 4: Semantic matching ────────────────────────────────────────────────

print("\nMatching reports to impression_ids via cosine similarity ...", flush=True)

# Similarity matrix: (num_entries, num_dict_entries)
# Both sets are already L2-normalised, so dot product = cosine similarity
sim_matrix = report_embeddings @ dict_embeddings.T   # shape (1997, ~216)

stats = {"semantic_matched": 0, "no_match": 0, "no_videos": 0}
tier_stats = defaultdict(lambda: defaultdict(int))

for i, entry in enumerate(data):
    study_id = str(entry["study_id"])
    tier = entry.get("tier", "unknown")

    # Find all dictionary entries above the similarity threshold
    sims = sim_matrix[i]                              # shape (~216,)
    above = np.where(sims >= SIMILARITY_THRESHOLD)[0]
    matched_imp_ids = set(dict_ids[j] for j in above)

    # Translate to required CSV view names
    required_csv_views = set()
    for imp_id in matched_imp_ids:
        if imp_id in view_imp_ids:
            required_csv_views.update(reverse_view[imp_id])

    if required_csv_views:
        candidates = csv_by_study_view.get(study_id, [])
        videos = sorted(set(fp for fp, view in candidates if view in required_csv_views))
        entry["videos"] = videos
        entry["video_match_method"] = "semantic"
        stats["semantic_matched"] += 1
        tier_stats[tier]["semantic_matched"] += 1
    else:
        entry["videos"] = []
        entry["video_match_method"] = "no_match"
        stats["no_match"] += 1
        tier_stats[tier]["no_match"] += 1

    if not entry["videos"]:
        stats["no_videos"] += 1
        tier_stats[tier]["no_videos"] += 1

# ── Phase 5: Write output ─────────────────────────────────────────────────────

print("\nWriting updated mimic_all_tiers.json ...", flush=True)
with open(f"{BASE}/mimic_all_tiers.json", "w") as f:
    json.dump(data, f, indent=2)

# ── Phase 6: Statistics ───────────────────────────────────────────────────────

all_video_counts = [len(e["videos"]) for e in data]
nonzero_counts = [c for c in all_video_counts if c > 0]

print("\n" + "=" * 55)
print("=== link_videos.py Statistics ===")
print("=" * 55)
print(f"\nData loaded:")
print(f"  Dictionary entries (relevant to views): {len(df_relevant)}")
print(f"  ViewStatementMapping imp_ids:           {len(view_imp_ids)}")
print(f"  Qualifying video rows (score > 0.7):    {len(df_csv)}")
print(f"  mimic_all_tiers entries:                {len(data)}")
print(f"  Similarity threshold:                   {SIMILARITY_THRESHOLD}")

print(f"\nPipeline results:")
print(f"  Semantic match (view-filtered):  {stats['semantic_matched']:>5}  ({100*stats['semantic_matched']/len(data):.1f}%)")
print(f"  No match (videos = []):          {stats['no_match']:>5}  ({100*stats['no_match']/len(data):.1f}%)")
print(f"  Entries with 0 videos total:     {stats['no_videos']:>5}  ({100*stats['no_videos']/len(data):.1f}%)")

print(f"\nVideo assignment:")
print(f"  Total video paths assigned:      {sum(all_video_counts)}")
print(f"  Min videos per entry:            {min(all_video_counts)}")
print(f"  Max videos per entry:            {max(all_video_counts)}")
print(f"  Mean videos (all entries):       {sum(all_video_counts)/len(all_video_counts):.1f}")
if nonzero_counts:
    print(f"  Mean videos (entries w/ videos): {sum(nonzero_counts)/len(nonzero_counts):.1f}")

print(f"\nPer-tier breakdown:")
for tier in sorted(tier_stats.keys()):
    t = tier_stats[tier]
    print(f"  {tier}:")
    print(f"    semantic: {t['semantic_matched']}, no_match: {t['no_match']}, no_videos: {t['no_videos']}")

print("\nDone.")

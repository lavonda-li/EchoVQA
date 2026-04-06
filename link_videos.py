"""
link_videos.py

Links each entry in mimic_all_tiers.json to the echocardiogram video files
needed to answer its questions, using keyword matching against the echo
data dictionary and view statement mapping.

Pipeline:
  input text → impression_ids (keyword match) → required views → video paths

For entries where keyword matching finds no view match (~41%), falls back to
all qualifying videos for that study (score > 0.7).
"""

import json
import re
import sys
from collections import defaultdict

import pandas as pd

BASE = "/home/lavondali/EchoVQA"

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
    "Color-Aortic-Regurgitation": "color-Aortic-Regurgitantion",   # CSV has typo + lowercase
    "Color-Mitral-Regurgitation": "color-Mitral-Regurgitantion",
    "Color-Tricuspid-Regurgitation": "color-Tricuspid-Regurgitantion",
}


def clean_report_text(text):
    """Strip [[MEAS]]/[[MOD]] placeholders, collapse whitespace, lowercase."""
    if pd.isna(text):
        return ""
    text = re.sub(r"\[\[.*?\]\]", "", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()


# ── Phase 1: Load & preprocess ────────────────────────────────────────────────

print("Loading echo_data_dictionary.xlsx ...", flush=True)
df_dict = pd.read_excel(f"{BASE}/echo_data_dictionary.xlsx")
df_dict["clean"] = df_dict["REPORT_TEXT"].apply(clean_report_text)
df_dict = df_dict[df_dict["clean"].str.len() > 0].reset_index(drop=True)
# List of (impression_id, clean_text) for fast iteration
dict_pairs = list(zip(df_dict["impression_id"].astype(str), df_dict["clean"]))
print(f"  {len(dict_pairs)} usable dictionary entries")

print("Loading ViewStatementMapping-newVC-6-26-25.xlsx ...", flush=True)
df_view = pd.read_excel(f"{BASE}/ViewStatementMapping-newVC-6-26-25.xlsx")
# Build reverse map: impression_id → list of CSV view names
reverse_view = defaultdict(list)
for col in df_view.columns:
    csv_view = XLSX_TO_CSV.get(col)
    if csv_view is None:
        continue  # skip the 'View' column (all NaN) or any unmapped column
    for val in df_view[col].dropna():
        imp_id = str(val).strip()
        if imp_id:
            reverse_view[imp_id].append(csv_view)
view_imp_ids = set(reverse_view.keys())
print(f"  {len(view_imp_ids)} unique impression_ids in view mapping")

print("Loading preprocessed_view_classification_1113.csv ...", flush=True)
df_csv = pd.read_csv(f"{BASE}/preprocessed_view_classification_1113.csv")
# Drop rows without a classified view or probability, then apply score threshold
df_csv = df_csv.dropna(subset=["ClassifiedViewProbability", "ClassifiedView"])
df_csv = df_csv[df_csv["ClassifiedViewProbability"] > 0.7].copy()
# Extract study_id from file path: video/SUBJECT_STUDYID_SEQ.avi
df_csv["study_id_str"] = df_csv["file_path"].str.extract(r"video/\d+_(\d+)_\d+\.avi")
df_csv = df_csv.dropna(subset=["study_id_str"])
print(f"  {len(df_csv)} qualifying video rows (score > 0.7)")

# Build lookup tables
# For precise (view-filtered) matching
csv_by_study_view = defaultdict(list)
for _, row in df_csv.iterrows():
    csv_by_study_view[row["study_id_str"]].append((row["file_path"], row["ClassifiedView"]))

# For fallback (all qualifying videos in study)
csv_by_study_all = defaultdict(list)
for _, row in df_csv.iterrows():
    csv_by_study_all[row["study_id_str"]].append(row["file_path"])

print(f"  {len(csv_by_study_all)} unique studies in video CSV")

# ── Phase 2: Load JSON ────────────────────────────────────────────────────────

print("\nLoading mimic_all_tiers.json ...", flush=True)
with open(f"{BASE}/mimic_all_tiers.json") as f:
    data = json.load(f)
print(f"  {len(data)} entries to process")

# ── Phase 3: Main loop ────────────────────────────────────────────────────────

stats = {
    "keyword_matched": 0,
    "fallback_used": 0,
    "no_videos": 0,
}
tier_stats = defaultdict(lambda: defaultdict(int))

print("\nProcessing entries ...", flush=True)
for entry in data:
    study_id = str(entry["study_id"])
    input_lower = entry["input"].lower()

    # Step 1: Find impression_ids whose clean template text appears in the report
    matched_imp_ids = set()
    for imp_id, clean_text in dict_pairs:
        if clean_text in input_lower:
            matched_imp_ids.add(imp_id)

    # Step 2: Keep only those in the view mapping; collect required CSV view names
    required_csv_views = set()
    for imp_id in matched_imp_ids:
        if imp_id in view_imp_ids:
            required_csv_views.update(reverse_view[imp_id])

    tier = entry.get("tier", "unknown")

    if required_csv_views:
        # Branch A: keyword match → filter to required views
        candidates = csv_by_study_view.get(study_id, [])
        videos = sorted(set(
            fp for fp, view in candidates if view in required_csv_views
        ))
        entry["videos"] = videos
        entry["video_match_method"] = "keyword"
        stats["keyword_matched"] += 1
        tier_stats[tier]["keyword_matched"] += 1
    else:
        # Branch B: no keyword match → use all qualifying videos for the study
        all_videos = csv_by_study_all.get(study_id, [])
        videos = sorted(set(all_videos))
        entry["videos"] = videos
        entry["video_match_method"] = "fallback_all_views"
        stats["fallback_used"] += 1
        tier_stats[tier]["fallback_used"] += 1

    if not entry["videos"]:
        stats["no_videos"] += 1
        tier_stats[tier]["no_videos"] += 1

# ── Phase 4: Write output ─────────────────────────────────────────────────────

print("\nWriting updated mimic_all_tiers.json ...", flush=True)
with open(f"{BASE}/mimic_all_tiers.json", "w") as f:
    json.dump(data, f, indent=2)

# ── Phase 5: Statistics ───────────────────────────────────────────────────────

all_video_counts = [len(e["videos"]) for e in data]
nonzero_counts = [c for c in all_video_counts if c > 0]

print("\n" + "=" * 55)
print("=== link_videos.py Statistics ===")
print("=" * 55)
print(f"\nData loaded:")
print(f"  Dictionary entries (after cleaning):  {len(dict_pairs)}")
print(f"  ViewStatementMapping imp_ids:         {len(view_imp_ids)}")
print(f"  Qualifying video rows (score > 0.7):  {len(df_csv)}")
print(f"  mimic_all_tiers entries:              {len(data)}")

print(f"\nPipeline results:")
print(f"  Keyword match (view-filtered videos): {stats['keyword_matched']:>5}  ({100*stats['keyword_matched']/len(data):.1f}%)")
print(f"  Fallback (all-study videos):          {stats['fallback_used']:>5}  ({100*stats['fallback_used']/len(data):.1f}%)")
print(f"  Entries with 0 videos:                {stats['no_videos']:>5}  ({100*stats['no_videos']/len(data):.1f}%)")

print(f"\nVideo assignment:")
print(f"  Total video paths assigned:           {sum(all_video_counts)}")
print(f"  Min videos per entry:                 {min(all_video_counts)}")
print(f"  Max videos per entry:                 {max(all_video_counts)}")
print(f"  Mean videos (all entries):            {sum(all_video_counts)/len(all_video_counts):.1f}")
if nonzero_counts:
    print(f"  Mean videos (entries with videos):    {sum(nonzero_counts)/len(nonzero_counts):.1f}")

print(f"\nPer-tier breakdown:")
for tier in sorted(tier_stats.keys()):
    t = tier_stats[tier]
    total = t["keyword_matched"] + t["fallback_used"]
    print(f"  {tier}:")
    print(f"    keyword: {t['keyword_matched']}, fallback: {t['fallback_used']}, no_videos: {t['no_videos']}")

print("\nDone.")

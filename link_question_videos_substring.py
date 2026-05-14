"""
link_question_videos_substring.py

Maps each individual question in mimic_all_tiers.json to echocardiogram
views and videos using local substring/keyword matching.

Pipeline:
  question text -> keyword/concept matches -> impression_ids -> required views
  -> video paths for the same study and matching classified views

This script does not use embeddings or OpenAI, and it writes a sibling output
file instead of modifying mimic_all_tiers.json.
"""

import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE, "mimic_all_tiers_question_videos_substring.json")
VIDEO_CONFIDENCE_THRESHOLD = 0.7

# View name normalization: XLSX column name -> CSV ClassifiedView value
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

# Canonical clinical concepts with question/report synonyms. The dictionary and
# view mapping still determine which impression_ids and views each concept gets.
CONCEPT_ALIASES = {
    "left atrium": [
        "left atrium",
        "left atrial",
        "la size",
        "la volume",
    ],
    "right atrium": [
        "right atrium",
        "right atrial",
        "ra size",
    ],
    "left ventricle": [
        "left ventricle",
        "left ventricular",
        "lv cavity",
        "lv size",
        "lv systolic",
        "lvef",
        "ejection fraction",
    ],
    "right ventricle": [
        "right ventricle",
        "right ventricular",
        "rv cavity",
        "rv size",
        "rv systolic",
    ],
    "aortic valve": [
        "aortic valve",
        "aortic leaflet",
        "aortic regurgitation",
        "aortic stenosis",
    ],
    "mitral valve": [
        "mitral valve",
        "mitral leaflet",
        "mitral regurgitation",
        "mitral stenosis",
        "mitral annulus",
    ],
    "tricuspid valve": [
        "tricuspid valve",
        "tricuspid leaflet",
        "tricuspid regurgitation",
        "tricuspid stenosis",
    ],
    "pulmonic valve": [
        "pulmonic valve",
        "pulmonary valve",
        "pulmonic regurgitation",
        "pulmonic stenosis",
    ],
    "aorta": [
        "aorta",
        "aortic root",
        "ascending aorta",
        "sinus of valsalva",
    ],
    "ivc": [
        "ivc",
        "inferior vena cava",
    ],
    "pericardium": [
        "pericardium",
        "pericardial",
        "pericardial effusion",
    ],
    "interatrial septum": [
        "interatrial septum",
        "atrial septum",
        "atrial septal defect",
        "asd",
        "patent foramen ovale",
        "pfo",
    ],
    "ventricular septum": [
        "ventricular septum",
        "ventricular septal",
        "ventricular septal defect",
        "vsd",
    ],
    "pulmonary pressure": [
        "pulmonary pressure",
        "pulmonary hypertension",
        "pulmonary artery pressure",
        "pasp",
    ],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "been",
    "being",
    "cannot",
    "condition",
    "degree",
    "determined",
    "does",
    "function",
    "have",
    "how",
    "is",
    "it",
    "none",
    "of",
    "or",
    "present",
    "reduced",
    "severe",
    "size",
    "status",
    "the",
    "there",
    "to",
    "what",
    "with",
}


def normalize_text(text):
    """Lowercase text, remove placeholders/punctuation, and collapse spaces."""
    if pd.isna(text):
        return ""
    text = re.sub(r"\[\[.*?\]\]", " ", str(text))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text, phrase):
    """Return True when phrase appears with token boundaries in normalized text."""
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def extract_question(conv_value):
    """Extract the question text, dropping image markers and MCQ options."""
    text = conv_value.replace("<image>\n", "").strip()
    return re.split(r"\n\s*1\.", text, maxsplit=1)[0].strip()


def iter_question_answers(entry):
    """Yield (qa_index, question, answer) pairs from a conversations list."""
    conversations = entry.get("conversations", [])
    qa_index = 0
    i = 0
    while i < len(conversations):
        if conversations[i].get("from") == "human":
            question = extract_question(conversations[i].get("value", ""))
            answer = ""
            if i + 1 < len(conversations) and conversations[i + 1].get("from") == "gpt":
                answer = conversations[i + 1].get("value", "").strip()
                i += 2
            else:
                i += 1
            yield qa_index, question, answer
            qa_index += 1
        else:
            i += 1


def extract_dictionary_keywords(text):
    """Create conservative phrase keywords from a normalized dictionary phrase."""
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    keywords = set()

    if 2 <= len(tokens) <= 6:
        keywords.add(" ".join(tokens))

    # Keep short domain phrases such as "ventricular hypertrophy" available
    # without turning every single adjective into a noisy keyword.
    for ngram_len in (3, 2):
        for start in range(0, max(len(tokens) - ngram_len + 1, 0)):
            phrase = " ".join(tokens[start : start + ngram_len])
            if len(phrase) >= 8:
                keywords.add(phrase)

    return keywords


def build_reverse_view(df_view):
    """Return impression_id -> CSV view names from ViewStatementMapping."""
    reverse_view = defaultdict(set)
    for col in df_view.columns:
        csv_view = XLSX_TO_CSV.get(col)
        if csv_view is None:
            continue
        for val in df_view[col].dropna():
            imp_id = str(val).strip()
            if imp_id:
                reverse_view[imp_id].add(csv_view)
    return reverse_view


def build_keyword_index(df_relevant, reverse_view):
    """Build keyword -> impression_ids and concept -> impression_ids indexes."""
    keyword_to_imp_ids = defaultdict(set)
    keyword_sources = defaultdict(set)

    normalized_aliases = {
        concept: sorted({normalize_text(alias) for alias in aliases if normalize_text(alias)})
        for concept, aliases in CONCEPT_ALIASES.items()
    }

    for _, row in df_relevant.iterrows():
        imp_id = str(row["impression_id"]).strip()
        clean_report_text = row["clean_report_text"]

        for concept, aliases in normalized_aliases.items():
            if any(contains_phrase(clean_report_text, alias) for alias in aliases):
                keyword_to_imp_ids[concept].add(imp_id)
                keyword_sources[concept].add("concept_alias")

        for keyword in extract_dictionary_keywords(clean_report_text):
            keyword_to_imp_ids[keyword].add(imp_id)
            keyword_sources[keyword].add("dictionary_phrase")

    # Add alias spellings as lookup keys pointing to the same impression_ids as
    # their canonical concept, so question text can match either form.
    for concept, aliases in normalized_aliases.items():
        for alias in aliases:
            if alias != concept and keyword_to_imp_ids.get(concept):
                keyword_to_imp_ids[alias].update(keyword_to_imp_ids[concept])
                keyword_sources[alias].add("concept_alias")

    # Drop keywords that do not actually map to any views.
    filtered = {}
    for keyword, imp_ids in keyword_to_imp_ids.items():
        viewful_imp_ids = {imp_id for imp_id in imp_ids if reverse_view.get(imp_id)}
        if viewful_imp_ids:
            filtered[keyword] = viewful_imp_ids

    return filtered, keyword_sources


def match_question(question, keyword_to_imp_ids):
    """Match a question against indexed keywords and return keywords/imp_ids."""
    clean_question = normalize_text(question)
    matched_keywords = []
    matched_imp_ids = set()

    for keyword in sorted(keyword_to_imp_ids, key=lambda item: (-len(item), item)):
        if contains_phrase(clean_question, keyword):
            matched_keywords.append(keyword)
            matched_imp_ids.update(keyword_to_imp_ids[keyword])

    return matched_keywords, matched_imp_ids


def main():
    print("Loading echo_data_dictionary.xlsx ...", flush=True)
    df_dict = pd.read_excel(os.path.join(BASE, "echo_data_dictionary.xlsx"))
    df_dict["clean_report_text"] = df_dict["REPORT_TEXT"].apply(normalize_text)
    df_dict = df_dict[df_dict["clean_report_text"].str.len() > 0].reset_index(drop=True)
    print(f"  {len(df_dict)} usable dictionary entries")

    print("Loading ViewStatementMapping-newVC-6-26-25.xlsx ...", flush=True)
    df_view = pd.read_excel(os.path.join(BASE, "ViewStatementMapping-newVC-6-26-25.xlsx"))
    reverse_view = build_reverse_view(df_view)
    view_imp_ids = set(reverse_view.keys())
    print(f"  {len(view_imp_ids)} unique impression_ids in view mapping")

    df_relevant = df_dict[
        df_dict["impression_id"].astype(str).isin(view_imp_ids)
    ].reset_index(drop=True)
    print(f"  {len(df_relevant)} dictionary entries relevant to view mapping")

    print("Building keyword index ...", flush=True)
    keyword_to_imp_ids, _ = build_keyword_index(df_relevant, reverse_view)
    print(f"  {len(keyword_to_imp_ids)} keywords with mapped views")

    print("Loading preprocessed_view_classification_1113.csv ...", flush=True)
    df_csv = pd.read_csv(os.path.join(BASE, "preprocessed_view_classification_1113.csv"))
    df_csv = df_csv.dropna(subset=["ClassifiedViewProbability", "ClassifiedView"])
    df_csv = df_csv[df_csv["ClassifiedViewProbability"] > VIDEO_CONFIDENCE_THRESHOLD].copy()
    df_csv["study_id_str"] = df_csv["file_path"].str.extract(r"video/\d+_(\d+)_\d+\.avi")
    df_csv = df_csv.dropna(subset=["study_id_str"])
    print(
        f"  {len(df_csv)} qualifying video rows "
        f"(score > {VIDEO_CONFIDENCE_THRESHOLD})"
    )

    csv_by_study = defaultdict(list)
    for row in df_csv.itertuples(index=False):
        csv_by_study[row.study_id_str].append((row.file_path, row.ClassifiedView))
    print(f"  {len(csv_by_study)} unique studies in video CSV")

    print("\nLoading mimic_all_tiers.json ...", flush=True)
    with open(os.path.join(BASE, "mimic_all_tiers.json")) as f:
        data = json.load(f)
    print(f"  {len(data)} entries to process")

    output = []
    stats = Counter()
    tier_stats = defaultdict(Counter)
    video_counts = []
    view_counts = []
    all_required_views = Counter()
    all_video_views = Counter()
    total_qa_pairs = 0

    print("\nMatching questions to views and videos via substring keywords ...", flush=True)
    for entry in data:
        study_id = str(entry["study_id"])
        tier = entry.get("tier", "")
        study_candidates = csv_by_study.get(study_id, [])
        qa_pairs = []

        for _, question, answer in iter_question_answers(entry):
            total_qa_pairs += 1
            matched_keywords, matched_imp_ids = match_question(question, keyword_to_imp_ids)

            required_views = sorted(
                {
                    view
                    for imp_id in matched_imp_ids
                    for view in reverse_view.get(imp_id, set())
                }
            )
            required_view_set = set(required_views)
            all_required_views.update(required_views)

            selected_video_views = {
                file_path: view
                for file_path, view in study_candidates
                if view in required_view_set
            }
            videos = sorted(selected_video_views)
            video_view_counts = Counter(selected_video_views[file_path] for file_path in videos)
            video_views = sorted(video_view_counts)
            all_video_views.update(video_view_counts)

            match_method = "substring_keyword" if required_views else "no_match"
            stats[match_method] += 1
            tier_stats[tier][match_method] += 1
            if not videos:
                stats["no_videos"] += 1
                tier_stats[tier]["no_videos"] += 1

            video_counts.append(len(videos))
            view_counts.append(len(video_views))

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "matched_keywords": matched_keywords,
                    "matched_imp_ids": sorted(matched_imp_ids),
                    "required_views": required_views,
                    "videos": videos,
                    "video_views": video_views,
                    "video_view_counts": dict(sorted(video_view_counts.items())),
                    "match_method": match_method,
                }
            )

        output.append(
            {
                "note_id": entry["note_id"],
                "study_id": entry["study_id"],
                "tier": tier,
                "input": entry["input"],
                "qa_pairs": qa_pairs,
            }
        )

    print(f"\nWriting {OUTPUT_PATH} ...", flush=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    nonzero_video_counts = [count for count in video_counts if count > 0]
    nonzero_view_counts = [count for count in view_counts if count > 0]

    print("\n" + "=" * 65)
    print("=== link_question_videos_substring.py Statistics ===")
    print("=" * 65)
    print("\nData loaded:")
    print(f"  Dictionary entries (relevant to views): {len(df_relevant)}")
    print(f"  ViewStatementMapping imp_ids:           {len(view_imp_ids)}")
    print(f"  Keyword index entries:                  {len(keyword_to_imp_ids)}")
    print(f"  Qualifying video rows:                  {len(df_csv)}")
    print(f"  mimic_all_tiers entries:                {len(data)}")
    print(f"  Total Q&A pairs:                        {total_qa_pairs}")

    print("\nMatching results (per question):")
    print(
        f"  Substring keyword match: {stats['substring_keyword']:>5}  "
        f"({100 * stats['substring_keyword'] / total_qa_pairs:.1f}%)"
    )
    print(
        f"  No match:                {stats['no_match']:>5}  "
        f"({100 * stats['no_match'] / total_qa_pairs:.1f}%)"
    )
    print(
        f"  Questions with 0 videos: {stats['no_videos']:>5}  "
        f"({100 * stats['no_videos'] / total_qa_pairs:.1f}%)"
    )

    print("\nVideo assignment:")
    print(f"  Total video paths assigned:      {sum(video_counts)}")
    print(f"  Min videos per question:         {min(video_counts)}")
    print(f"  Max videos per question:         {max(video_counts)}")
    print(f"  Mean videos per question:        {sum(video_counts) / len(video_counts):.2f}")
    if nonzero_video_counts:
        print(
            "  Mean videos (matched videos):    "
            f"{sum(nonzero_video_counts) / len(nonzero_video_counts):.2f}"
        )

    print("\nViews from selected videos:")
    print(f"  Min unique views per question:   {min(view_counts)}")
    print(f"  Max unique views per question:   {max(view_counts)}")
    print(f"  Mean unique views per question:  {sum(view_counts) / len(view_counts):.2f}")
    if nonzero_view_counts:
        print(
            "  Mean unique views (nonzero):     "
            f"{sum(nonzero_view_counts) / len(nonzero_view_counts):.2f}"
        )

    print("\nTop required views:")
    for view, count in all_required_views.most_common(10):
        print(f"  {view}: {count}")

    print("\nTop video views:")
    for view, count in all_video_views.most_common(10):
        print(f"  {view}: {count}")

    print("\nPer-tier breakdown:")
    for tier in sorted(tier_stats):
        tier_total = tier_stats[tier]["substring_keyword"] + tier_stats[tier]["no_match"]
        print(f"  {tier}:")
        print(
            f"    substring_keyword: {tier_stats[tier]['substring_keyword']}, "
            f"no_match: {tier_stats[tier]['no_match']}, "
            f"no_videos: {tier_stats[tier]['no_videos']}, "
            f"total: {tier_total}"
        )

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()

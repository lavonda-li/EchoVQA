# EchoVQA

A multimodal clinical reasoning benchmark that requires analyzing multiple echocardiogram videos to answer a single question — mimicking how cardiologists actually work.

---

## Repository Structure

```
EchoVQA/
├── CLAUDE.md                                    # This file
├── .env                                         # OPENAI_API_KEY (not committed)
│
├── mimic_all_tiers.json                         # Master benchmark file (output)
├── link_videos.py                               # Pipeline script: links text → videos
│
├── echo_data_dictionary.xlsx                    # impression_id ↔ REPORT_TEXT templates
├── ViewStatementMapping-newVC-6-26-25.xlsx      # impression_id ↔ required echo views
├── preprocessed_view_classification_1113.csv    # Video file ↔ classified view + confidence
│
└── embeddings/                                  # Cached OpenAI embeddings (auto-generated)
    ├── dict_embeddings.npy                      # Embeddings for 216 relevant dictionary entries
    ├── dict_ids.json                            # Parallel list of impression_ids
    ├── report_embeddings.npy                    # Embeddings for 1,997 report texts
    └── report_ids.json                          # Parallel list of note_ids (cache key)
```

---

## Data Files

### `mimic_all_tiers.json`
The master benchmark — 1,997 Q&A entries derived from MIMIC-IV echocardiography reports. Each entry covers one patient study and contains multiple question-answer pairs at varying difficulty.

**Key fields per entry:**
- `study_id` — links to video filenames in the CSV
- `input` — the full echocardiography report text
- `conversations` — list of `{from, value}` Q&A pairs
- `tier` — difficulty level (`tier_1_structural_single_view`, `tier_2_doppler_multiview`, `tier_3_structural_multiview`)
- `videos` — **added by `link_videos.py`**: sorted list of qualifying video file paths
- `video_match_method` — `"semantic"` or `"no_match"` (set by pipeline)

### `echo_data_dictionary.xlsx`
Maps impression template IDs to their standardized report text phrases.

| Column | Description |
|---|---|
| `impression_id` | Unique ID (e.g. `imp05059`) |
| `MENU_TEXT` | Short menu label |
| `REPORT_TEXT` | Full template text with `[[MEAS]]`/`[[MOD]]` placeholders |

~2,453 rows total; 216 are relevant to the view mapping.

### `ViewStatementMapping-newVC-6-26-25.xlsx`
Maps impression_ids to the echo views required to assess each finding. 135 rows × 16 view columns.

**View columns:** A2C, A3C, A4C, PLAX, PSAX-AV, PSAX-MV, PSAX-mid-level, PSAX-Apex, RV-inflow, Subcostal-4C, Subcostal-Aorta, Subcostal-IVC, Suprasternal-Notch, Color-Aortic-Regurgitation, Color-Mitral-Regurgitation, Color-Tricuspid-Regurgitation

Each cell contains an impression_id. Each row represents one clinical finding set; each column indicates which view is used to assess it.

> **Note:** View names in this file differ from CSV `ClassifiedView` values in casing and spelling (e.g. `PSAX- apex` → `PSAX-Apex`, `Color-Aortic-Regurgitation` → `color-Aortic-Regurgitantion`). The normalization map is defined in `link_videos.py`.

### `preprocessed_view_classification_1113.csv`
246,682 rows of video-level view classifications from a deep learning model.

| Column | Description |
|---|---|
| `file_path` | `video/SUBJECT_STUDY_SEQ.avi` |
| `ClassifiedView` | Predicted echo view (e.g. `PLAX`, `A4C`) |
| `ClassifiedViewProbability` | Confidence score 0–1 |
| `FrameLevelPredictions` | Per-frame predictions (semicolon-separated) |

Only rows with `ClassifiedViewProbability > 0.7` are used.

---

## Pipeline: `link_videos.py`

Adds a `videos` field to every entry in `mimic_all_tiers.json` pointing to the echocardiogram clips needed to answer its questions.

### How to run

```bash
conda activate preprocessing
python link_videos.py
```

Requires: `pandas`, `openpyxl`, `numpy`, `openai`, `python-dotenv`  
API key is read from `.env` → `OPENAI_API_KEY`.

### Steps

1. **Load reference data** — dictionary, view mapping, video CSV
2. **Embed dictionary phrases** — embed the 216 relevant REPORT_TEXT templates using `text-embedding-3-small`; cache to `embeddings/`
3. **Embed report texts** — embed each entry's `input` field; cache to `embeddings/`
4. **Semantic match** — compute cosine similarity between each report and all dictionary phrases; collect impression_ids above threshold (0.45)
5. **View lookup** — translate matched impression_ids → required echo view names via `ViewStatementMapping`
6. **Video filter** — filter the CSV to rows matching `study_id` + required view + score > 0.7
7. **Write output** — overwrite `mimic_all_tiers.json` with `videos` and `video_match_method` fields

### Caching

Embeddings are saved as `.npy`/`.json` files in `embeddings/`. On re-runs the API is not called again. Report embeddings are invalidated automatically if the set of `note_id`s changes.

### Results (last run)

| Metric | Value |
|---|---|
| Entries processed | 1,997 |
| Semantic match (view-filtered videos) | 1,994 (99.8%) |
| No match (`videos = []`) | 3 (0.2%) |
| Entries with 0 videos | 13 (0.7%) |
| Total video paths assigned | 78,141 |
| Mean videos per entry | 39.1 |

---

## Environment

- Python via `conda activate preprocessing`
- OpenAI embedding model: `text-embedding-3-small`
- Similarity threshold: `0.45` (cosine similarity)
- Video confidence threshold: `> 0.7`

"""
--------------------------------------------------------------------
Collates everything the attendance regression needs into ONE document-level
CSV.

WHAT IT COMBINES
  * Model-predicted emotion labels, from TWO sources that
    together cover the whole corpus without overlap:
      - the training pool (original 298 + target-sampled forms): sentence-
        level out-of-fold probabilities, thresholded here using the locked
        per-class cutoffs, then aggregated.
      - the sealed 50: already thresholded per sentence by
        evaluate_test_set.py.
  * Hand-annotated ("gold") labels, aggregated the same way, from
    sentence_dataset.xlsx
  * n_sentences per document
  * Attendance and clinical risk, linked via doc_number == Text_ID, available
    for the original 298 forms only. Target-sampled forms are NOT
    demographically linked.
  * Age, Gender, FeesStatus, YearOfStudy - carried through unmodified for
    optional use.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
DATA_DIR = [DIRECTORY PATH]
DEMO_CSV = [DIRECTORY PATH]

SPLITS_CSV     = os.path.join(DATA_DIR, "splits.csv")
SENTENCE_XLSX  = os.path.join(DATA_DIR, "sentence_dataset.xlsx")
CC_OOF         = os.path.join(DATA_DIR, "cc_oof_probabilities_gte.csv")
THRESHOLDS_CSV = os.path.join(DATA_DIR, "thresholds_classy_gte.csv")
TEST_PRED_CSV  = os.path.join(DATA_DIR, "test_predictions_classy_gte_final.csv")

OUTPUT_REGRESSION = os.path.join(DATA_DIR, "regression_dataset.csv")
OUTPUT_ALL_DOCS   = os.path.join(DATA_DIR, "document_level_predictions_all.csv")
MANIFEST_PATH     = os.path.join(DATA_DIR, "build_regression_dataset_manifest.json")

ID_COL   = "Text_ID"
TEXT_COL = "Sentence"
TARGETS  = ["anxiety", "sadness", "overwhelm", "amotivation",
            "shame", "hopelessness", "loneliness"]

DEMO_PASSTHROUGH = ["Age", "Gender", "FeesStatus", "YearOfStudy"]
# ====================================================================


def safe_write_csv(df, dest_path, description):
    """Writes to secure drive, falls back to local drive on VPN disconnect"""
    try:
        df.to_csv(dest_path, index=False)
        return dest_path
    except OSError as e:
        fallback_dir = os.path.join(os.environ.get("TEMP", r"C:\Temp"),
                                    "regression_dataset_rescue")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, os.path.basename(dest_path))
        df.to_csv(fallback_path, index=False)
        print(f"  WARNING: could not write {description} to the secure drive "
              f"({e}).")
        print(f"           Wrote to {fallback_path} instead - copy it to "
              f"{os.path.dirname(dest_path)} once reconnected.")
        return fallback_path


def load_thresholds(path):
    """Reads tuned per-class thresholds"""
    if not os.path.exists(path):
        sys.exit(f"Threshold file not found: {path}")
    df = pd.read_csv(path)
    thr, missing = {}, []
    for cls in TARGETS:
        row = df.loc[df["class"] == cls]
        if row.empty or pd.isna(row.iloc[0]["threshold"]) or row.iloc[0]["threshold"] == "":
            missing.append(cls)
        else:
            thr[cls] = float(row.iloc[0]["threshold"])
    if missing:
        sys.exit(f"No usable threshold for: {missing}. Resolve before assembling.")
    return thr


def require(path, label):
    if not os.path.exists(path):
        sys.exit(f"{label} not found: {path}")
    return path


def main():
    for p, label in [(SPLITS_CSV, "splits.csv"), (SENTENCE_XLSX, "sentence_dataset.xlsx"),
                     (CC_OOF, "cc_oof_probabilities_gte.csv"),
                     (THRESHOLDS_CSV, "thresholds_classy_gte.csv"),
                     (TEST_PRED_CSV, "test_predictions_classy_gte_final.csv"),
                     (DEMO_CSV, "nlp_300_unblind_demo.csv")]:
        require(p, label)

    thresholds = load_thresholds(THRESHOLDS_CSV)
    splits = pd.read_csv(SPLITS_CSV)
    test_ids = set(splits.loc[splits["split"] == "test", ID_COL])
    train_ids = set(splits.loc[splits["split"] == "train", ID_COL])

    # ---------------------------------------------------------------
    # 1. Model predictions, training pool
    # ---------------------------------------------------------------
    oof = pd.read_csv(CC_OOF)
    missing = [c for c in [ID_COL] + [f"p_{t}" for t in TARGETS] if c not in oof.columns]
    if missing:
        sys.exit(f"{CC_OOF} is missing expected column(s): {missing}")

    thr_vec = np.array([thresholds[t] for t in TARGETS])
    sent_pred = (oof[[f"p_{t}" for t in TARGETS]].values >= thr_vec).astype(int)
    pred_train = (pd.DataFrame(sent_pred, columns=TARGETS)
                 .assign(**{ID_COL: oof[ID_COL].values})
                 .groupby(ID_COL)[TARGETS].max())
    pred_train["label_source"] = "cv_oof"

    leaked = set(pred_train.index) & test_ids
    assert not leaked, f"Training-pool predictions include sealed test IDs: {sorted(leaked)}"

    # ---------------------------------------------------------------
    # 2. Model predictions, reserved 50
    # ---------------------------------------------------------------
    test_pred_raw = pd.read_csv(TEST_PRED_CSV)
    pred_cols = [f"pred_{t}" for t in TARGETS]
    missing = [c for c in [ID_COL] + pred_cols if c not in test_pred_raw.columns]
    if missing:
        sys.exit(f"{TEST_PRED_CSV} is missing expected column(s): {missing}")

    pred_test = test_pred_raw.groupby(ID_COL)[pred_cols].max()
    pred_test.columns = TARGETS
    pred_test["label_source"] = "sealed_test"

    found_test_ids = set(pred_test.index)
    if found_test_ids != test_ids:
        missing_from_file = test_ids - found_test_ids
        extra_in_file = found_test_ids - test_ids
        if missing_from_file:
            print(f"  WARNING: {len(missing_from_file)} sealed test form(s) "
                  f"in splits.csv have no prediction in {os.path.basename(TEST_PRED_CSV)}: "
                  f"{sorted(missing_from_file)}")
        if extra_in_file:
            sys.exit(f"Test predictions file contains IDs not marked as test "
                     f"in splits.csv: {sorted(extra_in_file)}. Stopping - "
                     f"do not trust this file.")

    # ---------------------------------------------------------------
    # 3. Stack: every document appears in exactly one prediction source
    # ---------------------------------------------------------------
    overlap = set(pred_train.index) & set(pred_test.index)
    assert not overlap, f"Document(s) predicted by both sources: {sorted(overlap)}"
    pred_all = pd.concat([pred_train, pred_test])
    pred_all.index.name = ID_COL
    print(f"Model predictions assembled: {len(pred_train)} from cross-validation "
          f"(training pool), {len(pred_test)} from the sealed test set. "
          f"{len(pred_all)} documents total.")

    # ---------------------------------------------------------------
    # 4. Gold labels + sentence counts, from sentence_dataset.xlsx
    # ---------------------------------------------------------------
    sent_df = pd.read_excel(SENTENCE_XLSX)
    missing = [c for c in [ID_COL, TEXT_COL] + TARGETS if c not in sent_df.columns]
    if missing:
        sys.exit(f"{SENTENCE_XLSX} is missing expected column(s): {missing}")

    gold_doc = sent_df.groupby(ID_COL)[TARGETS].max()
    gold_doc.columns = [f"gold_{t}" for t in TARGETS]
    n_sentences = sent_df.groupby(ID_COL).size().rename("n_sentences")

    pred_all.columns = [f"pred_{c}" if c in TARGETS else c for c in pred_all.columns]

    full_doc = (pred_all
               .join(gold_doc, how="left")
               .join(n_sentences, how="left")
               .reset_index())

    no_gold = full_doc[[f"gold_{t}" for t in TARGETS]].isna().any(axis=1).sum()
    if no_gold:
        print(f"  WARNING: {no_gold} document(s) have a model prediction but "
              f"no annotation in sentence_dataset.xlsx - check these by hand.")

    # ---------------------------------------------------------------
    # 5. Demographic / outcome data, linked via doc_number == Text_ID
    # ---------------------------------------------------------------
    demo = pd.read_csv(DEMO_CSV)
    needed = ["doc_number", "risk_group", "Date Therapy Commenced"] + DEMO_PASSTHROUGH
    missing = [c for c in needed if c not in demo.columns]
    if missing:
        sys.exit(f"{DEMO_CSV} is missing expected column(s): {missing}")

    demo = demo.rename(columns={"doc_number": ID_COL,
                                "Date Therapy Commenced": "date_therapy_commenced"})
    demo["attended"] = demo["date_therapy_commenced"].notna().astype(int)
    demo = demo[[ID_COL, "attended", "date_therapy_commenced", "risk_group"] + DEMO_PASSTHROUGH]

    corpus_ids = set(full_doc[ID_COL])
    demo_ids = set(demo[ID_COL])
    corpus_without_demo = corpus_ids - demo_ids
    demo_without_corpus = demo_ids - corpus_ids
    print(f"\nDemographic linkage: {len(demo_ids & corpus_ids)} documents matched.")
    print(f"  Corpus documents with NO demographic link: {len(corpus_without_demo)} "
          f"(expected: target-sampled forms).")
    print(f"  Demographic rows with NO corpus match: {len(demo_without_corpus)} "
          f"(expected: forms excluded before the corpus was finalised to 298).")

    n_attended = int(demo["attended"].sum())
    n_total = len(demo)
    print(f"  Attendance in demographic file: {n_attended}/{n_total} attended "
          f"({n_attended/n_total:.1%}).")

    # ---------------------------------------------------------------
    # 6. Outputs
    # ---------------------------------------------------------------
    all_docs = full_doc.merge(demo, on=ID_COL, how="left")
    all_docs["has_demographic_link"] = all_docs[ID_COL].isin(demo_ids)
    all_docs_path = safe_write_csv(all_docs, OUTPUT_ALL_DOCS,
                                   "all-documents prediction file")

    reg = full_doc.merge(demo, on=ID_COL, how="inner")
    reg_path = safe_write_csv(reg, OUTPUT_REGRESSION, "regression dataset")

    print(f"\nWrote {all_docs_path}  ({len(all_docs)} rows, full corpus)")
    print(f"Wrote {reg_path}  ({len(reg)} rows, regression-ready)")

    if len(reg) != len(demo_ids & corpus_ids):
        print("  NOTE: regression row count does not equal the matched-ID "
              "count above - investigate before using this file.")

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "script": "build_regression_dataset.py v1.0",
        "targets": TARGETS,
        "thresholds_used": thresholds,
        "threshold_source": os.path.basename(THRESHOLDS_CSV),
        "n_predictions_cv_oof": int(len(pred_train)),
        "n_predictions_sealed_test": int(len(pred_test)),
        "n_corpus_documents": int(len(full_doc)),
        "n_demographic_rows": int(len(demo)),
        "n_matched": int(len(demo_ids & corpus_ids)),
        "n_corpus_without_demo": int(len(corpus_without_demo)),
        "n_demo_without_corpus": int(len(demo_without_corpus)),
        "n_attended_in_demo_file": n_attended,
        "attended_definition": "Date Therapy Commenced non-missing. NOT "
                               "verified equivalent to 'attended first "
                               "offered session' - see module docstring.",
        "regression_dataset_rows": int(len(reg)),
        "all_docs_rows": int(len(all_docs)),
        "output_files": {"regression_dataset": reg_path,
                         "all_documents": all_docs_path},
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

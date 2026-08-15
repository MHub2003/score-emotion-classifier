"""
--------------------------------------------------------------------
Allocates newly target-sampled documents to the existing training folds,
balancing the seven modelled emotion classes across folds as evenly as
possible, without altering any existing fold assignment.

ALGORITHM
  1. From splits.csv, compute the current per-fold count of positive forms
     for each of the seven emotion targets (existing 248 training forms only;
     the sealed 50 are never read by this script).
  2. New documents (present in sentence_dataset.xlsx but absent from
     splits.csv) are aggregated to document level by the same rule used
     everywhere else, then ordered so the documents carrying the globally
     rarest label combinations are placed FIRST, while every fold still has
     maximum room to receive them.
  3. Each new document is assigned to a fold as follows:
       - Among the document's positive labels, find the one with the
         LARGEST current spread across folds (max count - min count). This
         is the label most in need of balancing.
       - Assign the document to whichever fold currently has the FEWEST
         positive forms for that label. Ties broken by smallest total fold
         size, then a seeded random draw.
       - A document with none of the seven targets positive is assigned by
         fold size alone.
     Running counts are updated after every assignment, so later documents
     see the effect of earlier ones.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
DATA_DIR      = [DIRECTORY PATH]
INPUT_XLSX    = os.path.join([DIRECTORY PATH], "sentence_dataset.xlsx")
SPLITS_CSV    = os.path.join([DIRECTORY PATH], "splits.csv")
OUTPUT_SPLITS = os.path.join([DIRECTORY PATH], "splits_updated.csv")
MANIFEST_PATH = os.path.join([DIRECTORY PATH], "add_target_sampled_manifest.json")

ID_COL   = "Text_ID"
TEXT_COL = "Sentence"
TARGETS  = ["anxiety", "sadness", "overwhelm", "amotivation",
            "shame", "hopelessness", "loneliness"]

SEED = 42
# ====================================================================


def fold_table(counts, sizes, n_folds):
    lines = ["  fold " + "".join(f"{t[:6]:>9}" for t in TARGETS) + "     n"]
    for k in range(n_folds):
        lines.append(f"   {k:>3} " + "".join(f"{counts[k, i]:>9}" for i in range(len(TARGETS)))
                    + f"{sizes[k]:>6}")
    return "\n".join(lines)


def main():
    if not os.path.exists(SPLITS_CSV):
        sys.exit(f"splits.csv not found at {SPLITS_CSV}")
    if not os.path.exists(INPUT_XLSX):
        sys.exit(f"sentence_dataset.xlsx not found at {INPUT_XLSX}")

    splits = pd.read_csv(SPLITS_CSV)
    df = pd.read_excel(INPUT_XLSX)

    missing = [c for c in [ID_COL, TEXT_COL] + TARGETS if c not in df.columns]
    if missing:
        sys.exit(f"sentence_dataset.xlsx is missing column(s): {missing}")

    existing_ids = set(splits[ID_COL])
    new_ids = sorted(set(df[ID_COL]) - existing_ids)

    if not new_ids:
        sys.exit(
            "No Text_IDs in sentence_dataset.xlsx fall outside splits.csv - "
            "nothing to add.\nIf you were expecting new documents, confirm "
            "they have been merged into sentence_dataset.xlsx\nusing IDs "
            "that do not already appear in splits.csv.")

    # ---- governance guard: new IDs cannot be the sealed test forms ----
    test_ids = set(splits.loc[splits["split"] == "test", ID_COL])
    assert not (set(new_ids) & test_ids), \
        "New IDs overlap the sealed test set. Stopping - this should be " \
        "structurally impossible; investigate before proceeding."

    n_folds = int(splits.loc[splits["split"] == "train", "fold"].max()) + 1
    n_existing_train = int((splits["split"] == "train").sum())
    print(f"{len(new_ids)} new document(s) found "
          f"(not present in splits.csv).")
    print(f"Existing structure: {n_existing_train} training forms across "
          f"{n_folds} folds. Sealed test forms untouched: {len(test_ids)}.")

    # --- soft check: exact-duplicate sentence content vs existing corpus ---
    existing_ids_only = existing_ids
    existing_text = set(
        df.loc[df[ID_COL].isin(existing_ids_only)]
        .groupby(ID_COL)[TEXT_COL].apply(lambda s: " ".join(sorted(s.astype(str)))))
    new_text = (df.loc[df[ID_COL].isin(new_ids)]
                .groupby(ID_COL)[TEXT_COL].apply(lambda s: " ".join(sorted(s.astype(str)))))
    dupes = [tid for tid, txt in new_text.items() if txt in existing_text]
    if dupes:
        print(f"\n  WARNING: {len(dupes)} new document(s) have sentence text "
              f"identical to an existing document: {dupes}")
        print("  This may mean the keyword search re-selected a document "
              "already in the corpus under a new ID.")
        print("  Check these by hand before trusting this allocation.\n")

    # --- document-level aggregation ---
    doc_all = df.groupby(ID_COL)[TARGETS].max()

    # --- baseline counts from EXISTING assignments only ---
    counts = np.zeros((n_folds, len(TARGETS)), dtype=int)
    sizes = np.zeros(n_folds, dtype=int)
    existing_train = splits.loc[splits["split"] == "train", [ID_COL, "fold"]]
    unresolved = 0
    for tid, fold in zip(existing_train[ID_COL], existing_train["fold"]):
        if tid not in doc_all.index:
            unresolved += 1
            continue
        counts[int(fold)] += doc_all.loc[tid, TARGETS].to_numpy()
        sizes[int(fold)] += 1
    if unresolved:
        print(f"  NOTE: {unresolved} existing training form(s) in splits.csv "
              f"have no matching row in sentence_dataset.xlsx and were "
              f"excluded from baseline counts.")

    print("\nBaseline (existing 248 training forms):")
    print(fold_table(counts, sizes, n_folds))

    # --- order new documents: rarest global label combinations first ---
    new_doc_labels = doc_all.loc[new_ids, TARGETS]
    global_counts = new_doc_labels.sum(axis=0) + counts.sum(axis=0)
    rarity_weight = 1.0 / global_counts.replace(0, 1)
    rarity_score = (new_doc_labels * rarity_weight).sum(axis=1)
    order = rarity_score.sort_values(ascending=False).index.tolist()

    print("\nNew document label counts:")
    print(new_doc_labels.loc[order].sum().to_string())

    # --- greedy incremental allocation ---
    rng = np.random.default_rng(SEED)
    assignments = {}
    for tid in order:
        labels = new_doc_labels.loc[tid].to_numpy()
        pos_idx = np.where(labels == 1)[0]

        if len(pos_idx) == 0:
            candidates = np.where(sizes == sizes.min())[0]
            chosen = int(rng.choice(candidates))
        else:
            spreads = counts[:, pos_idx].max(axis=0) - counts[:, pos_idx].min(axis=0)
            driver = pos_idx[int(np.argmax(spreads))]
            candidates = np.where(counts[:, driver] == counts[:, driver].min())[0]
            if len(candidates) > 1:
                min_size = sizes[candidates].min()
                candidates = candidates[sizes[candidates] == min_size]
            chosen = int(rng.choice(candidates))

        assignments[tid] = chosen
        counts[chosen] += labels
        sizes[chosen] += 1

    print("\nAfter allocation (existing 248 + new documents):")
    print(fold_table(counts, sizes, n_folds))

    # --- build output: append only ---
    new_rows = pd.DataFrame({
        ID_COL: list(assignments.keys()),
        "split": "train",
        "fold": list(assignments.values()),
    })
    updated = pd.concat([splits, new_rows], ignore_index=True)

    assert updated[ID_COL].is_unique, \
        "Duplicate Text_ID after merge - stopping, do not use this output."
    check = updated.set_index(ID_COL).loc[splits[ID_COL], ["split", "fold"]]
    orig = splits.set_index(ID_COL)[["split", "fold"]]
    assert check.equals(orig), \
        "An existing row would be altered by this merge - stopping, do not " \
        "use this output. This should not be possible; investigate."

    updated.to_csv(OUTPUT_SPLITS, index=False)

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "script": "add_target_sampled_folds.py v1.0",
        "algorithm": "greedy incremental allocation, in the spirit of "
                     "Sechidis, Tsoumakas & Vlahavas (2011); not "
                     "MultilabelStratifiedKFold",
        "stratified_on": TARGETS,
        "seed": SEED,
        "n_existing_training_forms": n_existing_train,
        "n_new_documents": len(new_ids),
        "n_folds": n_folds,
        "duplicate_text_warnings": dupes,
        "baseline_counts": counts.tolist(),  # note: post-allocation values
        "new_assignments": {str(k): int(v) for k, v in assignments.items()},
        "output_file": OUTPUT_SPLITS,
        "note": "splits.csv itself was NOT modified. Review "
                "splits_updated.csv and replace splits.csv by hand.",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote {OUTPUT_SPLITS}")
    print(f"Wrote {MANIFEST_PATH}")
    print("\nsplits.csv has NOT been modified. Review splits_updated.csv, "
          "then replace splits.csv yourself once satisfied.")
    if dupes:
        print("Resolve the duplicate-text warning above before doing so.")


if __name__ == "__main__":
    main()

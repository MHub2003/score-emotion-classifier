"""
Produces a single splits file that both harnesses (SetFit and Classy) then read.

WHAT IT DOES
  1. Reads sentence_dataset.xlsx
  2. Aggregates to the FORM level: a form "has" a label if ANY of its
     sentences has it.
  3. Carves off TEST_SIZE forms as a locked test set, using multilabel
     stratification so the rare classes are spread as evenly as possible.
  4. Splits the remaining (training) forms into N_FOLDS stratified folds.
     These folds are for model development AND per-class threshold selection.
  5. Saves splits.csv:  Text_ID | split | fold
       split = 'train' or 'test'
       fold  = 0..N_FOLDS-1 for train forms, -1 for test forms
"""

import sys
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)

# ---- configuration ---------------------------------------------------------

INPUT_FILE  = [DIRECTORY PATH]
OUTPUT_FILE = [DIRECTORY PATH]

ID_COL = "Text_ID"

TARGETS = ["anxiety", "sadness", "overwhelm", "amotivation", "shame",
	   "hopelessness", "loneliness"]

TEST_SIZE = 50      
N_FOLDS   = 5       
SEED      = 42      

# ---- main ------------------------------------------------------------------

def main():
    df = pd.read_excel(INPUT_FILE)

    missing = [c for c in [ID_COL] + TARGETS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset is missing column(s): {missing}")

    # 1. aggregate sentences -> one row per form (label present if ANY sentence has it)
    doc = df.groupby(ID_COL)[TARGETS].max().reset_index()
    ids = doc[ID_COL].to_numpy()
    Y   = doc[TARGETS].to_numpy()
    n   = len(doc)
    print(f"{n} forms, {len(df)} sentences.")

    if TEST_SIZE >= n:
        sys.exit(f"ERROR: TEST_SIZE ({TEST_SIZE}) must be smaller than the "
                 f"number of forms ({n}).")

    # 2. stratified carve-off of the test forms
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(msss.split(np.zeros(n), Y))

    split = np.array(["train"] * n, dtype=object)
    split[test_idx] = "test"
    fold = np.full(n, -1, dtype=int)

    # 3. stratified folds within the training forms
    mskf = MultilabelStratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    Y_train = Y[train_idx]
    for k, (_, val_local) in enumerate(mskf.split(np.zeros(len(train_idx)), Y_train)):
        fold[train_idx[val_local]] = k

    out = pd.DataFrame({ID_COL: ids, "split": split, "fold": fold})
    out.to_csv(OUTPUT_FILE, index=False)

    # 4. report: where did each class land?
    tr = out["split"].to_numpy() == "train"
    te = ~tr
    print(f"\nSplit: {tr.sum()} train / {te.sum()} test  (seed {SEED}, {N_FOLDS} folds)")
    print("\n  Positive FORMS per class (train / test):")
    for j, t in enumerate(TARGETS):
        col = Y[:, j].astype(bool)
        print(f"    {t:<14}{int((col & tr).sum()):>4} / {int((col & te).sum()):>3}")

    # per-fold size sanity check
    print("\n  Forms per training fold:")
    for k in range(N_FOLDS):
        print(f"    fold {k}: {(fold == k).sum()}")

    print(f"\nWrote {OUTPUT_FILE}.")
    print("Both harnesses should read this file and never tune on the test forms.")


if __name__ == "__main__":
    main()

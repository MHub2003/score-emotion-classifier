"""
select_thresholds.py  (v1.0)
--------------------------------------------------------------------
Per-class classification threshold selection.

WHAT THIS DOES
  Reads the saved out-of-fold (OOF) probability files produced by
  cc_evaluate.py and evaluate_sentences.py, and for EACH emotion class
  independently chooses the probability threshold that maximises
  DOCUMENT-LEVEL F1 on the cross-validation predictions.

  No model is trained here and nothing is re-run. This script only reads
  probability files that already exist, so it takes seconds.
"""

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ============================== CONFIG ==============================
DATA_DIR = [DIRECTORY PATH]
SPLITS_CSV = DATA_DIR + r"\splits.csv"      # only needed for the optimism check

ID_COL = "Text_ID"

TARGETS = ["anxiety", "sadness", "overwhelm", "amotivation",
           "shame", "hopelessness", "loneliness"]

GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

SOURCES = [
    ("classy", "cc_oof_probabilities_*.csv"),
    ("setfit", "setfit_oof_probabilities_*.csv"),
]
# ====================================================================


def load_oof(path):
    """Read an OOF probability file and return (gold_doc, prob_frame, ids).

    Expected schema: Text_ID, the seven gold 0/1 columns, then p_<target>
    columns. Both harnesses write this identically.
    """
    df = pd.read_csv(path)
    prob_cols = [f"p_{t}" for t in TARGETS]
    missing = [c for c in [ID_COL] + TARGETS + prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {missing}")
    return df


def doc_level(df, target, threshold, subset=None):
    """Aggregate one class to form level.

    A form is predicted positive if ANY of its sentences has a probability at
    or above the threshold. Returns (gold, pred) arrays aligned by Text_ID.
    """
    d = df if subset is None else df.loc[subset]
    crossed = (d[f"p_{target}"].values >= threshold).astype(int)
    tmp = pd.DataFrame({ID_COL: d[ID_COL].values,
                        "gold": d[target].values,
                        "pred": crossed})
    agg = tmp.groupby(ID_COL).max()
    return agg["gold"].values, agg["pred"].values


def scores(gold, pred):
    """Precision, sensitivity, specificity, F1 and positive count."""
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return prec, sens, spec, f1, int(tp + fn)


def longest_run_midpoint(flags):
    """Given a boolean list over the grid, return the index at the middle of
    the longest consecutive True run, plus that run's length."""
    best_start = best_len = 0
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if (j - i + 1) > best_len:
                best_start, best_len = i, j - i + 1
            i = j + 1
        else:
            i += 1
    return best_start + (best_len - 1) // 2, best_len


def choose_threshold(df, target, subset=None):
    """Sweep the grid for one class and return the chosen threshold plus
    diagnostics. Selection rule: maximise document-level F1; on ties, take
    the midpoint of the longest tied run."""
    f1s = []
    for t in GRID:
        gold, pred = doc_level(df, target, t, subset)
        f1s.append(scores(gold, pred)[3])
    f1s = np.array(f1s)
    best = f1s.max()

    if best == 0.0:
        # Model never produces a correct positive at any threshold.
        return {"threshold": np.nan, "f1": 0.0, "plateau_width": 0,
                "degenerate": True, "at_grid_edge": False}

    tied = list(np.isclose(f1s, best))
    idx, width = longest_run_midpoint(tied)
    chosen = GRID[idx]
    return {"threshold": chosen, "f1": float(best), "plateau_width": int(width),
            "degenerate": False,
            "at_grid_edge": chosen in (GRID[0], GRID[-1]),
            "f1_curve": [float(v) for v in f1s]}


def optimism_check(df, target, fold_of):
    """Select a threshold on four folds, apply it to the fifth, and POOL.

    Gives an estimate of the tuned-threshold F1 that is not inflated by having
    selected the threshold on the same predictions being scored.
    """
    if fold_of is None:
        return np.nan, []
    folds = sorted(set(fold_of.dropna().values))
    if len(folds) < 2:
        return np.nan, []

    fold_series = df[ID_COL].map(fold_of)
    pooled_gold, pooled_pred, chosen = [], [], []
    for k in folds:
        tr = fold_series != k
        te = fold_series == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        pick = choose_threshold(df, target, subset=tr)
        if pick["degenerate"] or np.isnan(pick["threshold"]):
            continue                      # nothing learnable from these folds
        chosen.append(pick["threshold"])
        gold, pred = doc_level(df, target, pick["threshold"], subset=te)
        pooled_gold.append(gold)
        pooled_pred.append(pred)

    if not pooled_gold:
        return np.nan, []
    gold = np.concatenate(pooled_gold)
    pred = np.concatenate(pooled_pred)
    return float(scores(gold, pred)[3]), chosen


def process(path, method, tag, fold_of):
    df = load_oof(path)
    n_forms = df[ID_COL].nunique()
    print("\n" + "=" * 78)
    print(f"{method.upper()} / {tag}   ({os.path.basename(path)}; "
          f"{len(df)} sentences, {n_forms} forms)")
    print("=" * 78)
    print(f"  {'class':<14}{'thresh':>8}{'F1':>7}{'prec':>7}{'sens':>7}"
          f"{'spec':>7}{'pos':>6}{'plateau':>9}{'honest':>8}")

    rows, curves = [], {}
    for t in TARGETS:
        pick = choose_threshold(df, t)
        honest, fold_thrs = optimism_check(df, t, fold_of)

        if pick["degenerate"]:
            print(f"  {t:<14}{'--':>8}{0.0:>7.2f}{'':>7}{'':>7}{'':>7}{'':>6}"
                  f"{0:>9}{'--':>8}   NO THRESHOLD ACHIEVES F1 > 0")
            rows.append({"method": method, "backbone": tag, "class": t,
                         "threshold": "", "f1": 0.0, "precision": "",
                         "sensitivity": "", "specificity": "", "n_positive": "",
                         "plateau_width": 0, "f1_holdout": "",
                         "flag": "degenerate"})
            continue

        gold, pred = doc_level(df, t, pick["threshold"])
        prec, sens, spec, f1, pos = scores(gold, pred)
        flag = []
        if pick["at_grid_edge"]:
            flag.append("at_grid_edge")
        if pick["plateau_width"] >= 8:
            flag.append("very_flat")
        if not np.isnan(honest) and (f1 - honest) > 0.10:
            flag.append("large_optimism")

        print(f"  {t:<14}{pick['threshold']:>8.2f}{f1:>7.2f}{prec:>7.2f}"
              f"{sens:>7.2f}{spec:>7.2f}{pos:>6}{pick['plateau_width']:>9}"
              f"{honest:>8.2f}" + ("   " + ", ".join(flag) if flag else ""))

        rows.append({"method": method, "backbone": tag, "class": t,
                     "threshold": pick["threshold"], "f1": round(f1, 4),
                     "precision": round(prec, 4), "sensitivity": round(sens, 4),
                     "specificity": round(spec, 4), "n_positive": pos,
                     "plateau_width": pick["plateau_width"],
                     "f1_holdout": "" if np.isnan(honest) else round(honest, 4),
                     "fold_thresholds": ";".join(f"{v:.2f}" for v in fold_thrs),
                     "fold_threshold_range": (round(max(fold_thrs) - min(fold_thrs), 2)
                                              if fold_thrs else ""),
                     "flag": ";".join(flag)})
        curves[t] = pick["f1_curve"]

    tuned = [r["f1"] for r in rows if r["flag"] != "degenerate"]
    honestv = [r["f1_holdout"] for r in rows
               if r["flag"] != "degenerate" and r["f1_holdout"] != ""]
    print(f"\n  Macro F1 at selected thresholds : {np.mean(tuned):.3f}"
          if tuned else "\n  Macro F1: n/a")
    if honestv:
        print(f"  Macro F1, fold-wise holdout     : {np.mean(honestv):.3f}"
              f"   (difference = selection optimism)")

    out = pd.DataFrame(rows)
    out_path = DATA_DIR + rf"\thresholds_{method}_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    curve_path = DATA_DIR + rf"\threshold_curves_{method}_{tag}.csv"
    pd.DataFrame(curves, index=GRID).rename_axis("threshold").to_csv(curve_path)
    print(f"  Saved -> {curve_path}   (per-class F1 at every grid point)")

    return out


def main():
    wanted = sys.argv[1:]          # e.g. "cc_gte" or "setfit_bge"; blank = all

    fold_of = None
    if os.path.exists(SPLITS_CSV):
        s = pd.read_csv(SPLITS_CSV)
        fold_of = s.set_index(ID_COL)["fold"].replace(-1, np.nan)
    else:
        print("splits.csv not found - skipping the fold-wise optimism check.")

    found = []
    for method, pattern in SOURCES:
        for path in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
            base = os.path.basename(path)
            tag = base.replace("cc_oof_probabilities_", "") \
                      .replace("setfit_oof_probabilities_", "") \
                      .replace(".csv", "")
            key = ("cc_" if method == "classy" else "setfit_") + tag
            if wanted and key not in wanted and tag not in wanted:
                continue
            found.append((path, method, tag))

    if not found:
        sys.exit(f"No OOF probability files found in {DATA_DIR}. "
                 f"Run the harnesses first.")

    all_rows = []
    for path, method, tag in found:
        all_rows.append(process(path, method, tag, fold_of))

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = DATA_DIR + r"\thresholds_all_configurations.csv"
    combined.to_csv(combined_path, index=False)

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "selection_metric": "document-level F1, per class, independently",
        "aggregation_rule": "sentence -> form by maximum (OR rule)",
        "grid": GRID,
        "tie_break_rule": "midpoint of the longest run of tied-optimal thresholds",
        "optimism_check": "threshold selected on four folds, scored on the fifth",
        "files_processed": [os.path.basename(p) for p, _, _ in found],
    }
    with open(DATA_DIR + r"\threshold_selection_manifest.json", "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"Combined table -> {combined_path}")
    print("Selection rule recorded in threshold_selection_manifest.json")
    print("=" * 78)


if __name__ == "__main__":
    main()

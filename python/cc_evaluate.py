"""
--------------------------------------------------------------------
Method B (Classy Classification: frozen embeddings + one-vs-rest SVM)
development harness.

RUN (inside the `classy` conda env):
    conda activate classy
    python cc_evaluate.py              # all four backbones, in sequence
    python cc_evaluate.py bge          # one backbone only

"""

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# ============================== CONFIG ==============================
DATA_DIR   = [DIRECTORY PATH]
INPUT_XLSX = DATA_DIR + r"\sentence_dataset.xlsx"
SPLITS_CSV = DATA_DIR + r"\splits.csv"

ID_COL   = "Text_ID"
TEXT_COL = "Sentence"

TARGETS = ["anxiety", "sadness", "overwhelm", "amotivation",
           "shame", "hopelessness", "loneliness"]

# Bucket for sentences carrying none of the seven targets. These act as the
# this pool is "none of the seven targets".
NEGATIVE_LABEL = "neutral"

BACKBONES = {
    "minilm":   "sentence-transformers/all-MiniLM-L6-v2",
    "bge":      "BAAI/bge-small-en-v1.5",
    "gte":      "thenlper/gte-small",
    "pubmed":   "pritamdeka/S-PubMedBert-MS-MARCO",
}

# Package-default search space and explicit seed.
SEED = 42
SVC_CONFIG = {
    "C": [1, 2, 5, 10, 20, 50, 100],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "max_cross_validation_folds": 5,   # ignored when multi_label=True
    "seed": SEED,
}

DISABLE_BALANCED_WEIGHTING = True
SELECTION_THRESHOLD = 0.50                          # for cross-config comparison
SWEEP = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
# ====================================================================


def disable_classy_class_weight():
    """Force Classy's SVM to run UNWEIGHTED (class_weight=None), to match SetFit."""
    import classy_classification.classifiers.classy_skeleton as skel
    from sklearn.svm import SVC as RealSVC

    def svc_unweighted(*args, **kwargs):
        kwargs["class_weight"] = None          
        return RealSVC(*args, **kwargs)

    skel.SVC = svc_unweighted
    assert skel.SVC(class_weight="balanced").class_weight is None, \
        "class_weight override failed - Classy still weighting; do not trust results."
    print("Classy SVM weighting DISABLED (class_weight=None), matching SetFit.")


def environment_manifest():
    from importlib.metadata import version, PackageNotFoundError
    pkgs = ["classy-classification", "scikit-learn", "sentence-transformers",
            "transformers", "torch", "numpy", "pandas", "spacy", "openpyxl"]
    out = {"python": sys.version.split()[0]}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def build_label_dict(train_df):
    """Classy wants {label: [example sentences]}, positives only.

    Each target's list = sentences (in THIS train fold) carrying that target.
    A multi-label sentence appears under several keys; Classy de-duplicates by
    text and assigns it all of them. Sentences with NONE of the seven targets
    go under the negative label, where they serve as shared negatives for every
    SVM. Labels with no examples in this fold are omitted (handled at predict
    time by defaulting their probability to 0.0).
    """
    data = {}
    for tgt in TARGETS:
        examples = train_df.loc[train_df[tgt] == 1, TEXT_COL].tolist()
        if examples:
            data[tgt] = examples
    neg_mask = (train_df[TARGETS].sum(axis=1) == 0)
    neg_examples = train_df.loc[neg_mask, TEXT_COL].tolist()
    if neg_examples:
        data[NEGATIVE_LABEL] = neg_examples
    return data


def fit_fold(train_df, test_sentences, backbone):
    """Train Classy on one fold."""
    from classy_classification import ClassyClassifier

    data = build_label_dict(train_df)
    cc = ClassyClassifier(
        data=data, model=backbone, device="cpu",
        multi_label=True, config=SVC_CONFIG, verbose=False,
    )

    best_params, cv_rows = {}, []
    gs = getattr(cc, "clf", None)
    if hasattr(gs, "best_params_"):
        best_params = dict(gs.best_params_)
        best_params["best_score_f1_weighted"] = float(gs.best_score_)
        for params, mean, std in zip(gs.cv_results_["params"],
                                     gs.cv_results_["mean_test_score"],
                                     gs.cv_results_["std_test_score"]):
            cv_rows.append({**params, "mean_test_score": float(mean),
                            "std_test_score": float(std)})

    preds = cc.pipe(list(test_sentences))   # list of {label: proba} dicts
    rows = [{tgt: float(p.get(tgt, 0.0)) for tgt in TARGETS} for p in preds]
    return pd.DataFrame(rows, index=test_sentences.index), best_params, cv_rows


def per_class_metrics(y_true, y_pred):
    """Precision, sensitivity, specificity and F1 for one class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return prec, sens, spec, f1, int(tp + fn)


def aggregate_to_document(analysis_df, prob_cols, threshold):
    """Sentence -> Document aggregation"""
    gold_doc = analysis_df.groupby(ID_COL)[TARGETS].max()
    crossed = (analysis_df[prob_cols] >= threshold).astype(int)
    crossed.columns = TARGETS
    crossed[ID_COL] = analysis_df[ID_COL].values
    pred_doc = crossed.groupby(ID_COL)[TARGETS].max().loc[gold_doc.index]
    return gold_doc, pred_doc


def report_document_level(gold_doc, pred_doc, threshold):
    """Per-class table plus macro/micro averages. Returns macro F1."""
    print(f"\n=== Document-level (threshold {threshold:.2f}, training folds) ===")
    print(f"  {'label':<14}{'prec':>7}{'sens':>7}{'spec':>7}{'F1':>7}{'pos':>6}")
    specs = []
    for tgt in TARGETS:
        prec, sens, spec, f1, pos = per_class_metrics(gold_doc[tgt].values,
                                                      pred_doc[tgt].values)
        specs.append(spec)
        print(f"  {tgt:<14}{prec:>7.2f}{sens:>7.2f}{spec:>7.2f}{f1:>7.2f}{pos:>6}")

    macro_f1 = f1_score(gold_doc.values, pred_doc.values,
                        average="macro", zero_division=0)
    micro_f1 = f1_score(gold_doc.values, pred_doc.values,
                        average="micro", zero_division=0)
    tn = fp = 0
    for tgt in TARGETS:
        c = confusion_matrix(gold_doc[tgt].values, pred_doc[tgt].values,
                             labels=[0, 1])
        tn += c[0, 0]
        fp += c[0, 1]
    micro_spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"  {'macro avg':<14}{'':>7}{'':>7}{np.mean(specs):>7.2f}{macro_f1:>7.2f}")
    print(f"  {'micro avg':<14}{'':>7}{'':>7}{micro_spec:>7.2f}{micro_f1:>7.2f}")
    return float(macro_f1)


def run_backbone(tag, backbone, df, sent_fold, folds):
    print("\n" + "=" * 70)
    print(f"BACKBONE: {tag}  ({backbone})")
    print("=" * 70)
    t0 = time.time()

    prob_cols = [f"p_{tgt}" for tgt in TARGETS]
    oof = pd.DataFrame(0.0, index=df.index, columns=prob_cols)
    best_rows, cv_all = [], []

    for k in folds:
        te = np.where(sent_fold == k)[0]
        tr = np.where(sent_fold != k)[0]
        t1 = time.time()
        print(f"  Fold {k}: train={len(tr)} sentences, validate={len(te)} sentences")
        fold_probs, best, cv_rows = fit_fold(
            df.iloc[tr], df.iloc[te][TEXT_COL], backbone)
        oof.iloc[te] = fold_probs[TARGETS].values
        if best:
            print(f"    selected: C={best.get('estimator__C')} "
                  f"kernel={best.get('estimator__kernel')} "
                  f"(inner f1_weighted={best.get('best_score_f1_weighted', float('nan')):.3f})")
            best_rows.append({"backbone": tag, "fold": int(k), **best})
        for row in cv_rows:
            cv_all.append({"backbone": tag, "fold": int(k), **row})
        print(f"    fold time {time.time() - t1:.0f}s")

    # ---- persist artefacts, tagged by backbone ----
    oof_frame = pd.concat(
        [df[[ID_COL] + TARGETS].reset_index(drop=True), oof.reset_index(drop=True)],
        axis=1)
    oof_path = DATA_DIR + rf"\cc_oof_probabilities_{tag}.csv"
    oof_frame.to_csv(oof_path, index=False)
    print(f"\n  Saved OOF probabilities -> {oof_path}")

    if best_rows:
        bp_path = DATA_DIR + rf"\cc_best_params_{tag}.csv"
        pd.DataFrame(best_rows).to_csv(bp_path, index=False)
        print(f"  Saved selected hyperparameters -> {bp_path}")
    if cv_all:
        cv_path = DATA_DIR + rf"\cc_grid_results_{tag}.csv"
        pd.DataFrame(cv_all).to_csv(cv_path, index=False)
        print(f"  Saved full grid results -> {cv_path}")

    # ---- metrics ----
    analysis_df = df[[ID_COL] + TARGETS].copy()
    for pc, tgt in zip(prob_cols, TARGETS):
        analysis_df[pc] = oof[pc].values

    y_true = df[TARGETS].values
    y_pred = (oof[prob_cols].values >= SELECTION_THRESHOLD).astype(int)
    print(f"\n=== Sentence-level (threshold {SELECTION_THRESHOLD:.2f}) ===")
    print(f"  {'label':<14}{'prec':>7}{'sens':>7}{'spec':>7}{'F1':>7}{'pos':>6}")
    for i, tgt in enumerate(TARGETS):
        prec, sens, spec, f1, pos = per_class_metrics(y_true[:, i], y_pred[:, i])
        print(f"  {tgt:<14}{prec:>7.2f}{sens:>7.2f}{spec:>7.2f}{f1:>7.2f}{pos:>6}")

    gold_doc, pred_doc = aggregate_to_document(
        analysis_df, prob_cols, SELECTION_THRESHOLD)
    macro_f1 = report_document_level(gold_doc, pred_doc, SELECTION_THRESHOLD)

    print("\n=== Probability separation (mean predicted prob) ===")
    print(f"  {'label':<14}{'present':>9}{'absent':>9}")
    for i, tgt in enumerate(TARGETS):
        col = oof[prob_cols[i]].values
        pres = col[y_true[:, i] == 1].mean() if (y_true[:, i] == 1).any() else float("nan")
        absn = col[y_true[:, i] == 0].mean() if (y_true[:, i] == 0).any() else float("nan")
        print(f"  {tgt:<14}{pres:>9.3f}{absn:>9.3f}")

    print("\n=== Document-level F1 by threshold (common grid, training folds) ===")
    print("  thr  " + "".join(f"{t[:6]:>8}" for t in TARGETS) + f"{'macro':>8}")
    sweep_rows = []
    for thr in SWEEP:
        _, pred_t = aggregate_to_document(analysis_df, prob_cols, thr)
        f1s = [f1_score(gold_doc[t], pred_t[t], zero_division=0) for t in TARGETS]
        macro = f1_score(gold_doc.values, pred_t.values,
                         average="macro", zero_division=0)
        print(f"  {thr:0.2f} " + "".join(f"{v:8.2f}" for v in f1s) + f"{macro:8.2f}")
        sweep_rows.append({"backbone": tag, "threshold": thr,
                           **{t: f1s[i] for i, t in enumerate(TARGETS)},
                           "macro_f1": macro})
    sweep_path = DATA_DIR + rf"\cc_threshold_sweep_{tag}.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    print(f"\n  Saved threshold sweep -> {sweep_path}")

    elapsed = time.time() - t0
    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "method": "classy-classification",
        "backbone_tag": tag,
        "backbone": backbone,
        "targets": TARGETS,
        "svc_config": SVC_CONFIG,
        "class_weight": None if DISABLE_BALANCED_WEIGHTING else "balanced",
        "selection_threshold": SELECTION_THRESHOLD,
        "threshold_grid": SWEEP,
        "n_sentences": int(len(df)),
        "n_forms": int(df[ID_COL].nunique()),
        "n_folds": int(len(folds)),
        "document_macro_f1_at_selection_threshold": macro_f1,
        "elapsed_seconds": round(elapsed, 1),
        "environment": environment_manifest(),
    }
    man_path = DATA_DIR + rf"\cc_run_manifest_{tag}.json"
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Saved run manifest -> {man_path}")
    print(f"  Backbone {tag} finished in {elapsed/60:.1f} min. "
          f"Document macro F1 @ {SELECTION_THRESHOLD:.2f} = {macro_f1:.3f}")
    return macro_f1


def main():
    requested = sys.argv[1:] or list(BACKBONES)
    unknown = [r for r in requested if r not in BACKBONES]
    if unknown:
        sys.exit(f"Unknown backbone tag(s): {unknown}. Choose from {list(BACKBONES)}")

    if DISABLE_BALANCED_WEIGHTING:
        disable_classy_class_weight()
    else:
        print("Classy SVM weighting BALANCED (Classy's default).")

    splits    = pd.read_csv(SPLITS_CSV)
    train_ids = set(splits.loc[splits["split"] == "train", ID_COL])
    test_ids  = set(splits.loc[splits["split"] == "test",  ID_COL])
    fold_of   = dict(zip(splits[ID_COL], splits["fold"]))
    folds     = sorted(splits.loc[splits["split"] == "train", "fold"].unique())

    df = pd.read_excel(INPUT_XLSX)
    missing = [c for c in [ID_COL, TEXT_COL] + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")

    df = df[df[ID_COL].isin(train_ids)].reset_index(drop=True)

    # ensuring the reserved 50 are unreachable.
    leaked = test_ids & set(df[ID_COL])
    assert not leaked, f"TEST FORMS PRESENT IN TRAINING DATA: {sorted(leaked)}"
    unassigned = set(df[ID_COL]) - set(fold_of)
    assert not unassigned, f"Training forms with no fold in splits.csv: {sorted(unassigned)}"

    sent_fold = df[ID_COL].map(fold_of).values
    print(f"{len(df)} training sentences across {df[ID_COL].nunique()} training forms; "
          f"{len(folds)} folds. {len(test_ids)} test forms excluded and verified absent.")

    results = {}
    for tag in requested:
        results[tag] = run_backbone(tag, BACKBONES[tag], df, sent_fold, folds)

    print("\n" + "=" * 70)
    print("SUMMARY - document-level macro F1 @ threshold "
          f"{SELECTION_THRESHOLD:.2f} (training folds)")
    for tag, score in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:<10}{score:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

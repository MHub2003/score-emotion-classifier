"""
evaluate_test_set.py  (v1.0)
--------------------------------------------------------------------
FINAL EVALUATION on the reserved test forms.

WHAT IT DOES
  1. Trains the selected configuration ONCE on ALL training forms.
  2. Predicts sentence-level probabilities for the evaluation forms.
  3. Applies the per-class thresholds recorded by select_thresholds.py.
     Thresholds are READ FROM FILE, never chosen here.
  4. Aggregates to document level and reports precision, sensitivity, 
     specificity and F1 per class, plus macro and micro averages.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# ============================== CONFIG ==============================
DATA_DIR   = [DIRECTORY PATH]
INPUT_XLSX = os.path.join(DATA_DIR, "sentence_dataset.xlsx")
SPLITS_CSV = os.path.join(DATA_DIR, "splits.csv")

ID_COL   = "Text_ID"
TEXT_COL = "Sentence"

TARGETS = ["anxiety", "sadness", "overwhelm", "amotivation",
           "shame", "hopelessness", "loneliness"]

BACKBONES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
    "gte":    "thenlper/gte-small",
    "pubmed": "pritamdeka/S-PubMedBert-MS-MARCO",
}

SEED = 42

# Must match cc_evaluate.py exactly.
SVC_CONFIG = {
    "C": [1, 2, 5, 10, 20, 50, 100],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "max_cross_validation_folds": 5,
    "seed": SEED,
}
NEGATIVE_LABEL = "neutral"

# Must match evaluate_sentences.py exactly.
BATCH_SIZE = 16
MAX_STEPS  = 300
NUM_EPOCHS = 1
MULTI_TARGET_STRATEGY = "one-vs-rest"

CONFIRM_PHRASE = "EVALUATE SEALED TEST SET"
DRY_RUN_HOLDOUT = 50
# ====================================================================


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_manifest(method):
    from importlib.metadata import version, PackageNotFoundError
    common = ["scikit-learn", "sentence-transformers", "transformers",
              "torch", "numpy", "pandas", "openpyxl"]
    pkgs = common + (["classy-classification"] if method == "classy"
                     else ["setfit", "datasets"])
    out = {"python": sys.version.split()[0]}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def per_class_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return prec, sens, spec, f1, int(tp + fn), int(tp), int(fp), int(fn), int(tn)


# ------------------------- threshold loading -------------------------

def load_thresholds(method, tag):
    """Read locked per-class thresholds. Never selects anything."""
    path = os.path.join(DATA_DIR, f"thresholds_{method}_{tag}.csv")
    if not os.path.exists(path):
        sys.exit(f"Threshold file not found: {path}\n"
                 f"Run select_thresholds.py before final evaluation.")
    df = pd.read_csv(path)
    thr, missing = {}, []
    for cls in TARGETS:
        row = df.loc[df["class"] == cls]
        if row.empty or pd.isna(row.iloc[0]["threshold"]) or row.iloc[0]["threshold"] == "":
            missing.append(cls)
        else:
            thr[cls] = float(row.iloc[0]["threshold"])
    if missing:
        sys.exit(f"No usable threshold for: {missing}. Resolve before evaluating.")
    return thr, path


# ------------------------- model fitting -------------------------

def fit_predict_classy(train_df, eval_texts, backbone):
    """Train Classy on all training sentences; return probability array."""
    import classy_classification.classifiers.classy_skeleton as skel
    from sklearn.svm import SVC as RealSVC

    def svc_unweighted(*args, **kwargs):
        kwargs["class_weight"] = None
        return RealSVC(*args, **kwargs)

    skel.SVC = svc_unweighted
    assert skel.SVC(class_weight="balanced").class_weight is None, \
        "class_weight override failed - do not trust results."
    print("  Classy SVM weighting disabled (class_weight=None).")

    from classy_classification import ClassyClassifier

    data = {}
    for tgt in TARGETS:
        ex = train_df.loc[train_df[tgt] == 1, TEXT_COL].tolist()
        if ex:
            data[tgt] = ex
    neg = train_df.loc[train_df[TARGETS].sum(axis=1) == 0, TEXT_COL].tolist()
    if neg:
        data[NEGATIVE_LABEL] = neg
    print(f"  Training classes: {sorted(data)} "
          f"({sum(len(v) for v in data.values())} labelled examples)")

    cc = ClassyClassifier(data=data, model=backbone, device="cpu",
                          multi_label=True, config=SVC_CONFIG, verbose=False)

    selected = {}
    gs = getattr(cc, "clf", None)
    if hasattr(gs, "best_params_"):
        selected = {k: v for k, v in gs.best_params_.items()}
        selected["best_score_f1_weighted"] = float(gs.best_score_)
        print(f"  Selected: C={selected.get('estimator__C')} "
              f"kernel={selected.get('estimator__kernel')}")

    preds = cc.pipe(list(eval_texts))
    proba = np.array([[float(p.get(t, 0.0)) for t in TARGETS] for p in preds])
    return proba, {"selected_hyperparameters": selected}


def fit_predict_setfit(train_df, eval_texts, backbone):
    """Fine-tune SetFit on all training sentences; return probability array."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_ds = Dataset.from_dict({
        "text": train_df[TEXT_COL].astype(str).tolist(),
        "label": train_df[TARGETS].values.tolist(),
    })
    args = TrainingArguments(
        output_dir=os.path.join(os.environ.get("TEMP", r"C:\Temp"),
                                "setfit_scratch", "final"),
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, max_steps=MAX_STEPS,
        seed=SEED, show_progress_bar=False,
        save_strategy="no",   # see evaluate_sentences.py for why
    )
    model = SetFitModel.from_pretrained(
        backbone, multi_target_strategy=MULTI_TARGET_STRATEGY)
    Trainer(model=model, args=args, train_dataset=train_ds).train()

    proba = np.asarray(model.predict_proba(list(eval_texts), as_numpy=True),
                       dtype=float)
    head = getattr(model, "model_head", None)
    info = {
        "training_arguments": {k: str(v) for k, v in args.to_dict().items()},
        "head_type": type(head).__name__ if head is not None else None,
    }
    try:
        info["n_fitted_binary_classifiers"] = len(head.estimators_)
    except Exception:
        pass
    return proba, info


# ------------------------- reporting -------------------------

def report(gold_doc, pred_doc, thresholds, mode):
    print(f"\n{'=' * 78}")
    print(f"RESULTS - {'SEALED TEST SET' if mode == 'final' else 'DRY RUN (simulated holdout)'}"
          f"   {len(gold_doc)} forms, document level")
    print("=" * 78)
    print(f"  {'class':<14}{'thr':>6}{'prec':>7}{'sens':>7}{'spec':>7}{'F1':>7}"
          f"{'pos':>5}{'TP':>5}{'FP':>5}{'FN':>5}")
    rows, specs = [], []
    for i, cls in enumerate(TARGETS):
        prec, sens, spec, f1, pos, tp, fp, fn, tn = per_class_metrics(
            gold_doc[cls].values, pred_doc[cls].values)
        specs.append(spec)
        warn = "   few positives" if pos < 5 else ""
        print(f"  {cls:<14}{thresholds[cls]:>6.2f}{prec:>7.2f}{sens:>7.2f}"
              f"{spec:>7.2f}{f1:>7.2f}{pos:>5}{tp:>5}{fp:>5}{fn:>5}{warn}")
        rows.append({"class": cls, "threshold": thresholds[cls],
                     "precision": round(prec, 4), "sensitivity": round(sens, 4),
                     "specificity": round(spec, 4), "f1": round(f1, 4),
                     "n_positive_forms": pos, "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn})

    macro_f1 = f1_score(gold_doc.values, pred_doc.values,
                        average="macro", zero_division=0)
    micro_f1 = f1_score(gold_doc.values, pred_doc.values,
                        average="micro", zero_division=0)
    tn_t = fp_t = 0
    for cls in TARGETS:
        c = confusion_matrix(gold_doc[cls].values, pred_doc[cls].values,
                             labels=[0, 1])
        tn_t += c[0, 0]
        fp_t += c[0, 1]
    micro_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) else 0.0
    print(f"  {'macro avg':<14}{'':>6}{'':>7}{'':>7}{np.mean(specs):>7.2f}{macro_f1:>7.2f}")
    print(f"  {'micro avg':<14}{'':>6}{'':>7}{'':>7}{micro_spec:>7.2f}{micro_f1:>7.2f}")

    for label, spec_v, f1_v in [("macro avg", float(np.mean(specs)), float(macro_f1)),
                                ("micro avg", float(micro_spec), float(micro_f1))]:
        rows.append({"class": label, "threshold": "", "precision": "",
                     "sensitivity": "", "specificity": round(spec_v, 4),
                     "f1": round(f1_v, 4), "n_positive_forms": "",
                     "tp": "", "fp": "", "fn": "", "tn": ""})
    return pd.DataFrame(rows), float(macro_f1)


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["classy", "setfit"])
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--final", action="store_true",
                    help="Evaluate the SEALED test forms. Otherwise a dry run.")
    a = ap.parse_args()

    mode = "final" if a.final else "dry"
    backbone = BACKBONES[a.backbone]
    stem = f"{a.method}_{a.backbone}_{mode}"
    pred_path = os.path.join(DATA_DIR, f"test_predictions_{stem}.csv")
    met_path  = os.path.join(DATA_DIR, f"test_metrics_{stem}.csv")
    man_path  = os.path.join(DATA_DIR, f"test_run_manifest_{stem}.json")

    # ---- one-shot protection ----
    if mode == "final" and os.path.exists(met_path):
        sys.exit(
            f"\nREFUSING TO RUN.\n{met_path}\nalready exists, so this "
            f"configuration has already been evaluated against the sealed "
            f"test set.\nThe sealed set provides one unbiased estimate; "
            f"repeating it does not.\nIf a re-run is genuinely warranted, "
            f"delete that file by hand and be prepared\nto state in the "
            f"write-up why the test set was used more than once.\n")

    thresholds, thr_path = load_thresholds(a.method, a.backbone)

    # ---- data ----
    splits = pd.read_csv(SPLITS_CSV)
    train_ids = set(splits.loc[splits["split"] == "train", ID_COL])
    test_ids  = set(splits.loc[splits["split"] == "test",  ID_COL])

    df = pd.read_excel(INPUT_XLSX)
    missing = [c for c in [ID_COL, TEXT_COL] + TARGETS if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}")

    if mode == "final":
        eval_ids = test_ids
        fit_ids  = train_ids
    else:
        rng = np.random.default_rng(SEED)
        pool = np.array(sorted(train_ids))
        eval_ids = set(rng.choice(pool, size=min(DRY_RUN_HOLDOUT, len(pool)),
                                  replace=False).tolist())
        fit_ids = train_ids - eval_ids

    train_df = df[df[ID_COL].isin(fit_ids)].reset_index(drop=True)
    eval_df  = df[df[ID_COL].isin(eval_ids)].reset_index(drop=True)

    # ---- guards ----
    assert not (fit_ids & eval_ids), "Training and evaluation forms overlap."
    if mode == "dry":
        assert not (set(eval_df[ID_COL]) & test_ids), \
            "Dry run must not touch sealed forms."
    assert len(eval_df) > 0, "No evaluation sentences found."
    assert len(train_df) > 0, "No training sentences found."

    # ---- banner and confirmation ----
    print("=" * 78)
    print(f"{'FINAL EVALUATION - SEALED TEST SET' if mode == 'final' else 'DRY RUN - no sealed data is read'}")
    print("=" * 78)
    print(f"  method            {a.method}")
    print(f"  backbone          {a.backbone}  ({backbone})")
    print(f"  training forms    {len(fit_ids)}  ({len(train_df)} sentences)")
    print(f"  evaluation forms  {len(eval_ids)}  ({len(eval_df)} sentences)")
    print(f"  thresholds from   {os.path.basename(thr_path)}")
    for cls in TARGETS:
        print(f"      {cls:<14}{thresholds[cls]:.2f}")

    if mode == "final":
        print("\n  This reads the sealed test forms and should happen once.")
        print(f"  Type exactly:  {CONFIRM_PHRASE}")
        try:
            typed = input("  > ").strip()
        except EOFError:
            sys.exit("\nNo terminal input available. Run --final interactively.")
        if typed != CONFIRM_PHRASE:
            sys.exit("Confirmation did not match. Nothing was run.")

    # ---- fit and predict ----
    t0 = time.time()
    print(f"\nTraining on {len(train_df)} sentences...")
    if a.method == "classy":
        proba, model_info = fit_predict_classy(train_df, eval_df[TEXT_COL], backbone)
    else:
        proba, model_info = fit_predict_setfit(train_df, eval_df[TEXT_COL], backbone)
    print(f"Trained and predicted in {(time.time() - t0)/60:.1f} min.")

    # ---- threshold, aggregate, score ----
    thr_vec = np.array([thresholds[c] for c in TARGETS])
    sent_pred = (proba >= thr_vec).astype(int)

    pred_doc = (pd.DataFrame(sent_pred, columns=TARGETS)
                .assign(**{ID_COL: eval_df[ID_COL].values})
                .groupby(ID_COL)[TARGETS].max())
    gold_doc = (eval_df.groupby(ID_COL)[TARGETS].max().loc[pred_doc.index])

    out = pd.concat([
        eval_df[[ID_COL] + TARGETS].reset_index(drop=True),
        pd.DataFrame(proba, columns=[f"p_{t}" for t in TARGETS]),
        pd.DataFrame(sent_pred, columns=[f"pred_{t}" for t in TARGETS]),
    ], axis=1)
    out.to_csv(pred_path, index=False)

    metrics, macro_f1 = report(gold_doc, pred_doc, thresholds, mode)
    metrics.insert(0, "backbone", a.backbone)
    metrics.insert(0, "method", a.method)
    metrics.to_csv(met_path, index=False)

    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "script": "evaluate_test_set.py v1.0",
        "mode": mode,
        "method": a.method,
        "backbone_tag": a.backbone,
        "backbone": backbone,
        "targets": TARGETS,
        "thresholds": thresholds,
        "threshold_source": os.path.basename(thr_path),
        "n_training_forms": len(fit_ids),
        "n_training_sentences": int(len(train_df)),
        "n_evaluation_forms": len(eval_ids),
        "n_evaluation_sentences": int(len(eval_df)),
        "document_macro_f1": macro_f1,
        "model": model_info,
        "input_hashes": {
            os.path.basename(INPUT_XLSX): sha256(INPUT_XLSX),
            os.path.basename(SPLITS_CSV): sha256(SPLITS_CSV),
            os.path.basename(thr_path): sha256(thr_path),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "environment": environment_manifest(a.method),
    }
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved predictions -> {pred_path}")
    print(f"Saved metrics     -> {met_path}")
    print(f"Saved manifest    -> {man_path}")
    if mode == "dry":
        print("\nDry run complete. No sealed data was read. Re-run with --final "
              "when you are ready to spend the seal.")
    else:
        print("\nSealed test set evaluated. Do not run this configuration again.")


if __name__ == "__main__":
    main()

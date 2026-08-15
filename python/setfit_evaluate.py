"""
evaluate_sentences.py  (v0.4 - 7 targets, instrumented, multi-backbone)
--------------------------------------------------------------------
Method A (SetFit: contrastive fine-tuning + one-vs-rest logistic head)
development harness, kept in lock-step with cc_evaluate.py v0.4.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, f1_score

# ============================== CONFIG ==============================
DATA_DIR   = [DIRECTORY PATH]
INPUT_XLSX = DATA_DIR + r"\sentence_dataset.xlsx"
SPLITS_CSV = DATA_DIR + r"\splits.csv"

CHECKPOINTS = os.path.join(os.environ.get("TEMP", r"C:\Temp"), "setfit_scratch")

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

SEED       = 42
BATCH_SIZE = 16
MAX_STEPS  = 300
NUM_EPOCHS = 1
MULTI_TARGET_STRATEGY = "one-vs-rest"

SELECTION_THRESHOLD = 0.50                          
SWEEP = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
SHOW_PROGRESS = False
# ====================================================================


def environment_manifest():
    from importlib.metadata import version, PackageNotFoundError
    pkgs = ["setfit", "scikit-learn", "sentence-transformers", "transformers",
            "torch", "datasets", "numpy", "pandas", "openpyxl"]
    out = {"python": sys.version.split()[0]}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def safe_write(write_fn, dest_path, description):
    """Write to the secure drive, falls back to local if VPN disconnects"""
    try:
        write_fn(dest_path)
        return dest_path, False
    except OSError as e:
        fallback_dir = os.path.join(os.environ.get("TEMP", r"C:\Temp"),
                                    "setfit_scratch", "rescued_outputs")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, os.path.basename(dest_path))
        write_fn(fallback_path)
        print(f"  WARNING: could not write {description} to the secure drive "
              f"({e}).")
        print(f"           Wrote to {fallback_path} instead - copy it to "
              f"{os.path.dirname(dest_path)} once reconnected.")
        return fallback_path, True


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return getattr(obj, "__name__", None) or str(obj)


def describe_head(model):
    """Read the fitted head's parameters off the object"""
    head = getattr(model, "model_head", None)
    info = {"head_type": type(head).__name__ if head is not None else None}
    try:
        info["head_params"] = jsonable(head.get_params(deep=False))
    except Exception:
        pass
    inner = getattr(head, "estimator", None)
    if inner is not None:
        info["inner_estimator"] = type(inner).__name__
        try:
            info["inner_params"] = jsonable(inner.get_params())
        except Exception:
            pass
    try:
        info["n_fitted_binary_classifiers"] = len(head.estimators_)
    except Exception:
        pass
    return info


def train_and_predict(train_texts, train_labels, test_texts, backbone, args):
    """Fine-tune SetFit on one fold; return (proba, body_info, head_info)."""
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    model = SetFitModel.from_pretrained(
        backbone, multi_target_strategy=MULTI_TARGET_STRATEGY)
    Trainer(model=model, args=args, train_dataset=train_ds).train()

    proba = model.predict_proba(test_texts, as_numpy=True)
    body_info = {
        "max_seq_length": int(getattr(model.model_body, "max_seq_length", -1)),
        "embedding_dimension": int(model.model_body.get_sentence_embedding_dimension()),
    }
    head_info = describe_head(model)
    del model
    gc.collect()
    return np.asarray(proba, dtype=float), body_info, head_info


def per_class_metrics(y_true, y_pred):
    """Precision, sensitivity, specificity and F1 for one class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return prec, sens, spec, f1, int(tp + fn)


def aggregate_to_document(gold_df, proba, groups, threshold):
    """Sentence -> Document"""
    crossed = pd.DataFrame((proba >= threshold).astype(int), columns=TARGETS)
    crossed[ID_COL] = groups
    pred_doc = crossed.groupby(ID_COL)[TARGETS].max()
    gold_doc = gold_df.groupby(ID_COL)[TARGETS].max().loc[pred_doc.index]
    return gold_doc, pred_doc


def metrics_table(gold, pred, level, tag, threshold):
    """Per-class rows plus macro/micro, printed and returned for CSV."""
    print(f"\n=== {level} (threshold {threshold:.2f}, training folds) ===")
    print(f"  {'label':<14}{'prec':>7}{'sens':>7}{'spec':>7}{'F1':>7}{'pos':>6}")
    rows, specs = [], []
    gold_arr = gold.values if hasattr(gold, "values") else gold
    pred_arr = pred.values if hasattr(pred, "values") else pred
    for i, t in enumerate(TARGETS):
        prec, sens, spec, f1, pos = per_class_metrics(gold_arr[:, i], pred_arr[:, i])
        specs.append(spec)
        print(f"  {t:<14}{prec:>7.2f}{sens:>7.2f}{spec:>7.2f}{f1:>7.2f}{pos:>6}")
        rows.append({"backbone": tag, "level": level, "threshold": threshold,
                     "label": t, "precision": prec, "sensitivity": sens,
                     "specificity": spec, "f1": f1, "n_positive": pos})

    macro_f1 = f1_score(gold_arr, pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(gold_arr, pred_arr, average="micro", zero_division=0)
    tn = fp = 0
    for i in range(len(TARGETS)):
        c = confusion_matrix(gold_arr[:, i], pred_arr[:, i], labels=[0, 1])
        tn += c[0, 0]
        fp += c[0, 1]
    micro_spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"  {'macro avg':<14}{'':>7}{'':>7}{np.mean(specs):>7.2f}{macro_f1:>7.2f}")
    print(f"  {'micro avg':<14}{'':>7}{'':>7}{micro_spec:>7.2f}{micro_f1:>7.2f}")
    rows.append({"backbone": tag, "level": level, "threshold": threshold,
                 "label": "macro avg", "precision": "", "sensitivity": "",
                 "specificity": float(np.mean(specs)), "f1": float(macro_f1),
                 "n_positive": ""})
    rows.append({"backbone": tag, "level": level, "threshold": threshold,
                 "label": "micro avg", "precision": "", "sensitivity": "",
                 "specificity": float(micro_spec), "f1": float(micro_f1),
                 "n_positive": ""})
    return rows, float(macro_f1)


def run_backbone(tag, backbone, df, sent_fold, folds):
    print("\n" + "=" * 70)
    print(f"BACKBONE: {tag}  ({backbone})")
    print("=" * 70)
    t0 = time.time()

    os.makedirs(CHECKPOINTS, exist_ok=True)
    args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINTS, tag),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        seed=SEED,
        show_progress_bar=SHOW_PROGRESS,
        save_strategy="no",
    )

    X      = df[TEXT_COL].astype(str).tolist()
    Y      = df[TARGETS].values
    groups = df[ID_COL].values
    oof_proba = np.zeros(Y.shape, dtype=float)
    body_info = head_info = None

    for k in folds:
        te = np.where(sent_fold == k)[0]
        tr = np.where(sent_fold != k)[0]
        t1 = time.time()
        print(f"  Fold {k}: train={len(tr)} sentences, validate={len(te)} sentences")
        proba, body_info, head_info = train_and_predict(
            [X[i] for i in tr], Y[tr].tolist(), [X[i] for i in te], backbone, args)
        oof_proba[te] = proba
        print(f"    fold time {(time.time() - t1)/60:.1f} min")

    # ---- persist probabilities, tagged, in the same schema as the CC harness ----
    oof_frame = pd.concat(
        [df[[ID_COL] + TARGETS].reset_index(drop=True),
         pd.DataFrame(oof_proba, columns=[f"p_{t}" for t in TARGETS])],
        axis=1)
    csv_path = DATA_DIR + rf"\setfit_oof_probabilities_{tag}.csv"
    csv_path, csv_rescued = safe_write(
        lambda p: oof_frame.to_csv(p, index=False), csv_path,
        "OOF probabilities (CSV)")
    npz_path = DATA_DIR + rf"\oof_probabilities_{tag}.npz"
    npz_path, npz_rescued = safe_write(
        lambda p: np.savez(p, proba=oof_proba, Y=Y, groups=groups,
                           targets=np.array(TARGETS)),
        npz_path, "OOF probabilities (NPZ)")
    print(f"\n  Saved OOF probabilities -> {csv_path}")
    print(f"  Saved OOF probabilities -> {npz_path}")

    # ---- metrics ----
    all_rows = []
    y_pred = (oof_proba >= SELECTION_THRESHOLD).astype(int)
    rows, _ = metrics_table(Y, y_pred, "Sentence-level", tag, SELECTION_THRESHOLD)
    all_rows += rows

    gold_df = df[[ID_COL] + TARGETS]
    gold_doc, pred_doc = aggregate_to_document(
        gold_df, oof_proba, groups, SELECTION_THRESHOLD)
    rows, macro_f1 = metrics_table(gold_doc, pred_doc, "Document-level",
                                   tag, SELECTION_THRESHOLD)
    all_rows += rows

    met_path = DATA_DIR + rf"\setfit_metrics_{tag}.csv"
    met_path, _ = safe_write(
        lambda p: pd.DataFrame(all_rows).to_csv(p, index=False), met_path,
        "metrics")
    print(f"\n  Saved metrics -> {met_path}")

    print("\n=== Probability separation (mean predicted prob) ===")
    print(f"  {'label':<14}{'present':>9}{'absent':>9}")
    for i, t in enumerate(TARGETS):
        pres = oof_proba[Y[:, i] == 1, i].mean() if (Y[:, i] == 1).any() else float("nan")
        absn = oof_proba[Y[:, i] == 0, i].mean() if (Y[:, i] == 0).any() else float("nan")
        print(f"  {t:<14}{pres:>9.3f}{absn:>9.3f}")

    print("\n=== Document-level F1 by threshold (common grid, training folds) ===")
    print("  thr  " + "".join(f"{t[:6]:>8}" for t in TARGETS) + f"{'macro':>8}")
    sweep_rows = []
    for thr in SWEEP:
        g, p = aggregate_to_document(gold_df, oof_proba, groups, thr)
        f1s = [f1_score(g[t], p[t], zero_division=0) for t in TARGETS]
        macro = f1_score(g.values, p.values, average="macro", zero_division=0)
        print(f"  {thr:0.2f} " + "".join(f"{v:8.2f}" for v in f1s) + f"{macro:8.2f}")
        sweep_rows.append({"backbone": tag, "threshold": thr,
                           **{t: f1s[i] for i, t in enumerate(TARGETS)},
                           "macro_f1": macro})
    sweep_path = DATA_DIR + rf"\setfit_threshold_sweep_{tag}.csv"
    sweep_path, _ = safe_write(
        lambda p: pd.DataFrame(sweep_rows).to_csv(p, index=False), sweep_path,
        "threshold sweep")
    print(f"\n  Saved threshold sweep -> {sweep_path}")

    elapsed = time.time() - t0
    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "method": "setfit",
        "backbone_tag": tag,
        "backbone": backbone,
        "targets": TARGETS,
        "multi_target_strategy": MULTI_TARGET_STRATEGY,
        "training_arguments_resolved": jsonable(args.to_dict()),
        "contrastive_pairs_budget": MAX_STEPS * BATCH_SIZE,
        "model_body": body_info,
        "model_head": head_info,
        "selection_threshold": SELECTION_THRESHOLD,
        "threshold_grid": SWEEP,
        "n_sentences": int(len(df)),
        "n_forms": int(df[ID_COL].nunique()),
        "n_folds": int(len(folds)),
        "document_macro_f1_at_selection_threshold": macro_f1,
        "elapsed_seconds": round(elapsed, 1),
        "environment": environment_manifest(),
    }
    man_path = DATA_DIR + rf"\setfit_run_manifest_{tag}.json"
    man_path, man_rescued = safe_write(
        lambda p: json.dump(manifest, open(p, "w", encoding="utf-8"), indent=2),
        man_path, "run manifest")
    print(f"  Saved run manifest -> {man_path}")
    if csv_rescued or npz_rescued or man_rescued:
        print("  ONE OR MORE OUTPUTS WERE RESCUED TO LOCAL SCRATCH - "
              "copy them across before this machine is used for anything else.")
    print(f"  Backbone {tag} finished in {elapsed/60:.1f} min. "
          f"Document macro F1 @ {SELECTION_THRESHOLD:.2f} = {macro_f1:.3f}")
    return macro_f1


def main():
    requested = sys.argv[1:] or list(BACKBONES)
    unknown = [r for r in requested if r not in BACKBONES]
    if unknown:
        sys.exit(f"Unknown backbone tag(s): {unknown}. Choose from {list(BACKBONES)}")

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

    # Ensure reserved 50 are unreachable
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

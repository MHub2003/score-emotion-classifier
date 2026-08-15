"""
Converts a filled sentence-level annotation template into a modelling-ready
binary dataset, with validation and per-label counts.
"""

import re
import sys
import pandas as pd

# ---- configuration ---------------------------------------------------------

INPUT_FILE  = [DIRECTORY PATH]
OUTPUT_FILE = [DIRECTORY PATH]


ID_COL   = "Text_ID"
TEXT_COL = "Sentence"
TAG_COL  = "Classifications"

EMOTION_TARGETS  = ["anxiety", "sadness", "overwhelm", "amotivation",
                    "shame", "hopelessness", "loneliness"]

RECORDED_ONLY    = ["other", "neutral"]                    
VOCAB            = EMOTION_TARGETS + RECORDED_ONLY                 
TARGETS = EMOTION_TARGETS

RESIDUAL = "neutral"

# ---- helpers ---------------------------------------------------------------

def parse_tags(cell):
    """Split a Classifications cell into a clean, lower-cased list of tags.
    Tolerates ';' or ',' separators and stray whitespace."""
    if pd.isna(cell):
        return []
    parts = re.split(r"[;,]", str(cell))
    return [p.strip().lower() for p in parts if p.strip()]


def validate(df):
    """Return a list of human-readable problems. Empty list == clean."""
    errors = []
    for i, row in df.iterrows():
        tags = parse_tags(row[TAG_COL])
        where = f"row {i + 2} (Text_ID={row[ID_COL]})"   # +2: header + 1-indexing

        # 1. every sentence must carry at least one tag
        if not tags:
            errors.append(f"{where}: no tags. Every sentence needs one; "
                          f"use 'neutral' if it is purely factual.")
            continue

        # 2. only controlled-vocabulary tags allowed (catches typos)
        for t in tags:
            if t not in VOCAB:
                errors.append(f"{where}: unknown tag '{t}' (not in the vocabulary).")

        # 3. 'neutral' must sit alone
        if RESIDUAL in tags and len(set(tags)) > 1:
            others = ", ".join(sorted(set(tags) - {RESIDUAL}))
            errors.append(f"{where}: 'neutral' cannot co-occur with other tags "
                          f"(also found: {others}).")
    return errors


def build_binary(df):
    """Add one 0/1 column per vocabulary tag."""
    tag_lists = df[TAG_COL].apply(parse_tags)
    for tag in VOCAB:
        df[tag] = tag_lists.apply(lambda ts, tag=tag: int(tag in ts))
    return df


def report(df):
    n_forms = df[ID_COL].nunique()
    n_sent  = len(df)
    neg     = int((df[TARGETS].sum(axis=1) == 0).sum())

    print("=" * 50)
    print(f"  {n_forms} forms, {n_sent} sentences")
    print("=" * 50)
    print("\n  Emotions (targets):")
    for t in EMOTION_TARGETS:
        print(f"    {t:<14}{int(df[t].sum()):>5}")
    print("\n  Recorded only (not modelled):")
    for t in RECORDED_ONLY:
        print(f"    {t:<14}{int(df[t].sum()):>5}")
    print(f"\n  Negative pool (all 10 targets = 0): {neg}")
    print(f"  -> these sentences act as shared negatives for every classifier.")


# ---- main ------------------------------------------------------------------

def main():
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        sys.exit(f"ERROR: cannot find {INPUT_FILE} in the current folder.")

    missing = [c for c in (ID_COL, TEXT_COL, TAG_COL) if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected column(s): {missing}. "
                 f"Found: {list(df.columns)}")

    errors = validate(df)
    if errors:
        print(f"\nVALIDATION FAILED - {len(errors)} issue(s):\n")
        for e in errors:
            print("  -", e)
        print("\nFix these in the template and re-run. No output written.")
        sys.exit(1)

    df = build_binary(df)
    report(df)

    out_cols = [ID_COL, TEXT_COL] + TARGETS + RECORDED_ONLY
    df[out_cols].to_excel(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}  ({len(df)} rows, {len(out_cols)} columns).")


if __name__ == "__main__":
    main()

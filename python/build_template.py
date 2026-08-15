"""
Builds an excel template for hand annotation/labelling of
sentence units. Splits documents into constituent sentences
based on presence of select punctuation marks followed by
a space.
"""

import pandas as pd
import re

SRC = [DIRECTORY PATH]
OUT = [DIRECTORY PATH]

df = pd.read_excel(SRC).dropna(subset=["Text"])

def split_sentences(text):
	parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
	return [p.strip() for p in parts if p.strip()]

rows = []
for _, r in df.iterrows():
	for sentence in split_sentences(r["Text"]):
		rows.append({"Text_ID": r["Text_ID"], "Sentence": sentence, "Classifications": ""})

out = pd.DataFrame(rows, columns=["Text_ID", "Sentence", "Classifications"])
out.to_excel(OUT, index=False)
print(f"Wrote {len(out)} sentences from {df['Text_ID'].nunique()} forms to:\n{OUT}")
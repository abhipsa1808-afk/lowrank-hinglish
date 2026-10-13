"""Clean the English-Hinglish dataset and save small train/val/test splits."""
import json, random, re
from datasets import load_dataset

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
N_TRAIN, N_VAL, N_TEST = 4000, 200, 100


def clean(text):
    return " ".join(str(text).split()).strip()


def is_good(en, hi):
    if not en or not hi:
        return False
    if DEVANAGARI.search(hi):                 # must be Roman script
        return False
    if en.lower() == hi.lower():              # no translation happened
        return False
    if any(t in en + hi for t in ("http", "www.", "@", "#")):
        return False
    n_en, n_hi = len(en.split()), len(hi.split())
    if not (3 <= n_en <= 25 and 3 <= n_hi <= 30):
        return False
    if n_hi > 2 * n_en + 3 or n_en > 2 * n_hi + 3:   # badly mismatched pair
        return False
    return True


def filter_split(rows):
    seen, out = set(), []
    for r in rows:
        en, hi = clean(r["en"]), clean(r["hi_en"])
        if is_good(en, hi) and en.lower() not in seen:
            seen.add(en.lower())
            out.append({"en": en, "hinglish": hi})
    return out


def save(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


ds = load_dataset("rvv-karma/English-Hinglish")
train_pool = filter_split(ds["train"])
test_pool = filter_split(ds["test"])
print(f"Clean pairs: train pool {len(train_pool):,} / {len(ds['train']):,}, "
      f"test pool {len(test_pool):,} / {len(ds['test']):,}")

random.seed(42)
random.shuffle(train_pool)
random.shuffle(test_pool)
train, val = train_pool[:N_TRAIN], train_pool[N_TRAIN:N_TRAIN + N_VAL]
test = test_pool[:N_TEST]

save(train, "data/train.jsonl")
save(val, "data/val.jsonl")
save(test, "data/test.jsonl")
print(f"Saved: {len(train)} train, {len(val)} val, {len(test)} test -> data/\n")

for r in train[:5]:
    print("EN:", r["en"])
    print("HI:", r["hinglish"], "\n")

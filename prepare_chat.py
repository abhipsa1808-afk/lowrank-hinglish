import json, re
from datasets import load_dataset

DEV = re.compile(r"[\u0900-\u097F]")

def ok(en, hi):
    if not en or not hi or DEV.search(hi) or en.lower() == hi.lower(): return False
    a, b = len(en.split()), len(hi.split())
    return 3 <= a <= 30 and 3 <= b <= 35 and b <= 2 * a + 3 and a <= 2 * b + 3

ds = load_dataset("festvox/cmu_hinglish_dog")
out = {}
for split, name, cap in [("train", "train_chat", 4000), ("validation", "val_chat", 200), ("test", "test_chat", 100)]:
    rows = []
    for r in ds[split]:
        t = r["translation"]
        en, hi = " ".join(str(t["en"]).split()), " ".join(str(t["hi_en"]).split())
        if ok(en, hi): rows.append({"en": en, "hinglish": hi})
    out[name] = rows[:cap]
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in out[name])
    print(f"{name}: {len(out[name])} pairs (from {len(rows)} clean)")
print()
for x in out["train_chat"][100:106]:
    print("EN:", x["en"]); print("HI:", x["hinglish"], "\n")

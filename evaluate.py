import gc, json, os, re
from collections import Counter
import torch
from hinglish import load, translate

DEV = re.compile(r"[\u0900-\u097F]")
DEMO = ["I'll be late, start without me.", "Did you eat lunch?",
        "Don't worry, everything will be fine.", "Can you send me the notes from today's class?",
        "Have you seen the new movie? It's really good.", "Set an alarm for 7 am tomorrow"]
TEST = [json.loads(l) for l in open("data/test_chat.jsonl", encoding="utf-8")][:30]
MODELS = {"base": None, "v1": "adapter/hinglish_lora.pt",
          "v2": "adapter/hinglish_lora_v2.pt", "v3": "adapter/hinglish_lora_v3.pt"}


def f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    c = sum((Counter(p) & Counter(r)).values())
    return 0.0 if not c else 2 * c / (len(p) + len(r))


res = {}
for name, path in MODELS.items():
    model, tok = load(path)
    preds = [translate(model, tok, r["en"], max_new_tokens=50) for r in TEST]
    res[name] = {
        "roman_pct": round(100 * sum(not DEV.search(p) for p in preds) / len(preds), 1),
        "word_f1": round(100 * sum(f1(p, r["hinglish"]) for p, r in zip(preds, TEST)) / len(TEST), 1),
        "demo": {s: translate(model, tok, s, max_new_tokens=50) for s in DEMO},
    }
    print(f"{name:5s} | roman script {res[name]['roman_pct']:5.1f}% | word-F1 {res[name]['word_f1']:5.1f}")
    del model; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/eval.json", "w"), indent=2, ensure_ascii=False)
print("Saved results/eval.json")

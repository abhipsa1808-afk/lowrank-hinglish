import json, math, os, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora, count_params

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = ("Translate this English sentence into Hinglish "
               "(Hindi written in English letters, mixed with English words, "
               "the way Indians text each other).")
R, ALPHA, LR, BATCH, EPOCHS, EVAL_EVERY = 16, 32, 2e-4, 4, 1, 200
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)
random.seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
n = inject_lora(model, TARGETS, r=R, alpha=ALPHA)
trainable, total = count_params(model)
print(f"LoRA in {n} layers | trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def build_prompt(en):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\nEnglish: {en}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class PairData(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = tokenizer(build_prompt(r["en"]), add_special_tokens=False)["input_ids"]
        a = tokenizer(r["hinglish"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        return p + a, [-100] * len(p) + a


def collate(batch):
    L = max(len(x) for x, _ in batch)
    pad = tokenizer.pad_token_id
    ids = [x + [pad] * (L - len(x)) for x, _ in batch]
    labels = [y + [-100] * (L - len(y)) for _, y in batch]
    mask = [[1] * len(x) + [0] * (L - len(x)) for x, _ in batch]
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(mask)


train_dl = DataLoader(PairData("data/train_chat.jsonl"), batch_size=BATCH, shuffle=True, collate_fn=collate)
val_dl = DataLoader(PairData("data/val_chat.jsonl"), batch_size=BATCH, collate_fn=collate)


@torch.no_grad()
def val_loss():
    model.eval()
    tot = 0.0
    for ids, labels, mask in val_dl:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
        tot += out.loss.item()
    model.train()
    return tot / len(val_dl)


params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps = min(EPOCHS * len(train_dl), 600)
warm = lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, warm)   # warmup + cosine decay
log = {"step": [], "train": [], "val_step": [], "val": []}

print(f"Start val loss: {val_loss():.3f} | training {steps} steps")
model.train()
step, t0 = 0, time.time()
for epoch in range(EPOCHS):
    for ids, labels, mask in train_dl:
        loss = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                     labels=labels.to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % 10 == 0 and device == "mps":
            torch.mps.empty_cache()
        log["step"].append(step)
        log["train"].append(loss.item())
        if step % 20 == 0:
            print(f"step {step}/{steps} | loss {loss.item():.3f} | {time.time()-t0:.0f}s")
        if step % EVAL_EVERY == 0 or step == steps:
            v = val_loss()
            log["val_step"].append(step)
            log["val"].append(v)
            print(f"  >> val loss {v:.3f}")
            os.makedirs("adapter", exist_ok=True)
            ck = {k: t.detach().cpu() for k, t in model.state_dict().items() if "lora_" in k}
            torch.save({"state": ck, "r": R, "alpha": ALPHA, "targets": list(TARGETS),
                        "model": MODEL_NAME}, "adapter/hinglish_lora_v3.pt")
            print("  >> checkpoint saved")
            if step >= steps:
                break


os.makedirs("adapter", exist_ok=True)
lora_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
torch.save({"state": lora_state, "r": R, "alpha": ALPHA,
            "targets": list(TARGETS), "model": MODEL_NAME}, "adapter/hinglish_lora_v3.pt")
json.dump(log, open("adapter/train_log_v3.json", "w"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4.5))
plt.plot(log["step"], log["train"], alpha=0.35, label="train loss")
plt.plot(log["val_step"], log["val"], "o-", label="val loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
plt.title("LoRA fine-tuning: English -> Hinglish")
plt.tight_layout()
plt.savefig("adapter/loss_curve_v3.png", dpi=150)
mb = os.path.getsize("adapter/hinglish_lora_v3.pt") / 1e6
print(f"Saved adapter/hinglish_lora_v3.pt ({mb:.1f} MB) + loss_curve_v3.png | {time.time()-t0:.0f}s total")

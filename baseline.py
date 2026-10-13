import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
model.eval()

total = sum(p.numel() for p in model.parameters())
print(f"Model has {total/1e6:.0f}M parameters\n")

INSTRUCTION = ("Translate this English sentence into Hinglish "
               "(Hindi written in English letters, mixed with English words, "
               "the way Indians text each other).")

sentences = [
    "I'll be late, start without me.",
    "Did you eat lunch?",
    "This movie was so boring, I almost fell asleep.",
    "Can you send me the notes from today's class?",
    "Don't worry, everything will be fine.",
]

for s in sentences:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\nEnglish: {s}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"EN: {s}")
    print(f"BASE MODEL: {reply.strip()}\n")

"""Load Qwen (optionally with the trained LoRA adapter) and translate EN -> Hinglish."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora, LoRALinear

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = ("Translate this English sentence into Hinglish "
               "(Hindi written in English letters, mixed with English words, "
               "the way Indians text each other).")


def load(adapter_path=None, merge=True):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    if adapter_path:
        ck = torch.load(adapter_path, map_location="cpu")
        inject_lora(model, tuple(ck["targets"]), r=ck["r"], alpha=ck["alpha"], dropout=0.0)
        model.load_state_dict(ck["state"], strict=False)
        if merge:
            for m in model.modules():
                if isinstance(m, LoRALinear):
                    m.merge()
    return model.to(device).eval(), tok


@torch.no_grad()
def translate(model, tok, text, max_new_tokens=60):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\nEnglish: {text}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

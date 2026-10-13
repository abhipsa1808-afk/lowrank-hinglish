import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora, count_params, LoRALinear

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
model.eval()

inputs = tokenizer("Kal milte hain, bye!", return_tensors="pt").to(device)
with torch.no_grad():
    before = model(**inputs).logits

n = inject_lora(model, target_names=("q_proj", "v_proj"), r=8, alpha=16)
model.eval()
with torch.no_grad():
    after = model(**inputs).logits

trainable, total = count_params(model)
print(f"Injected LoRA into {n} layers")
print(f"Trainable params: {trainable:,} of {total:,} ({100*trainable/total:.3f}%)")
print(f"Output unchanged after injection: {torch.allclose(before, after, atol=1e-4)}")

first = next(m for m in model.modules() if isinstance(m, LoRALinear))
print(f"Example layer: A {tuple(first.lora_A.shape)}, B {tuple(first.lora_B.shape)}")

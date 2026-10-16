import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora, LoRALinear
from hinglish import MODEL_NAME, translate

ADAPTER = "adapter/hinglish_lora_v3.pt" if os.path.exists("adapter/hinglish_lora_v3.pt") \
    else "adapter/hinglish_lora.pt"


@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16   # bf16 halves RAM on CPU
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    ck = torch.load(ADAPTER, map_location="cpu")
    inject_lora(model, tuple(ck["targets"]), r=ck["r"], alpha=ck["alpha"], dropout=0.0)
    model.load_state_dict(ck["state"], strict=False)
    return model.to(device).eval(), tok, ck


def set_lora(model, on):
    # LoRALinear skips the B@A path when merged=True; we use that flag as an on/off switch
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merged = not on


st.set_page_config(page_title="lowrank-hinglish", page_icon="🇮🇳", layout="centered")
st.title("🇮🇳 lowrank-hinglish")
st.caption("English → Hinglish with LoRA written from scratch (no PEFT) on Qwen2.5-0.5B · by Abhipsa Rout")

with st.spinner("Loading model (first time takes ~1 min)..."):
    model, tok, ck = load_model()

n_lora = sum(t.numel() for t in ck["state"].values())
mb = os.path.getsize(ADAPTER) / 1e6
c1, c2, c3 = st.columns(3)
c1.metric("Base model", "494M params")
c2.metric("Trained by LoRA", f"{n_lora/1e6:.2f}M params")
c3.metric("Adapter file", f"{mb:.1f} MB")

examples = ["Play some old Bollywood songs", "Remind me to call mom in the evening",
            "Set an alarm for 7 am tomorrow", "Did you eat lunch?"]
pick = st.selectbox("Try an example, or type your own below:", examples)
text = st.text_input("English sentence", value=pick)

if st.button("Translate", type="primary") and text.strip():
    left, right = st.columns(2)
    with st.spinner("Generating..."):
        set_lora(model, False)
        base_out = translate(model, tok, text, max_new_tokens=40)
        set_lora(model, True)
        lora_out = translate(model, tok, text, max_new_tokens=40)
    left.subheader("❌ Base model")
    left.write(base_out)
    right.subheader("✅ With my LoRA")
    right.write(lora_out)

st.divider()
st.markdown("**How it works:** each attention and MLP layer gets two tiny matrices A and B. "
            "The frozen weight W becomes W + (α/r)·B·A, and only A and B are trained. "
            "The comparison above just switches that B·A path off and on.")

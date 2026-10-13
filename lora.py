"""LoRA (Low-Rank Adaptation) written from scratch - no PEFT library.

Paper: Hu et al., 2021 - "LoRA: Low-Rank Adaptation of Large Language Models"
Idea:  freeze the big weight W and learn a low-rank update delta_W = B @ A,
       A is (r x in), B is (out x r), r << in, out.
       y = x W^T + (alpha / r) * x A^T B^T
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.merged = False
        for p in self.base.parameters():
            p.requires_grad = False
        dev, dt = base.weight.device, base.weight.dtype
        # A random, B zeros -> B @ A = 0, so the model starts unchanged
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out

    @torch.no_grad()
    def merge(self):
        """Fold B @ A into W so inference has zero extra cost."""
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


def inject_lora(model, target_names=("q_proj", "v_proj"), r=8, alpha=16, dropout=0.05):
    """Freeze the whole model, then swap target Linear layers for LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    targets = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                targets.append((module, child_name, child))
    for parent, child_name, child in targets:
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return len(targets)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(64, 32)
    layer = LoRALinear(base, r=4, alpha=8)
    layer.eval()
    x = torch.randn(5, 64)
    assert torch.allclose(layer(x), base(x)), "Test 1 failed"
    print("Test 1 passed: at init LoRA output == original output")
    trainable, total = count_params(layer)
    assert trainable == 4 * (64 + 32), "Test 2 failed"
    print(f"Test 2 passed: trainable {trainable} of {total} params")
    with torch.no_grad():
        layer.lora_B.normal_()
    before = layer(x)
    layer.merge()
    assert torch.allclose(before, layer(x), atol=1e-5), "Test 3 failed"
    print("Test 3 passed: merged weights give identical output")
    print("All LoRA tests passed!")

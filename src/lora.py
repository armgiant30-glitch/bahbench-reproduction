"""LoRA modules used by the paper-style Transformer reproduction.

The wrapper follows W + BA while keeping W frozen; B is initialized to zero.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """把 nn.Linear 包一层 LoRA：原权重冻结，只训 A、B 两个小矩阵。

        A ∈ R^{r×k}，B ∈ R^{h×r}，前向 = base(x) + (alpha/r) · B(Ax)
        B 初始化为 0 → 训练开始时 ΔW = 0，不会破坏预训练权重。
    """

    def __init__(self, base: nn.Linear, r=16, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scale = (alpha if alpha is not None else r) / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        return out + self.scale * (x @ self.lora_A.T) @ self.lora_B.T

    @property
    def lora_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


# 不同模型家族里注意力投影层的命名
LORA_TARGET_NAMES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "qv": ("q_proj", "v_proj"),
}


def inject_lora(module, r=16, targets=("q_proj", "k_proj", "v_proj"), prefix=""):
    """递归替换注意力里的 Q/K/V 线性层。返回 (替换数量, 命中的层名列表)。"""
    count, names = 0, []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r=r))
            count += 1
            names.append(full)
        else:
            c, n = inject_lora(child, r=r, targets=targets, prefix=full)
            count += c
            names += n
    return count, names


# ===========================================================================
# 下游模型：基础模型（可选 LoRA） + 平均池化 + 线性分类头
# ===========================================================================

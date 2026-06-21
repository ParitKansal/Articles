---
excerpt: "Low-Rank Adaptation (LoRA) reduces the number of trainable parameters by learning rank-decomposition matrices, making LLM fine-tuning incredibly efficient without sacrificing performance."
read_time: "10 min"
---

# Low-Rank Adaptation (LoRA)

**Source:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

## Introduction

Fine-tuning Large Language Models (LLMs) traditionally requires updating billions of parameters. While effective, this approach is computationally expensive, memory-intensive, and impractical for many researchers and organizations. As model sizes continued to grow, the need for more efficient adaptation techniques became increasingly important.

In 2021, Edward J. Hu and colleagues introduced **Low-Rank Adaptation (LoRA)**, a Parameter-Efficient Fine-Tuning (PEFT) method that dramatically reduces the number of trainable parameters while maintaining performance comparable to full fine-tuning. Instead of updating all model weights, LoRA learns small low-rank matrices that represent task-specific weight updates.

Today, LoRA has become one of the most widely used techniques for adapting LLMs, diffusion models, and multimodal architectures.

## Motivation

Consider a pretrained weight matrix:
$$
W \in \mathbb{R}^{d \times k}
$$

Traditional fine-tuning updates the entire matrix:
$$
W' = W + \Delta W
$$
where $\Delta W \in \mathbb{R}^{d \times k}$ contains the learned modifications.

For modern LLMs, these matrices can contain millions or even billions of parameters. Updating every parameter requires large GPU memory, high computational cost, extensive storage requirements, and separate copies of the model for different tasks.

LoRA addresses these challenges by observing that task-specific updates often occupy a much smaller subspace than the original parameter space.

## Core Idea

Instead of learning the full update matrix $\Delta W$, LoRA approximates it using two low-rank matrices:
$$
\Delta W = BA
$$
where:
* $A \in \mathbb{R}^{r \times k}$
* $B \in \mathbb{R}^{d \times r}$
* and $r \ll \min(d, k)$

The rank $r$ is a user-defined hyperparameter, typically chosen from $r \in \{4, 8, 16, 32, 64\}$.

The adapted weight becomes:
$$
W' = W + BA
$$

Instead of training $d \times k$ parameters, LoRA trains only $r(d+k)$ parameters, resulting in massive efficiency gains.

## Why Low-Rank Updates Work

LoRA is motivated by the concept of **intrinsic dimensionality**. Research has shown that neural networks often learn meaningful solutions within surprisingly small subspaces of the full parameter space. During fine-tuning, the required weight updates are frequently much simpler than the original pretrained weights.

This suggests that $\Delta W$ can often be approximated accurately using a low-rank representation. Rather than searching through billions of independent parameter directions, LoRA learns updates within a highly compact subspace.

## Mathematical Formulation

Consider a standard linear layer:
$$
y = Wx
$$

LoRA modifies it as:
$$
y = Wx + BAx
$$
or equivalently:
$$
y = (W + BA)x
$$

During training, the original weights $W$ remain frozen. Only the low-rank matrices $A$ and $B$ receive gradients and are updated.

### Scaling Factor

The original LoRA paper introduces a scaling parameter $\alpha$, resulting in:
$$
y = Wx + \frac{\alpha}{r}BAx
$$
where $r$ is the LoRA rank and $\alpha$ controls the update magnitude. The scaling factor stabilizes training and ensures consistent behavior across different rank values, so you don't need to retune learning rates drastically when changing $r$.

## Rank Factorization Example

Suppose an LLM contains a weight matrix:
$$
W \in \mathbb{R}^{16000 \times 16000}
$$

The total parameter count is:
$$
16000 \times 16000 = 256,000,000
$$

Using a rank $r = 300$, we obtain:
* $A \in \mathbb{R}^{300 \times 16000}$
* $B \in \mathbb{R}^{16000 \times 300}$

The total trainable parameters become:
$$
(300 \times 16000) + (16000 \times 300) = 9,600,000
$$

This is only about **3.75%** of the original matrix size!

## Applying LoRA Inside Transformers

A Transformer consists of several linear layers, and LoRA can be attached to any of them.

### Attention Projections

* **Query Projection (Q):** $Q = X(W_q + BA)$. Query vectors determine what information each token searches for. LoRA on Q modifies attention behavior by changing how tokens formulate queries.
* **Key Projection (K):** $K = X(W_k + BA)$. Keys determine how information presents itself to other tokens. LoRA on K changes how tokens expose their semantic content.
* **Value Projection (V):** $V = X(W_v + BA)$. Values contain the actual information passed through attention. LoRA on V changes the content exchanged between tokens.
* **Output Projection (O):** $Y = \text{Attention}(X)(W_o + BA)$. This modifies how attention outputs are integrated into subsequent layers.

### Feed Forward Networks (FFN)

Modern Transformers contain large feed-forward networks between attention layers. For models such as LLaMA, these include `gate_proj`, `up_proj`, and `down_proj`. These layers often contain more parameters than attention projections.

* **Up / Gate Projection:** LoRA modifies how hidden representations expand into higher-dimensional spaces and influences feature selection/activation gating.
* **Down Projection:** LoRA modifies how expanded features are compressed back into the model's hidden dimension.

### Embeddings and LM Head

* **Token Embeddings:** $E = \text{TokenEmbedding}(x) + BA$. This changes the semantic representations of tokens themselves. While possible, embedding LoRA is less common because embedding matrices are already extremely large.
* **Language Modeling Head:** $\text{logits} = h(W_{lm} + BA)$. This directly affects token prediction behavior and can be useful for specialized domains.

## Typical LoRA Configurations

* **Original LoRA Paper:** The original work primarily targeted the Query ($Q$) and Value ($V$) projections. This configuration provided excellent efficiency and performance.
* **Common Modern Configuration:** Many LLM fine-tuning pipelines target all attention matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) to adapt the entire attention mechanism.
* **Full Attention + FFN Configuration:** High-quality instruction tuning often applies LoRA to all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This enables both attention adaptation and deep knowledge adaptation at the cost of more VRAM.

## Weight Merging

A major advantage of LoRA is that adapters can be merged directly into the base model. After training, the matrices can be pre-calculated:
$$
W_{merged} = W + BA
$$

The resulting model behaves identically to a standard model. Unlike traditional bottleneck adapters (which add extra layers), merged LoRA models introduce **zero inference overhead**. This eliminates latency penalties during deployment.

## LoRA vs Full Fine-Tuning vs Traditional Adapters

| Feature | Full Fine-Tuning | Traditional Adapters | LoRA |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | 100% | ~2-5% | <1-2% typically |
| **Adds New Layers** | No | Yes | No (Parallel) |
| **Frozen Backbone** | No | Yes | Yes |
| **Inference Latency** | None | Increased | None after merge |
| **Storage Cost** | Very High | Low | Very Low |
| **Deployment** | Difficult | Moderate | Excellent |

## QLoRA

One of the most important developments built upon LoRA is **QLoRA**. 

QLoRA quantizes the base model to 4-bit precision, keeps the quantized model frozen, and trains 16-bit/32-bit LoRA adapters on top. This combination dramatically reduces memory consumption, enabling fine-tuning of very large models on consumer-grade GPUs while maintaining strong performance.

---

### Implementation: Tiny LoRA GPT in PyTorch

Below is an educational PyTorch implementation demonstrating how LoRA is implemented conceptually, including `LoRALinear`, `LoRAEmbedding`, and how they fit into a SwiGLU-based Transformer block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Generic LoRA Layer
# ============================================================
class LoRALinear(nn.Module):
    """
    W' = W + BA
    Original weight W is frozen. Only A and B are trainable.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # The scaling factor ensures the magnitude of the update remains 
        # consistent even if we change the rank later.
        self.scaling = alpha / rank

        # Frozen base weight W
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features),
            requires_grad=bias
        ) if bias else None

        # LoRA matrix A (down projection). Initialized with normal distribution.
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        # LoRA matrix B (up projection). Initialized to zero so that at the 
        # start of training, BA = 0 and the model acts like the original model.
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # 1. Original frozen path: Wx + b
        base = F.linear(x, self.weight, self.bias)

        # 2. LoRA path: (xA^T)B^T. We apply the scaling factor here.
        lora = (x @ self.A.T) @ self.B.T

        # 3. Combine: Wx + b + (alpha/r)BAx
        return base + self.scaling * lora


# ============================================================
# LoRA Embedding
# ============================================================
class LoRAEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, rank=8, alpha=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Freeze base embeddings
        self.embedding.weight.requires_grad = False

        # LoRA parameters for embeddings
        self.A = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, hidden_size))
        self.scaling = alpha / rank

    def forward(self, input_ids):
        # Base token embeddings
        base = self.embedding(input_ids)

        # Compute full delta matrix: BA
        delta_embedding = self.A @ self.B

        # Look up delta embeddings for the input tokens
        lora = F.embedding(input_ids, delta_embedding)

        return base + self.scaling * lora


# ============================================================
# Multi Head Attention with LoRA Everywhere
# ============================================================
class LoRAMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rank=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # In modern full-parameter tuning, Q, K, V, and O all use LoRA
        self.q_proj = LoRALinear(hidden_size, hidden_size, rank=rank)
        self.k_proj = LoRALinear(hidden_size, hidden_size, rank=rank)
        self.v_proj = LoRALinear(hidden_size, hidden_size, rank=rank)
        self.o_proj = LoRALinear(hidden_size, hidden_size, rank=rank)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ V).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


# ============================================================
# SwiGLU MLP with LoRA (LLaMA Style)
# ============================================================
class LoRAMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank=8):
        super().__init__()
        self.gate_proj = LoRALinear(hidden_size, intermediate_size, rank=rank)
        self.up_proj = LoRALinear(hidden_size, intermediate_size, rank=rank)
        self.down_proj = LoRALinear(intermediate_size, hidden_size, rank=rank)

    def forward(self, x):
        # SwiGLU: SiLU(gate(x)) * up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


# ============================================================
# Transformer Block & Full Model Assembly
# ============================================================
class LoRATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, rank=8):
        super().__init__()
        self.attn = LoRAMultiHeadAttention(hidden_size, num_heads, rank)
        self.mlp = LoRAMLP(hidden_size, intermediate_size, rank)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyLoRAGPT(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=512, num_heads=8, intermediate_size=2048, num_layers=4, rank=8):
        super().__init__()
        self.embed = LoRAEmbedding(vocab_size, hidden_size, rank)
        self.blocks = nn.ModuleList([
            LoRATransformerBlock(hidden_size, num_heads, intermediate_size, rank)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = LoRALinear(hidden_size, vocab_size, rank)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

# Example Execution
model = TinyLoRAGPT()
tokens = torch.randint(0, 32000, (2, 128))
logits = model(tokens)

print("LoRA Output Shape:", logits.shape)
```

---

### Appendix: Modern Transformer Architecture Background

The base architecture used in the `TinyLoRAGPT` example above follows the standard patterns established by models like **LLaMA**, utilizing pre-normalization, SwiGLU MLP layers, and dual residual connections.

#### Residual (Skip) Connections
Modern LLMs rely heavily on residual connections outside the MLP and Attention modules to prevent vanishing gradients in 30–100+ layer architectures. A typical block features two separate residual paths:

1. **Around the Attention module:**
   $$x_1 = x + \text{Attention}(\text{RMSNorm}(x))$$

2. **Around the MLP module:**
   $$x_2 = x_1 + \text{MLP}(\text{RMSNorm}(x_1))$$

![Transformer Residual Block](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/transformer_residual_diagram.png)

#### The SwiGLU MLP
The phrase "Modern LLMs use a gated MLP" describes the internal structure of the feed-forward network. Instead of a single projection followed by a ReLU/GELU, the SwiGLU architecture projects the input into two paths: a gate path (passed through the SiLU activation) and an up path.

Mathematically, for an input $h_1$:
$$g = h_1 W_g$$
$$u = h_1 W_u$$
$$m = \text{SiLU}(g) \odot u$$
$$f = m W_d$$

Consolidated, this becomes:
$$
\text{MLP}(h_1) = \bigl(\text{SiLU}(h_1 W_g) \odot (h_1 W_u)\bigr)W_d
$$

This dual-projection SwiGLU mechanism is why LoRA must be applied to three separate linear layers (`gate_proj`, `up_proj`, `down_proj`) within the MLP block for full adaptation.

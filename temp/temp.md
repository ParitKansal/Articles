Yes, modern LLMs absolutely still use skip (residual) connections.

The phrase "Modern LLMs use a gated MLP (SwiGLU)" only describes the internal structure of the feed-forward network. The residual connections are outside the MLP.

A typical Transformer block in LLaMA-style models looks like:

[
x \rightarrow \text{RMSNorm}
\rightarrow \text{Attention}
\rightarrow +
]

More explicitly:

### Attention sublayer

Input:

[
x
]

Normalize:

[
h = \text{RMSNorm}(x)
]

Attention output:

[
a = \text{Attention}(h)
]

Residual addition:

[
x_1 = x + a
]

---

### MLP (SwiGLU) sublayer

Normalize again:

[
h_1 = \text{RMSNorm}(x_1)
]

Compute SwiGLU:

[
g = h_1W_g
]

[
u = h_1W_u
]

[
m = \text{SiLU}(g)\odot u
]

[
f = mW_d
]

Residual addition:

[
x_2 = x_1 + f
]

Final output of block:

[
x_2
]

---

Diagrammatically:

```text
                ┌─────────────┐
x ──RMSNorm────► Attention    │
│               └─────────────┘
│                       │
└───────────────(+ )◄───┘
        x1

                ┌─────────────┐
x1──RMSNorm────► SwiGLU MLP   │
│               └─────────────┘
│                       │
└───────────────(+ )◄───┘
        x2
```

So each Transformer block has **two residual connections**:

1. Around the Attention module
   [
   x + \text{Attention}(x)
   ]

2. Around the MLP module
   [
   x_1 + \text{MLP}(x_1)
   ]

Without these residual paths, training 30–100+ layer Transformers would be extremely difficult because gradients would vanish or explode.

For a LLaMA block, the full mathematical expression is often written as:

[
y = x + \text{Attention}(\text{RMSNorm}(x))
]

[
z = y + \text{MLP}(\text{RMSNorm}(y))
]

where the MLP is the SwiGLU:

[
\text{MLP}(h)
=============

\bigl(\text{SiLU}(hW_g)\odot(hW_u)\bigr)W_d
]

This is essentially the complete math of one modern decoder-only Transformer block.

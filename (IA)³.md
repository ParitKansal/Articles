---
excerpt: "(IA)³ scales down fine-tuning by learning extreme low-parameter vectors that selectively inhibit and amplify internal activations within a frozen language model."
read_time: "5 min"
---

# (IA)³: Infused Adapter by Inhibiting and Amplifying Inner Activations

## Introduction

As Large Language Models (LLMs) continue to grow in size, fine-tuning them for downstream tasks becomes increasingly expensive. Traditional fine-tuning updates every parameter in the model, requiring substantial computational resources, memory, and storage.

To address this challenge, researchers introduced **Parameter-Efficient Fine-Tuning (PEFT)** methods such as Adapters, Prompt Tuning, Prefix Tuning, and LoRA. Among these approaches, **(IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)** stands out due to its extreme parameter efficiency.

Unlike methods that add new layers or modify weight matrices, (IA)³ adapts a model by learning only a few scaling vectors that selectively amplify or suppress internal activations. This allows the model to specialize for a task while keeping nearly all original parameters frozen.

The method was introduced in the paper: *"Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"*.

## The Core Idea

The fundamental insight behind (IA)³ is remarkably simple: **Instead of changing the model's weights, learn which existing features should be amplified and which should be suppressed.**

Large language models already contain vast amounts of knowledge. For many downstream tasks, it is unnecessary to rewrite this knowledge. Instead, it is often sufficient to adjust how strongly certain features contribute during inference. (IA)³ accomplishes this by introducing trainable vectors that act like volume controls inside the Transformer.

Imagine a sound mixer:
```text
Vocals   🔊
Drums    🔉
Guitar   🔊🔊
Bass     🔈
```
The song remains unchanged, but adjusting the volume levels changes the final output. Similarly, (IA)³ adjusts the importance of internal neural features without modifying the model's original weights.

## Understanding the Transformer Components

Before understanding (IA)³, let's briefly review the Transformer architecture. A standard Transformer block contains a **Multi-Head Attention** module followed by a **Feed Forward Network (FFN)**.

The attention mechanism computes:
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$
where $Q$ is the Query, $K$ is the Key, and $V$ is the Value.

The Feed Forward Network (FFN) is typically defined as:
$$
\text{FFN}(x) = W_2(\sigma(W_1x))
$$
where $W_1$ expands the hidden dimension, $\sigma$ is an activation function (typically GELU), and $W_2$ projects back to the original dimension.

## Where Does (IA)³ Modify the Model?

(IA)³ introduces three trainable vectors inside every Transformer block: $l_k$, $l_v$, and $l_f$.

These vectors are inserted into:
1. Key activations
2. Value activations
3. Feed Forward activations

The original model weights remain completely frozen.

![(IA)³ Architecture](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/ia3_architecture.png)

### Scaling the Keys
In attention, the keys are modified using element-wise multiplication ($\odot$):
$$
K' = l_k \odot K
$$
Example: If $K = [4, 5, 6]$ and $l_k = [1.0, 0.2, 2.0]$, the result is $K' = [4, 1, 12]$. The second dimension becomes less important while the third dimension becomes more important. Since Keys participate directly in attention score computation, scaling them changes where the model attends.

### Scaling the Values
Similarly, the values are scaled:
$$
V' = l_v \odot V
$$
Example: If $V = [10, 20, 30]$ and $l_v = [1.5, 0.5, 2.0]$, the result is $V' = [15, 10, 60]$. Values contain the information retrieved by attention. Scaling them determines which information should have a stronger influence on the output.

### Why Are Queries Not Modified?
A common question is: *Why does (IA)³ modify Keys and Values but not Queries?*

Attention scores are computed as $QK^T$. Scaling Keys already changes attention scores significantly. Researchers found that modifying Keys, Values, and FFN activations provided most of the adaptation benefits while keeping the parameter count extremely small. Adding Query scaling would introduce additional parameters with little performance improvement.

A useful interpretation is:
- **Query:** What am I looking for?
- **Key:** What information is available?
- **Value:** What information should be returned?

(IA)³ focuses on modifying the available information and retrieved information rather than changing the search request itself.

### Scaling the Feed Forward Network
The FFN is another critical component of Transformers. (IA)³ modifies the hidden activation inside the network:
$$
h = \sigma(W_1x)
$$
$$
h' = l_f \odot h
$$
$$
y = W_2h'
$$

Research has shown that a significant portion of a Transformer's knowledge resides in its FFN layers. While Attention determines *where* to look, the FFN determines *how* to process the information. 

By scaling the intermediate features $h$, some features become more influential while others are suppressed. This allows the model to perform task-specific feature selection without modifying the original network.

### Why Multiplication Instead of Addition?
One might wonder why (IA)³ uses multiplication ($h' = l_f \odot h$) instead of addition ($h' = h + l_f$).

Multiplication behaves like a gating mechanism. If $h = [10, 5, 2]$ and $l_f = [0, 1, 2]$, multiplication yields $[0, 5, 4]$ (the first feature is completely disabled). Addition would yield $[10, 6, 4]$, leaving the feature active. Multiplication allows for amplification, suppression, and complete inhibition, making it significantly more expressive.

## Parameter Efficiency

Consider a model with a hidden size of `4096` and an FFN size of `11008`.
Per layer, the trainable parameters are:
- $l_k = 4096$
- $l_v = 4096$
- $l_f = 11008$
Total: $\approx 19,200$ trainable parameters per layer.

For a 32-layer model, this results in $\approx 614,000$ trainable parameters. Compared to billions of frozen parameters, this is extremely small. The typical trainable percentage is only **0.01% – 0.02%**.

## Comparison with LoRA

LoRA modifies weights using low-rank matrices ($W' = W + BA$), where $A$ and $B$ are trainable, and the original weights remain frozen.

(IA)³, on the other hand, does not modify weights at all. Instead, it scales activations directly:
$$
K' = l_k \odot K
$$
$$
V' = l_v \odot V
$$
$$
h' = l_f \odot h
$$

| Feature | LoRA | (IA)³ |
| :--- | :--- | :--- |
| **Learns** | Low-rank matrices | Scaling vectors |
| **Parameter Count** | Low | Extremely Low |
| **Memory Usage** | Low | Very Low |
| **Complexity** | Moderate | Simple |
| **Inference Overhead** | Small | Nearly Zero |

## Advantages of (IA)³

- **Extremely Parameter Efficient:** Only a tiny fraction of parameters are trained.
- **Fast Training:** Fewer trainable parameters result in faster optimization.
- **Low Memory Usage:** Suitable for resource-constrained environments.
- **Easy Deployment:** Task-specific checkpoints are very small.
- **Minimal Inference Cost:** Only simple scaling operations are added.
- **Strong Few-Shot Performance:** The original paper demonstrated performance competitive with full fine-tuning and other PEFT methods.

## Simple PyTorch Implementation

### Value Scaling
```python
import torch
import torch.nn as nn

class IA3ValueScaling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lv = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, V):
        return V * self.lv
```

### Attention Scaling
```python
class IA3Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lk = nn.Parameter(torch.ones(hidden_dim))
        self.lv = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, K, V):
        K = K * self.lk
        V = V * self.lv
        return K, V
```

### FFN Scaling
```python
class IA3FFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lf = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden):
        hidden = hidden * self.lf
        return hidden
```

## Key Takeaways

(IA)³ is one of the most parameter-efficient fine-tuning methods ever proposed. Instead of modifying weight matrices or adding new layers, it learns three small scaling vectors that selectively amplify and suppress activations inside Transformer blocks.

The method works by:
* Scaling **Keys** to control attention scores.
* Scaling **Values** to control retrieved information.
* Scaling **FFN activations** to control feature processing.

This simple idea allows large language models to adapt to new tasks while updating only 0.01–0.02% of their parameters.

A useful way to remember the major PEFT approaches is:
* **Adapters** → Add new layers
* **LoRA** → Modify weights
* **Prompt Tuning** → Learn virtual tokens
* **Prefix Tuning** → Learn layer-wise prefixes
* **IA³** → Scale activations

In one sentence:
**(IA)³ adapts a Transformer not by changing what it knows, but by changing how strongly it uses the knowledge it already has.**

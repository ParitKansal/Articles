---
excerpt: "Intrinsic SAID demonstrated that large language models can be effectively fine-tuned by optimizing a tiny number of parameters inside a low-dimensional subspace, inspiring modern PEFT methods."
read_time: "8 min"
---

# Intrinsic SAID: The Idea That Inspired Modern PEFT

**Source:** *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (Aghajanyan et al., 2020)

## Introduction

Modern language models often contain millions or even billions of parameters. Traditionally, adapting these models to a new task requires updating every parameter through full fine-tuning. For example, consider a model with $D = 1,000,000,000$ parameters. Full fine-tuning requires learning updates for all one billion parameters, which introduces significant computational and memory costs.

A natural question arises: *Do we really need to update all parameters to learn a new task?*

In 2020, Aghajanyan et al. proposed a surprising answer through a method called **SAID (Subspace Approximation for Intrinsic Dimensionality)**. Their research demonstrated that large neural networks can often be adapted effectively by optimizing only a tiny number of trainable parameters inside a low-dimensional subspace. This discovery laid the theoretical foundation for modern Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, Adapters, IA³, and many others.

## The Core Idea

Imagine a person standing inside a massive stadium. Although the stadium is a large three-dimensional space, the person may only need to walk along a narrow path to reach a destination. Similarly, a neural network may contain billions of parameters, but useful task-specific updates may only require movement inside a much smaller hidden subspace.

Instead of searching across the entire parameter space:
$$
\theta \in \mathbb{R}^{D}
$$
SAID searches within a lower-dimensional space:
$$
z \in \mathbb{R}^{d}
$$
where $d \ll D$.

For example, if $D = 1,000,000,000$ and $d = 200$, only 200 trainable parameters are optimized while still producing updates across the entire model.

## What is Intrinsic Dimensionality?

Intrinsic dimensionality represents the minimum number of degrees of freedom required to achieve near-optimal performance on a task.

Formally:
$$
d^* = \text{minimum dimension needed to reach a target performance}
$$
where $d^* \ll D$.

The remarkable finding of the SAID paper was that larger models often exhibit lower relative intrinsic dimensionality. In other words: **Bigger models may require fewer effective dimensions for adaptation.**

## The SAID Reparameterization

Instead of directly optimizing the full parameter vector $\theta$, SAID introduces a low-dimensional trainable vector $z \in \mathbb{R}^{d}$.

The model parameters are defined as:
$$
\theta = \theta_0 + F(z)
$$
where:
* $\theta_0$ = pretrained model parameters
* $z$ = trainable intrinsic vector
* $F$ = projection from low-dimensional space to full parameter space

Only $z$ is optimized. The pretrained weights remain frozen.

![SAID Architecture Diagram](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/said_architecture.png)

## Why Not Use a Normal Projection Matrix?

Suppose $D = 1,000,000,000$ and $d = 200$. A dense projection matrix would require $1,000,000,000 \times 200$ parameters. This equals $200,000,000,000$ values. Such a matrix is far too large to store.

To solve this problem, SAID uses a specialized projection called the **Fastfood Transform**.

### Fastfood Transformation

The Fastfood Transform efficiently projects low-dimensional vectors into extremely large spaces without storing huge dense matrices. Instead of a dense matrix multiplication, Fastfood uses structured matrices:
$$
F = H G \Pi H B
$$
where:
* $H$ = Hadamard matrix
* $G$ = Gaussian scaling matrix
* $B$ = random sign matrix
* $\Pi$ = random permutation matrix

This approach provides memory complexity of $\mathcal{O}(D)$ and time complexity of $\mathcal{O}(D \log d)$, instead of $\mathcal{O}(Dd)$ for standard dense projections.

## Understanding the Training Process

The workflow can be summarized in three steps:

1. **Start with a Pretrained Model:** All pretrained weights $\theta_0$ remain frozen.
2. **Create a Small Trainable Vector:** Create $z \in \mathbb{R}^{d}$. For example, $z \in \mathbb{R}^{200}$. Only these parameters are optimized.
3. **Project into the Full Space:** Compute $\Delta\theta = F(z)$, then update the model weights $\theta = \theta_0 + \Delta\theta$. The resulting parameters are used for the task.

## Educational PyTorch Implementation

The real Fastfood transform is mathematically involved. For educational purposes, we can replace it with a random projection matrix.

```python
import torch
import torch.nn as nn

# Full parameter space
D = 1000

# Intrinsic dimension
d = 20

# Frozen pretrained weights
theta_0 = torch.randn(D)

# Trainable intrinsic vector
z = nn.Parameter(torch.zeros(d))

# Random projection
F_proj = torch.randn(D, d)

# Project into full space
delta_theta = F_proj @ z

# Reparameterized weights
theta = theta_0 + delta_theta
```

Although simplified, this captures the core SAID idea.

## Applying SAID to Real Models

A neural network contains many parameter tensors (e.g., `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`).

SAID conceptually flattens all parameters into a single vector $\theta_0$. The projected update $\Delta\theta$ is then split and reshaped back into the original tensor shapes.

```text
Δθ
│
├── ΔW₁
├── Δb₁
├── ΔW₂
├── Δb₂
└── ...
```

Each update is added to the corresponding pretrained tensor.

## Experimental Findings

The paper evaluated SAID on the MRPC benchmark. Key observations included:

- **Larger Models Need Lower Relative Intrinsic Dimension:** As model size increased, the required intrinsic dimensionality decreased relative to total parameter count.
- **Extremely Small Subspaces Work:** In some experiments, only a few hundred trainable parameters achieved approximately 90% of the performance of full fine-tuning.
- **Fine-Tuning Is Highly Redundant:** The results suggested that much of the parameter space is unnecessary for adaptation. Effective task-specific updates occupy a surprisingly small subspace.

## Why This Discovery Matters

Before SAID, many researchers assumed that successful fine-tuning required updating all model parameters. SAID challenged this assumption by demonstrating $d \ll D$ while maintaining strong performance.

This changed how researchers thought about model adaptation. The question shifted from *"How can we fine-tune all parameters efficiently?"* to *"Can we identify and optimize only the important directions?"* This question directly influenced future PEFT research.

## Limitations

Despite its importance, SAID has several practical limitations:

1. **Projection Complexity:** The Fastfood transformation is more complex than methods such as LoRA or Adapters.
2. **Intrinsic Dimension Search:** Determining the optimal intrinsic dimension often requires multiple experiments (e.g., $d=50, d=100, d=200$). This increases experimentation cost.
3. **Hardware Efficiency:** Modern GPUs are optimized for dense matrix operations. Structured projections may not fully utilize hardware acceleration.
4. **Less Practical Than Later Methods:** Although theoretically elegant, SAID was eventually surpassed by more practical approaches.

## Relationship to LoRA

SAID introduced a crucial insight: **Useful fine-tuning updates live in a low-dimensional space.**

LoRA builds directly on this observation. Instead of using a fixed random projection ($\theta = \theta_0 + F(z)$), LoRA learns a low-rank update ($\Delta W = BA$) where $r \ll \min(m,n)$.

Unlike SAID, LoRA learns the subspace itself rather than relying on a fixed random projection. This makes LoRA more flexible and practical.

### SAID vs LoRA

| Feature | SAID | LoRA |
| :--- | :--- | :--- |
| **Low-dimensional adaptation** | Yes | Yes |
| **Fixed projection** | Yes | No |
| **Learns projection** | No | Yes |
| **Uses Fastfood** | Yes | No |
| **Practical for LLMs** | Limited | Excellent |
| **Historical importance** | Very High | Very High |

## Key Takeaways

* Intrinsic SAID introduced the concept of intrinsic dimensionality for model adaptation.
* It demonstrated that large models can often be fine-tuned using surprisingly few trainable parameters.
* The method optimizes a small vector $z$ and projects it into the full parameter space.
* Fastfood enables efficient projection without storing massive matrices.
* The work provided strong evidence that fine-tuning occurs in a low-dimensional subspace.
* SAID established the theoretical foundations that inspired modern PEFT techniques such as LoRA, Adapters, Prefix Tuning, and IA³.

Although rarely used directly today, Intrinsic SAID remains one of the most influential ideas in the history of parameter-efficient fine-tuning. It transformed our understanding of neural network adaptation and paved the way for the PEFT revolution.

---

### Appendix: Complete Training Example

Below is a complete, simplified PyTorch implementation training an MLP on a toy dataset using a random projection (instead of Fastfood) to demonstrate the SAID concept in practice.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# ==================================================
# Reproducibility
# ==================================================
torch.manual_seed(42)

# ==================================================
# Load Dataset
# ==================================================
# We use a simple digits dataset for educational purposes
digits = load_digits()

X = digits.data
y = digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==================================================
# Model
# ==================================================
# Define a standard Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()

# ==================================================
# Freeze All Parameters (Step 1 of SAID)
# ==================================================
# Just like in PEFT, we freeze all pretrained weights.
for p in model.parameters():
    p.requires_grad = False

# ==================================================
# Save Original Parameters
# ==================================================
original_params = []
param_shapes = []
param_sizes = []

# We conceptually flatten the model into a single parameter space \theta_0
for p in model.parameters():
    original_params.append(p.detach().clone())
    param_shapes.append(p.shape)
    param_sizes.append(p.numel())

D = sum(param_sizes) # D is the full parameter space dimension
print(f"\nTotal Model Parameters: {D:,}")

# ==================================================
# Intrinsic Dimension (Step 2 of SAID)
# ==================================================
d = 172 # d is our low-dimensional subspace (d << D)

# z is the ONLY trainable parameter in the entire network
z = nn.Parameter(torch.zeros(d))

# Random Projection Matrix (Simple proxy for the Fastfood transform)
# F_proj maps z (dimension d) back to the full space (dimension D)
F_proj = torch.randn(D, d) * 0.01

# ==================================================
# Parameter Statistics
# ==================================================
full_model_params = D
trainable_params = z.numel()
compression_ratio = full_model_params / trainable_params
trainable_percentage = (trainable_params / full_model_params) * 100

print("\n========== PARAMETER STATS ==========")
print(f"Full Model Parameters      : {full_model_params:,}")
print(f"Trainable Parameters (z)   : {trainable_params:,}")
print(f"Trainable Percentage       : {trainable_percentage:.4f}%")
print(f"Compression Ratio          : {compression_ratio:.2f}x")
print(f"Projection Matrix Shape    : {tuple(F_proj.shape)}")
print("=====================================\n")

# We only pass `z` to the optimizer! The rest of the model is untouched.
optimizer = optim.Adam([z], lr=1e-2)

# ==================================================
# Build Updated Parameters (Step 3 of SAID)
# ==================================================
def build_updated_parameters():
    # 1. Project low-dimensional z into the full space: \Delta\theta = F(z)
    delta_theta = F_proj @ z
    
    updated_params = []
    offset = 0

    # 2. Add \Delta\theta to \theta_0 and reshape back to original tensor shapes
    for original, size, shape in zip(original_params, param_sizes, param_shapes):
        # Extract the slice of \Delta\theta for this specific layer
        delta = delta_theta[offset:offset + size].view(shape)
        
        # \theta = \theta_0 + \Delta\theta
        updated_params.append(original + delta)
        offset += size

    return updated_params

# ==================================================
# Forward Using Custom Parameters
# ==================================================
def forward_with_params(X, params):
    # Manually extract the updated parameters for each layer
    fc1_w, fc1_b = params[0], params[1]
    fc2_w, fc2_b = params[2], params[3]
    fc3_w, fc3_b = params[4], params[5]

    # Perform the standard forward pass using our updated weights
    h1 = torch.relu(X @ fc1_w.T + fc1_b)
    h2 = torch.relu(h1 @ fc2_w.T + fc2_b)
    logits = h2 @ fc3_w.T + fc3_b

    return logits

# ==================================================
# Training Loop
# ==================================================
losses = []
train_acc_history = []
test_acc_history = []

epochs = 1000

for epoch in range(epochs):
    # Dynamically build the full parameters using the current value of `z`
    params = build_updated_parameters()

    # -------------------------
    # Train Forward
    # -------------------------
    train_logits = forward_with_params(X_train, params)
    loss = F.cross_entropy(train_logits, y_train)
    losses.append(loss.item())

    # Backpropagate through the projection matrix to update ONLY `z`
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -------------------------
    # Train & Test Accuracy
    # -------------------------
    with torch.no_grad():
        train_pred = train_logits.argmax(dim=1)
        train_acc = ((train_pred == y_train).float().mean().item() * 100)

        # Build parameters for evaluation
        params_eval = build_updated_parameters()
        test_logits = forward_with_params(X_test, params_eval)
        test_pred = test_logits.argmax(dim=1)
        test_acc = ((test_pred == y_test).float().mean().item() * 100)

    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")

print("\nTraining Finished")

# ==================================================
# Final Evaluation
# ==================================================
with torch.no_grad():
    params = build_updated_parameters()
    logits = forward_with_params(X_test, params)
    accuracy = ((logits.argmax(dim=1) == y_test).float().mean().item() * 100)

print("\n========== FINAL RESULTS ==========")
print(f"Test Accuracy            : {accuracy:.2f}%")
print(f"Intrinsic Dimension (d)  : {d}")
print(f"Trainable Parameters     : {trainable_params:,}")
print(f"Full Parameters          : {full_model_params:,}")
print("===================================")

# Output generation code for plots is omitted for brevity...
```
```text
Train Shape: torch.Size([1437, 64])
Test Shape : torch.Size([360, 64])

Total Model Parameters: 17,226

========== PARAMETER STATS ==========
Full Model Parameters      : 17,226
Trainable Parameters (z)   : 172
Trainable Percentage       : 0.9985%
Compression Ratio          : 100.15x
Projection Matrix Shape    : (17226, 172)
=====================================

Epoch    0 | Loss 2.3061 | Train Acc 11.27% | Test Acc 9.44%
Epoch  100 | Loss 1.2719 | Train Acc 66.04% | Test Acc 64.44%
Epoch  200 | Loss 0.5229 | Train Acc 84.69% | Test Acc 80.00%
Epoch  300 | Loss 0.3914 | Train Acc 87.61% | Test Acc 83.61%
Epoch  400 | Loss 0.2967 | Train Acc 90.88% | Test Acc 85.28%
Epoch  500 | Loss 0.2430 | Train Acc 92.62% | Test Acc 83.89%
Epoch  600 | Loss 0.2207 | Train Acc 93.39% | Test Acc 84.72%
Epoch  700 | Loss 0.2070 | Train Acc 92.55% | Test Acc 85.00%
Epoch  800 | Loss 0.1946 | Train Acc 92.90% | Test Acc 84.44%
Epoch  900 | Loss 0.1850 | Train Acc 93.60% | Test Acc 82.50%

Training Finished

========== FINAL RESULTS ==========
Test Accuracy            : 82.78%
Intrinsic Dimension (d)  : 172
Trainable Parameters     : 172
Full Parameters          : 17,226
===================================
```

---

### Appendix: SAID with DistilBERT on SST2

Below is a more advanced example applying the SAID concept to the classifier head of a pretrained `distilbert-base-uncased` model fine-tuned on the SST2 sentiment analysis dataset.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# Reproducibility
# =====================================================
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# =====================================================
# Dataset Setup (Stanford Sentiment Treebank)
# =====================================================
dataset = load_dataset("stanfordnlp/sst2")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=268
    )

dataset = dataset.map(
    tokenize,
    batched=True
)

dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

train_loader = torch.utils.data.DataLoader(
    dataset["train"],
    batch_size=512,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset["validation"],
    batch_size=512*2
)

# =====================================================
# Model Definition
# =====================================================
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(device)

# =====================================================
# Freeze Everything (SAID Step 1)
# =====================================================
# All pretrained parameters are frozen to prevent full fine-tuning
for p in model.parameters():
    p.requires_grad = False

# =====================================================
# Save Original Classifier Parameters
# =====================================================
# We will only apply SAID to the final classifier layer for this example.
# We save the original weights (\theta_0) and biases.
base_weight = (
    model.classifier.weight.detach().clone()
)

base_bias = (
    model.classifier.bias.detach().clone()
)

weight_size = base_weight.numel()
bias_size = base_bias.numel()

D = weight_size + bias_size # Full parameter space D for the classifier

print("\nClassifier parameter count:", D)

# =====================================================
# Intrinsic Dimension Setup (SAID Step 2)
# =====================================================
d = 256 # Low-dimensional subspace d

# z is our only trainable parameter
z = nn.Parameter(
    torch.zeros(d, device=device)
)

# Random projection matrix F_proj mapping from d back to D
F_proj = torch.randn(
    D,
    d,
    device=device
) * 0.01

print("Intrinsic Dimension:", d)
print("Compression:", D / d)

# =====================================================
# Build Parameters Function (SAID Step 3)
# =====================================================
def build_classifier():
    # Project low-dimensional z into the full space: \Delta\theta = F(z)
    delta_theta = F_proj @ z

    # Split \Delta\theta into weight and bias updates
    delta_w = delta_theta[:weight_size]
    delta_b = delta_theta[weight_size:]

    # \theta = \theta_0 + \Delta\theta
    weight = (
        base_weight.to(device)
        + delta_w.view_as(base_weight)
    )

    bias = (
        base_bias.to(device)
        + delta_b.view_as(base_bias)
    )

    return weight, bias

# =====================================================
# Custom Forward Pass
# =====================================================
def forward_said(input_ids, attention_mask):
    # Pass inputs through the frozen DistilBERT backbone
    outputs = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    hidden = outputs.last_hidden_state[:, 0]
    hidden = model.pre_classifier(hidden)
    hidden = torch.relu(hidden)
    hidden = model.dropout(hidden)

    # Dynamically build the classifier using our intrinsic vector z
    weight, bias = build_classifier()

    # Apply the custom classifier
    logits = F.linear(
        hidden,
        weight,
        bias
    )

    return logits

# =====================================================
# Evaluation Loop
# =====================================================
@torch.no_grad()
def evaluate():
    total = 0
    correct = 0

    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = forward_said(ids, mask)
        preds = logits.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    return 100 * correct / total

# =====================================================
# Optimizer
# =====================================================
# We only optimize z!
optimizer = optim.Adam(
    [z],
    lr=1e-2
)

# =====================================================
# Training Loop
# =====================================================
epochs = 5

for epoch in range(epochs):
    running_loss = 0

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass using SAID parameters
        logits = forward_said(
            ids,
            mask
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Updates z

        running_loss += loss.item()

        if step % 20 == 0:
            print(
                f"Epoch {epoch+1} "
                f"Step {step} "
                f"Loss {loss.item():.4f}"
            )

    val_acc = evaluate()

    print(
        f"\nEpoch {epoch+1} Completed"
        f"\nAverage Loss: {running_loss/len(train_loader):.4f}"
        f"\nValidation Accuracy: {val_acc:.2f}%\n"
    )

# =====================================================
# Final Evaluation
# =====================================================
final_acc = evaluate()

print("\n========== RESULTS ==========")
print(f"Validation Accuracy: {final_acc:.2f}%")
print("Intrinsic Dimension:", d)
print("Trainable Params:", z.numel())
print("Classifier Params:", D)
print("Compression Ratio:", round(D/d, 2))
print("=============================")
```

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.weight | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
classifier.weight       | MISSING    | 
pre_classifier.weight   | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

Classifier parameter count: 1538
Intrinsic Dimension: 256
Compression: 6.0078125
Epoch 1 Step 0 Loss 0.7029
Epoch 1 Step 20 Loss 0.6674
Epoch 1 Step 40 Loss 0.6511
Epoch 1 Step 60 Loss 0.6290
Epoch 1 Step 80 Loss 0.6096
Epoch 1 Step 100 Loss 0.6000
Epoch 1 Step 120 Loss 0.5563

Epoch 1 Completed
Average Loss: 0.6255
Validation Accuracy: 78.78%

Epoch 2 Step 0 Loss 0.5571
Epoch 2 Step 20 Loss 0.5587
Epoch 2 Step 40 Loss 0.5363
Epoch 2 Step 60 Loss 0.5338
Epoch 2 Step 80 Loss 0.5120
Epoch 2 Step 100 Loss 0.5194
Epoch 2 Step 120 Loss 0.5225

Epoch 2 Completed
Average Loss: 0.5318
Validation Accuracy: 79.59%

Epoch 3 Step 0 Loss 0.5089
Epoch 3 Step 20 Loss 0.4983
Epoch 3 Step 40 Loss 0.4857
Epoch 3 Step 60 Loss 0.4694
Epoch 3 Step 80 Loss 0.4850
Epoch 3 Step 100 Loss 0.4945
Epoch 3 Step 120 Loss 0.4753

Epoch 3 Completed
Average Loss: 0.4830
Validation Accuracy: 80.16%

Epoch 4 Step 0 Loss 0.4692
Epoch 4 Step 20 Loss 0.4909
Epoch 4 Step 40 Loss 0.4378
Epoch 4 Step 60 Loss 0.4453
Epoch 4 Step 80 Loss 0.4485
Epoch 4 Step 100 Loss 0.4700
Epoch 4 Step 120 Loss 0.4446

Epoch 4 Completed
Average Loss: 0.4547
Validation Accuracy: 80.39%

Epoch 5 Step 0 Loss 0.4444
Epoch 5 Step 20 Loss 0.4739
Epoch 5 Step 40 Loss 0.4399
Epoch 5 Step 60 Loss 0.4424
Epoch 5 Step 80 Loss 0.4145
Epoch 5 Step 100 Loss 0.4406
Epoch 5 Step 120 Loss 0.4114

Epoch 5 Completed
Average Loss: 0.4369
Validation Accuracy: 81.08%


========== RESULTS ==========
Validation Accuracy: 81.08%
Intrinsic Dimension: 256
Trainable Params: 256
Classifier Params: 1538
Compression Ratio: 6.01
=============================
```

![SAID DistilBERT Training Metrics](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/said_distilbert_sst2_metrics.png)
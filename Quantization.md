# Quantization

Quantization reduces the memory footprint and computational cost of neural networks by representing weights and activations in lower-precision data types. A float32 weight occupies 4 bytes; the same value stored as int8 occupies 1 byte — a 4× reduction. More importantly, integer arithmetic on modern hardware is significantly faster and more power-efficient than floating-point arithmetic, and many edge devices support only integer operations.

The trade-off is precision loss. The central question quantization methods answer is: how do you choose which 256 integer values best represent the continuous range of a tensor's values, and how do you recover accuracy when the approximation introduces error?

This article builds from the mathematical foundations up through practical deployment, including the specific techniques used for large language models.

---

## Number Representation

Before discussing quantization it helps to be precise about what these data types actually store.

### Integers

An n-bit unsigned integer can represent 2ⁿ distinct values in the range $[0, 2^n − 1]$. An n-bit signed integer in two's complement represents $[−2^{n−1}, 2^{n−1} − 1]$. For int8: [−128, 127], giving 256 distinct values.

The critical property is that integers are *exact*: 42 in int8 is precisely 42, with no rounding error. Quantization introduces error in the *mapping* from float to int, but once you are in integer space, integer arithmetic is exact.

### Floating-point

IEEE 754 floating-point numbers have three components: a sign bit, an exponent field, and a mantissa (significand):

| Type | Sign | Exponent | Mantissa | Bias | Total |
|---|---|---|---|---|---|
| float64 | 1 | 11 | 52 | 1023 | 64 bits |
| float32 | 1 | 8 | 23 | 127 | 32 bits |
| bfloat16 | 1 | 8 | 7 | 127 | 16 bits |
| float16 | 1 | 5 | 10 | 15 | 16 bits |

The value of a normalized float is:

$$
x = (-1)^{\text{sign}} \times \left(1 + \frac{\text{mantissa}}{2^m}\right) \times 2^{\text{exponent} - \text{bias}}
$$

where m is the number of mantissa bits and bias = $2^{e−1} − 1$ for e exponent bits (127 for float32, 15 for float16).

**General Algorithm for Floating-Point Conversion:**

To convert *any* decimal number into its IEEE 754 binary representation, follow these 5 universal steps:
1. **Determine the Sign:** If the number is negative, the sign bit is `1`; if positive, it is `0`.
2. **Convert to Base-2 (Binary):** Convert the absolute value of the decimal number into binary. For example, $5.75 = 4 + 1 + 0.5 + 0.25 = 101.11_2$.
3. **Normalize the Binary Number:** Shift the binary point left or right until there is exactly one `1` to its left. Record the number of shifts as the true exponent. For example, $101.11_2$ becomes $1.0111_2 \times 2^2$. The true exponent is $2$.
4. **Calculate the Stored Exponent:** Add the format's specific **bias** (e.g., 15 for float16, 127 for float32) to the true exponent. Convert this sum into binary to get the stored exponent field.
5. **Extract the Mantissa:** Take the bits *after* the binary point in your normalized number (e.g., `0111`), and pad them with zeros on the right to completely fill the available mantissa bits.

**Step-by-Step Example: Encoding `-27.375` in `float16`**

Let's apply these 5 steps to manually encode the decimal number `-27.375` into a 16-bit `float16` format. This example is more complex as it includes both an integer and a fractional component, as well as a negative sign.

**1. Determine the Sign:**
The number `-27.375` is negative, so the **sign bit is `1`**.

**2. Convert to Base-2 (Binary):**
We convert the absolute value `27.375` into binary by handling the integer and fractional parts separately.
* **Integer part (27):** $27 = 16 + 8 + 2 + 1 = 11011_2$
* **Fractional part (0.375):** $0.375 = 0.25 + 0.125 = 2^{-2} + 2^{-3} = .011_2$

Combining them: $27.375_{10} = 11011.011_2$.

**3. Normalize the Binary Number:**
We shift the binary point to the left until there is exactly one `1` to its left:
* $11011.011_2$ becomes $1.1011011_2 \times 2^4$.

The **true exponent is `4`**.

**4. Calculate the Stored Exponent:**
For `float16`, the exponent bias is 15.
* **Stored Exponent** = True Exponent + Bias = $4 + 15 = 19$.
* The number `19` in 5-bit binary is **`10011`**.

**5. Extract the Mantissa:**
Looking at our normalized form ($1.1011011_2 \times 2^4$), the bits *after* the binary point are `1011011`.
`float16` has 10 mantissa bits. We take `1011011` and pad it with zeros on the right to fill all 10 bits:
* **Stored Mantissa** = **`1011011000`**.

**Putting it together:**
Concatenating the Sign (1 bit), Exponent (5 bits), and Mantissa (10 bits), we get the raw 16-bit binary representation in memory:
`1 10011 1011011000`

If we plug these stored integer values (Exponent = 19, Mantissa bits `1011011000` = $728$ in decimal) back into the decoding formula, we perfectly recover our original number:
$$ 
x = (-1)^1 \times \left(1 + \frac{728}{1024}\right) \times 2^{19 - 15} 
$$
$$ 
x = -1 \times (1 + 0.7109375) \times 2^4 
$$
$$ 
x = -1 \times 1.7109375 \times 16 = -27.375 
$$

### Dynamic Range and Precision Scaling

The **exponent field** is what gives floating-point numbers their massive dynamic range. For instance, `float32` can represent microscopically small numbers (~1.2×10⁻³⁸) all the way up to astronomically large ones (~3.4×10³⁸). 

However, this comes with a crucial caveat: **floating-point values are not uniformly spaced.** Because the exponent scales the mantissa by powers of two, the gap between consecutive representable numbers grows as the numbers get larger. 
* **Near zero**, floating-point numbers are incredibly dense, offering extreme precision.
* **At large magnitudes**, they become sparse, and the distance between adjacent numbers can be huge.

This non-uniformity is the exact reason why quantization is so impactful. When you quantize a weight tensor with a narrow range like `[−0.01, 0.01]` into evenly spaced integers, you are compressing a highly dense cluster of floats. When you quantize a wider tensor like `[−100, 100]`, you are compressing a much sparser distribution. Understanding this spacing helps explain why different layers in a neural network react to quantization differently, and why outliers are so destructive.

### `bfloat16` vs `float16`: The Battle of 16-bit Floats

When working with deep learning, you will frequently encounter two different 16-bit floating-point formats. While they both occupy 16 bits in memory, they allocate those bits very differently:

* **`bfloat16` (Brain Float 16):** Prioritizes **range** over precision. It uses the exact same 8-bit exponent as `float32`, but drastically shrinks the mantissa down to 7 bits. This means `bfloat16` can represent the same astronomically large (and microscopically small) numbers as `float32`, but sacrifices about 2 decimal digits of precision. It is the gold standard for **training** neural networks because backpropagated gradients can wildly fluctuate across many orders of magnitude without overflowing.
* **`float16` (Standard FP16):** Prioritizes **precision** over range. It uses a 10-bit mantissa for better precision, but restricts the exponent to only 5 bits. Because of this small exponent, `float16` has a hard maximum limit of `~65,504` and a minimum representable floor of `~6×10⁻⁵`. This narrow dynamic range is notoriously problematic—for example, it causes the dreaded `NaN` (Not a Number) crashes during Layer Normalization when a tiny epsilon value (like `10⁻¹²`) simply "drops off" the bottom of what `float16` can physically represent (underflow).

---

## Affine Quantization: The Math

The fundamental problem: a weight tensor W contains float32 values in some range [α, β]. You want to represent these as int8 values in [−128, 127]. You need a mapping that preserves relative magnitudes as faithfully as possible within 256 discrete levels.

### Deriving Scale and Zero-point

The affine (asymmetric) quantization scheme defines:

$$
x = S \cdot (x_q - Z)
$$

where:
- $x ∈ ℝ$ is the original float32 value
- $x_q ∈ {q_{min}, ..., q_{max}}$ is the quantized integer value
- $S ∈ ℝ^+$ is the **scale** (units: float per integer step)
- $Z ∈ ℤ$ is the **zero-point** (the integer that maps to exactly 0.0 in float32). Note that the floating-point `0.0` does not necessarily map to the integer `0`; it maps to whatever integer $Z$ evaluates to based on the tensor's range.

It is critical to preserve zero exactly because it is ubiquitous in neural networks: padding, attention masks, ReLU activations, and sparse representations all depend on zero being represented without any rounding error. If `0.0` did not map to an exact integer, we would be forced to round it, and that tiny resulting error would accumulate destructively across millions of operations.

To derive S and Z, require that the endpoints α and β map to the extreme integers q_min and q_max:

$$
\alpha = S \cdot (q_{\min} - Z)
$$
$$
\beta = S \cdot (q_{\max} - Z)
$$
Subtracting the first equation from the second:

$$
\beta - \alpha = S \cdot (q_{\max} - q_{\min}) \implies S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}
$$
For int8: q_min = −128, q_max = 127, so q_max − q_min = 255.

$$
S = \frac{\beta - \alpha}{255}
$$
Substituting back to solve for Z:

$$
Z = \text{round}\!\left(q_{\min} - \frac{\alpha}{S}\right), \quad Z = \text{clip}(Z,\; q_{\min},\; q_{\max})
$$
### Quantize and Dequantize

**Quantization** (float32 → int8):

$$
x_q = \text{clip}\!\left(\text{round}\!\left(\frac{x}{S} + Z\right),\; q_{\min},\; q_{\max}\right)
$$
**Dequantization** (int8 → float32):

$$
\hat{x} = S \cdot (x_q - Z)
$$
The round-trip quantization error for any value x ∈ [α, β] is bounded by:

$$
|x - \hat{x}| = \left|x - S \cdot \left(\text{round}\!\left(\frac{x}{S} + Z\right) - Z\right)\right| \leq \frac{S}{2}
$$
Maximum absolute error per element is S/2 = (β − α) / 510. A wider range means a coarser step and more error. This is the fundamental precision-range trade-off.

### Numerical Example

Suppose a weight tensor has range [α, β] = [−1.2, 0.8].

$$
S = \frac{0.8 - (-1.2)}{255} = \frac{2.0}{255} \approx 0.00784
$$
$$
Z = \text{round}\!\left(-128 - \frac{-1.2}{0.00784}\right) = \text{round}(-128 + 153.06) = \text{round}(25.06) = 25
$$
Quantizing x = 0.5:
$$
x_q = \text{round}\!\left(\frac{0.5}{0.00784} + 25\right) = \text{round}(63.78 + 25) = 89
$$
Dequantizing x_q = 89:
$$
\hat{x} = 0.00784 \times (89 - 25) = 0.00784 \times 64 \approx 0.5018
$$
Error: |0.5 − 0.5018| = 0.0018 ≤ S/2 = 0.00392 ✓

Verify zero is preserved: x = 0.0
$$
x_q = \text{round}\!\left(\frac{0.0}{0.00784} + 25\right) = 25 = Z, \qquad \hat{x} = 0.00784 \times (25 - 25) = 0.0 \checkmark
$$
---

## Symmetric Quantization

Symmetric quantization is a constrained special case of affine quantization: the float32 range is forced to be symmetric around zero, i.e., $[−α_{max}, α_{max}]$ where $α_{max} = max(|α|, |β|)$.

The integer range is also made symmetric: $[−127, 127]$ (excluding −128 to maintain symmetry).

With a symmetric range, $Z = 0$ identically:

$$
S = \frac{\alpha_{\max}}{127}, \qquad x_q = \text{clip}\!\left(\text{round}\!\left(\frac{x}{S}\right),\; -127,\; 127\right), \qquad \hat{x} = S \cdot x_q
$$
**Why exclude −128?** By mapping $-127$ to $-\alpha_{\max}$ and $127$ to $\alpha_{\max}$, we maintain perfect symmetry. Including $-128$ would make the negative representable range one step wider than the positive range, fundamentally breaking the $Z = 0$ property.

**The computational payoff:** Consider a quantized matrix multiplication $Y = A \cdot B$ with inner dimension $k$. If we substitute the dequantization formulas ($A \approx S_A \cdot (A_q - Z_A)$ and $B \approx S_B \cdot (B_q - Z_B)$) into the multiplication and expand using basic algebra, the scalar zero-points ($Z_A$ and $Z_B$) factor out of the dot products. This turns the complex cross-terms into simple sums, creating this exact formula for what the processor must calculate:

$$
Y = S_A \cdot S_B \cdot \bigl(A_q B_q \;-\; Z_B \cdot \text{rowsum}(A_q) \;-\; Z_A \cdot \text{colsum}(B_q) \;+\; Z_A Z_B \cdot k\bigr)
$$
Because the zero-points are single scalar numbers rather than full matrices:
- **$\text{rowsum}(A_q)$**: Emerges from factoring out $Z_B$ from the dot product, leaving just the sum across the row of $A_q$.
- **$\text{colsum}(B_q)$**: Emerges from factoring out $Z_A$, leaving just the sum down the column of $B_q$.
- **$k$**: Adding the scalar product $Z_A Z_B$ to itself $k$ times across the inner dimension is simply $Z_A Z_B \cdot k$.

### Mathematical Proof and Example

**Mathematical Proof:**
For a single element $Y_{ij}$, the dot product over the inner dimension $k$ expands as:
$$
Y_{ij} = S_A S_B \sum_{p=1}^k \bigl( (A_q)_{ip} - Z_A \bigr) \bigl( (B_q)_{pj} - Z_B \bigr)
$$
Expanding the terms inside the sum:
$$
Y_{ij} = S_A S_B \left( \sum_{p=1}^k (A_q)_{ip} (B_q)_{pj} \;-\; Z_B \sum_{p=1}^k (A_q)_{ip} \;-\; Z_A \sum_{p=1}^k (B_q)_{pj} \;+\; Z_A Z_B \sum_{p=1}^k 1 \right)
$$
Here, $\sum_{p=1}^k (A_q)_{ip}$ is exactly the sum of row $i$ of $A_q$, $\sum_{p=1}^k (B_q)_{pj}$ is the sum of column $j$ of $B_q$, and $\sum_{p=1}^k 1$ evaluates exactly to $k$.

**Full Matrix Example:**
Imagine multiplying a $4 \times 3$ matrix by a $3 \times 2$ matrix (inner dimension $k=3$, output is $4 \times 2$). We will ignore the scales ($S_A, S_B$) here to focus on the integer math. Let's say our values are:
$$
A_q = \begin{bmatrix} 2 & 1 & 3 \\ 1 & 2 & 1 \\ 3 & 1 & 2 \\ 2 & 2 & 1 \end{bmatrix}, \quad Z_A = 1 \qquad B_q = \begin{bmatrix} 3 & 2 \\ 1 & 4 \\ 2 & 1 \end{bmatrix}, \quad Z_B = 2
$$
First, the **"true" dequantized dot product** (subtracting zero-points from the matrices first):
$$
A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 2 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \qquad B = \begin{bmatrix} 1 & 0 \\ -1 & 2 \\ 0 & -1 \end{bmatrix}
$$
$$
Y_{true} = A \cdot B = \begin{bmatrix} 1 & -2 \\ -1 & 2 \\ 2 & -1 \\ 0 & 2 \end{bmatrix}
$$
Now let's use the **Quantized Matrix Formula**, which computes matrices for each of the four terms:
1. **$A_q B_q$ (Main Matrix Multiplication)**: 
$$
\begin{bmatrix} 2(3)+1(1)+3(2) & 2(2)+1(4)+3(1) \\ 1(3)+2(1)+1(2) & 1(2)+2(4)+1(1) \\ 3(3)+1(1)+2(2) & 3(2)+1(4)+2(1) \\ 2(3)+2(1)+1(2) & 2(2)+2(4)+1(1) \end{bmatrix} = \begin{bmatrix} 13 & 11 \\ 7 & 11 \\ 14 & 12 \\ 10 & 13 \end{bmatrix}
$$
2. **$Z_B \cdot \text{rowsum}(A_q)$**: $\text{rowsum}(A_q) = \begin{bmatrix} 6 \\ 4 \\ 6 \\ 5 \end{bmatrix}$. Multiply by $Z_B (2)$ and broadcast to match output shape: $\begin{bmatrix} 12 & 12 \\ 8 & 8 \\ 12 & 12 \\ 10 & 10 \end{bmatrix}$
3. **$Z_A \cdot \text{colsum}(B_q)$**: $\text{colsum}(B_q) = \begin{bmatrix} 6 & 7 \end{bmatrix}$. Multiply by $Z_A (1)$ and broadcast to match output shape: $\begin{bmatrix} 6 & 7 \\ 6 & 7 \\ 6 & 7 \\ 6 & 7 \end{bmatrix}$
4. **$Z_A Z_B \cdot k$**: $1 \times 2 \times 3 = 6$. Broadcasted everywhere: $\begin{bmatrix} 6 & 6 \\ 6 & 6 \\ 6 & 6 \\ 6 & 6 \end{bmatrix}$

Plugging these matrices into the formula:
$$
Y = \begin{bmatrix} 13 & 11 \\ 7 & 11 \\ 14 & 12 \\ 10 & 13 \end{bmatrix} - \begin{bmatrix} 12 & 12 \\ 8 & 8 \\ 12 & 12 \\ 10 & 10 \end{bmatrix} - \begin{bmatrix} 6 & 7 \\ 6 & 7 \\ 6 & 7 \\ 6 & 7 \end{bmatrix} + \begin{bmatrix} 6 & 6 \\ 6 & 6 \\ 6 & 6 \\ 6 & 6 \end{bmatrix} = \begin{bmatrix} 1 & -2 \\ -1 & 2 \\ 2 & -1 \\ 0 & 2 \end{bmatrix}
$$
The result is perfectly identical! The quantized formula lets the processor calculate the heavy matrix multiplication entirely in raw integer math, applying the much simpler rowsum/colsum vector additions at the very end.


When $Z_A = Z_B = 0$ (symmetric quantization), these last three correction terms vanish entirely. This eliminates $\mathcal{O}(mn)$ addition operations from an $\mathcal{O}(mnk)$ matrix multiplication—a significant constant-factor reduction when computing large tensors.

**When to use which:**
- **Symmetric quantization** is generally best for **weights**. Neural network weights naturally tend to form symmetric distributions centered directly around zero, making symmetric ranges highly efficient.
- **Affine quantization** is crucial for **activations**. *(Note: in deep learning, the term "activations" refers to the actual data tensors flowing out of an activation function, not the math function itself. These data tensors must be quantized so the next layer can use them in integer math).* Because activation functions often restrict these data tensors to skewed or positive-only ranges—like ReLU ($[0, \infty)$) or Sigmoid ($[0, 1]$)—their output tensors often contain no negative numbers. If we forced a symmetric range (e.g., $[-15.0, 15.0]$) onto a ReLU output, the entire negative half of our $256$ available INT8 slots would literally never be used, throwing away half of our precision. Affine quantization allows the zero-point to shift the integer range so that all $256$ slots are tightly packed around just the actual numbers present, halving the rounding error.

---

## Quantized Matrix Multiplication

Let me walk through how integer arithmetic is actually used in inference, since this is the payoff of quantization.

We want to compute $Y = X · W$ where $X ∈ ℝ^{m×k}$ and $W ∈ ℝ^{k×n}$. After quantizing both tensors with their respective parameters:

$$
X \approx S_X \cdot (X_q - Z_X), \qquad W \approx S_W \cdot (W_q - Z_W)
$$

Substituting:

$$
Y = X \cdot W \\
  \approx S_X \cdot S_W \cdot (X_q - Z_X)(W_q - Z_W) \\
  = S_X \cdot S_W \cdot \bigl[X_q W_q \;-\; Z_W \cdot X_q \mathbf{1} \;-\; Z_X \cdot \mathbf{1}^\top W_q \;+\; Z_X Z_W \cdot k\bigr]
$$

Where the matrix dimensions are:
- **$Y$** and all resulting terms in the brackets are **$m \times n$**
- **$X$** and **$X_q$** are **$m \times k$**
- **$W$** and **$W_q$** are **$k \times n$**
- **$\mathbf{1}$** in $X_q \mathbf{1}$ is a **$k \times n$** matrix of ones, and **$\mathbf{1}^\top$** in $\mathbf{1}^\top W_q$ is an **$m \times k$** matrix of ones

Breaking this down:
1. **$X_q W_q$** — The main integer matrix multiplication, accumulating into int32. This is the expensive operation that runs on integer tensor cores.
2. **$Z_W \cdot \text{rowsum}(X_q)$** — Because $Z_W$ is a scalar, the dot product along the inner dimension of length $k$ just factors $Z_W$ out, leaving a simple sum across the row of $X_q$. This is computed in $\mathcal{O}(mk)$ rather than $\mathcal{O}(mnk)$.
3. **$Z_X \cdot \text{colsum}(W_q)$** — Similarly, $Z_X$ factors out, leaving a sum down the column of $W_q$. Because $W$ is fixed during inference, this can be precomputed once.
4. **$Z_X Z_W k$** — Both $Z_X$ and $Z_W$ are scalars, so the dot product simply adds $Z_X Z_W$ to itself $k$ times. This is also precomputed.
5. **$S_X S_W$** — A final scalar multiply to convert the integer result back to float32.

Most of the heavy lifting happens in step (1), which is handled by specialized hardware designed for integer math (like NVIDIA Tensor Cores or similar chips from Intel and ARM). Because `int8` numbers take up only a quarter of the space of `float32` numbers, these processors can calculate them about 4 times faster.

When we multiply and add thousands of these `int8` numbers together, the final sum can easily grow into the millions. To ensure this sum doesn't "spill over" or exceed the maximum allowed value (a problem known as overflow), we temporarily store the running total in a larger data type: `int32`.

Once the math is finished, we don't keep the results in `int32`. We either shrink them back down to `int8` so the next layer of the neural network can process them efficiently, or we convert them back to `float32` if the next step doesn't support integers.

---

## Granularity: Finding the Right Level of Detail

When quantizing a neural network, we have to decide how many numbers share the same **scale** and **zero-point**. If we assign one scale to a massive block of numbers, we save memory, but we risk losing precision if those numbers vary wildly. If we assign scales to smaller groups, we get better accuracy at the cost of using slightly more memory to store all those extra scales. This trade-off is called **granularity**.

Here are the standard levels of granularity used in practice:

### 1. Per-tensor (The Broadest Stroke)
We calculate a single scale and zero-point for the *entire* matrix. 
- **The Pros:** It uses virtually no extra memory.
- **The Cons:** It is highly vulnerable to outliers. 
- **Example:** Imagine a $1000 \times 1000$ weight matrix where almost all values are between -1 and 1, but one single cell holds the value 50. With per-tensor quantization, the scale must stretch to accommodate the 50 across the entire matrix. This crushes the normal -1 to 1 values into just a handful of integer steps, ruining their precision.

### 2. Per-channel / Per-axis (The Sweet Spot)
Instead of one scale for the whole matrix, we calculate a separate scale for each *row* or *column* (each output channel).
- **The Pros:** If one channel has a massive outlier, it only ruins the precision for that specific channel. The rest of the channels get their own, tightly-fitted scales. 
- **Example:** Consider an image filter where the Red channel has extreme values (-50 to 50) but Green and Blue are muted (-1 to 1). Per-channel assigns a wide scale to Red and tight, highly precise scales to Green and Blue. The outlier in Red doesn't destroy the precision of the other colors.
- **The Verdict:** This is PyTorch's default for quantizing weights because it consistently boosts accuracy by 0.5–2% with almost zero memory penalty.

### 3. Per-token (For Large Language Models)
When processing text, different words (tokens) can trigger wildly different activation patterns. Per-token granularity calculates a fresh scale for the activations of *each individual token* as it passes through the network.
- **Example:** Imagine an AI reading the sentence "The quick brown fox." The word "fox" might trigger a massive numerical spike in the model, while the word "The" is very quiet. Per-token assigns a unique scale to the numbers generated by "fox" and a separate, tighter scale to the numbers from "The", keeping both perfectly accurate.
- **The Verdict:** This is essential for preventing the extreme outliers that commonly appear in the activations of massive language models.

### 4. Per-group (The Ultra-fine Detail)
When compressing models to extremes (like 4-bit quantization), even per-channel isn't detailed enough. Per-group granularity chops each row of the matrix into tiny blocks—usually 32 or 128 numbers—and gives each block its own scale.
- **The Pros:** Incredible precision that preserves the model's intelligence even when crushed down to 4-bit weights.
- **The Cons:** It requires storing thousands of extra scale factors. 
- **Example:** If you have a single matrix row containing 4096 weights, per-group might chop it into 32 separate blocks of 128 weights. If block #5 contains a massive outlier, only block #5 gets a stretched scale. The remaining 31 blocks stay completely unaffected and highly precise.
- **The Verdict:** This is the secret sauce behind modern 4-bit LLM compression techniques like GPTQ and AWQ.

---

## Calibration

Weight quantization parameters are derived directly from the weight tensors at quantization time and are therefore fixed entirely offline. Activations present a harder problem: their distributions are input-dependent, making them unknowable until inference time.

Calibration is the process of estimating the float range $[\alpha, \beta]$ for each activation tensor before deployment, so that the scale $S$ and zero-point $Z$ can be frozen into the model as constants. The choice of calibration method determines how accurately those constants capture the true runtime distribution — and directly controls the trade-off between rounding error (from a wide range) and clipping error (from a narrow one).

---

### Post-training Dynamic Quantization

The scale for each activation tensor is computed on-the-fly at runtime from the actual min/max of the current input batch. No calibration dataset is required.

$$
\alpha_{\text{batch}} = \min(x), \quad \beta_{\text{batch}} = \max(x), \quad S = \frac{\beta_{\text{batch}} - \alpha_{\text{batch}}}{255}
$$

**Example.** A single batch of ReLU activations spans $[0.0,\ 3.7]$:

$$
S = \frac{3.7 - 0.0}{255} \approx 0.01451, \quad Z = \text{round}\!\left(-128 - \frac{0.0}{0.01451}\right) = -128
$$

The next batch might span $[0.0,\ 5.1]$, producing $S \approx 0.02$ — an entirely fresh scale computed on the spot.

- **Advantage:** Zero approximation error on the range; always adapts to the actual input.
- **Disadvantage:** Adds per-forward-pass overhead; not supported on all hardware backends since it requires runtime range computation.
- **Best suited for:** Memory-bandwidth-bound models — RNNs, large-embedding Transformer encoders at batch size 1 — where compute overhead is negligible. PyTorch's `quantize_dynamic` implements this for `nn.Linear` and `nn.LSTM`.

---

### Post-training Static Quantization

A representative calibration dataset (typically 100–200 batches) is run through the model with observer modules attached to each activation tensor. The collected statistics are used to compute $[\alpha, \beta]$ once; those values are then frozen into the quantized model as constants. At inference time, no range computation occurs — the stored $S$ and $Z$ are used directly.

The choice of calibration algorithm determines what $[\alpha, \beta]$ is extracted from the collected activation histograms. Each method makes a different bet on the core tension underlying all of them:

> **The fundamental trade-off:** A wider range $[\alpha, \beta]$ guarantees no clipping but increases the step size $S = (\beta - \alpha)/255$, coarsening the grid and enlarging rounding error (bounded by $S/2$). A narrower range shrinks $S$ and improves precision for the bulk of values, but clips outliers to the boundary, introducing unbounded error on those samples. Every calibration method is really a different policy for where to draw that line.

#### Min-Max

$$
\alpha = \min_{\text{all batches}} x, \qquad \beta = \max_{\text{all batches}} x
$$

Guaranteed to contain every observed value, so clipping error is zero. But a single extreme outlier stretches $S$ for the entire tensor.

**Example.** 500 activation values, 499 in $[-1.0,\ 1.0]$ and one outlier at $47.3$:

$$
S_{\text{min-max}} = \frac{47.3 - (-47.3)}{255} \approx 0.371
$$

The 499 normal values now span only $\frac{2.0}{0.371} \approx 5.4$ integer steps — nearly all 256 levels are wasted on the outlier's range. The outlier is preserved exactly; everything else loses precision.

#### Moving Average Min-Max

$$
\alpha \leftarrow \gamma \cdot \alpha + (1-\gamma) \cdot \min(x_{\text{batch}}), \qquad \beta \leftarrow \gamma \cdot \beta + (1-\gamma) \cdot \max(x_{\text{batch}})
$$

where $\gamma \in [0, 1)$ is a momentum hyperparameter (typically $\gamma = 0.9$). Rather than committing to any single batch's extreme, the range is smoothed across calibration batches, making it more robust to anomalous spikes.

**Example.** Three batches with observed maxima $[1.2,\ 1.5,\ 8.0]$ and $\gamma = 0.9$, starting from $\beta_0 = 0$:

| Batch | Observed max | Running $\beta$ |
|---|---|---|
| 1 | 1.2 | $0.9 \times 0 + 0.1 \times 1.2 = 0.12$ |
| 2 | 1.5 | $0.9 \times 0.12 + 0.1 \times 1.5 = 0.258$ |
| 3 | 8.0 | $0.9 \times 0.258 + 0.1 \times 8.0 = 1.032$ |

The spike in batch 3 nudges $\beta$ to only $1.032$ rather than hard-anchoring to $8.0$ as min-max would. Over a full calibration run, the moving average settles near the true typical range.

#### Percentile Clipping

$$
\alpha = \text{percentile}(x,\ p), \qquad \beta = \text{percentile}(x,\ 1-p), \qquad \text{e.g., } p = 0.001
$$

Deliberately discards the most extreme $p\%$ on each tail, accepting bounded clipping error on those samples in exchange for a much narrower range and finer grid for the remaining values.

**Example.** Same 500-value distribution, with $p = 0.002$ (clip one value per tail):

$$
S_{\text{percentile}} = \frac{1.0 - (-1.0)}{255} \approx 0.00784
$$

The 499 typical values now span $\frac{2.0}{0.00784} \approx 255$ integer levels — nearly the full int8 grid. The single outlier at $47.3$ clips to $1.0$, incurring a clipping error of $46.3$ on that one sample. Whether that trade-off is acceptable depends on outlier frequency and model sensitivity.

#### Entropy (KL Divergence) Minimization

Finds the range $[\alpha^*, \beta^*]$ that minimizes the information lost when compressing the float32 activation histogram into 256 bins:

$$
[\alpha^*, \beta^*] = \arg\min_{[\alpha,\, \beta]} \; D_{\mathrm{KL}}\!\left(P_{\text{float}} \;\|\; P_{\text{quant}}\right)
$$

where $P_{\text{float}}$ is the observed float32 histogram and $P_{\text{quant}}$ is the histogram reconstructed after quantizing and dequantizing with the candidate range. A range that clips too aggressively inflates the boundary bins of $P_{\text{quant}}$ (high KL from tail mass piling up); a range that is too wide wastes bins on sparse regions, leaving the dense center coarsely covered (high KL from rounding error). KL minimization penalizes both failure modes and finds the threshold where the 256 levels are most *informatively* placed — in practice around $2.5\sigma$–$3.5\sigma$ for near-Gaussian activations. TensorRT uses this method by default.

> For the full derivation, step-by-step algorithm, and a worked numerical example, see [Appendix A: KL Calibration In Depth](#appendix-a-kl-divergence-calibration-in-depth).

#### MSE Minimization

$$
[\alpha^*, \beta^*] = \arg\min_{[\alpha,\, \beta]} \; \mathbb{E}\!\left[(x - \hat{x})^2\right]
$$

where $\hat{x} = \text{dequantize}(\text{quantize}(x;\,\alpha, \beta))$ and the expectation is over the calibration data. This directly minimizes the metric that propagates into downstream accuracy. Candidate ranges are evaluated on a grid; the one with the lowest mean squared reconstruction error is selected.

**Example.** Three candidate ranges evaluated on the 500-value distribution (499 values in $[-1, 1]$, one outlier at $47.3$):

| Range | $S$ | Rounding MSE | Clipping MSE | Total MSE |
|---|---|---|---|---|
| $[-47.3,\ 47.3]$ (min-max) | 0.371 | $\approx 0.034$ | $0$ | $0.034$ |
| $[-1.0,\ 1.0]$ (percentile) | 0.00784 | $\approx 1.5 \times 10^{-5}$ | $\approx 4.4 / 500 = 0.0088$ | $\approx 0.0088$ |
| $[-2.0,\ 2.0]$ (MSE optimal) | 0.01569 | $\approx 6 \times 10^{-5}$ | $0$ | $\approx 6 \times 10^{-5}$ |

MSE calibration selects $[-2.0,\ 2.0]$ — wide enough to avoid clipping the outlier's neighborhood, narrow enough to keep rounding error small. Entropy and MSE calibration find the empirically optimal balance; min-max minimizes clipping at the cost of maximal rounding error when outliers are present.

---

### Quantization-aware Training

All PTQ methods share a fundamental limitation: because the model was trained in full precision (float32 or bfloat16), the weights that minimize the original loss function are rarely aligned with the discrete steps of a quantization grid. Rounding these weights post-training introduces accumulated error across layers, which can severely degrade model accuracy—especially in extreme low-bit regimes (like 4-bit quantization), where each parameter is restricted to just 16 representable values. QAT closes this accuracy gap by directly incorporating quantization constraints into the training loop.

#### The Fake-Quantize Operator

QAT achieves this by inserting **fake-quantize (FQ)** operators immediately after weight and activation tensors in the model's computational graph during the forward pass:

$$
x_{\text{fq}} = S \cdot \text{clip}\!\left(\text{round}\!\left(\frac{x}{S} + Z\right),\; q_{\min},\; q_{\max}\right) - S \cdot Z
$$

The resulting tensor $x_{\text{fq}}$ remains in float32 precision, but its values are restricted to a discrete, finite set matching the grid of the target integer format: $\{S(q - Z) : q \in [q_{\min}, q_{\max}]\}$. Consequently, every subsequent operation (like matrix multiplications) in the forward pass runs on these discretized values, mirroring deployment-time inference. Because the model's loss is evaluated under this discretization, gradients guide the weights toward configurations that are inherently robust to quantization.

Essentially, the fake-quantize operator simulates the quantization-dequantization pipeline in-place:

$$
x_{\text{fq}} = \text{dequantize}\!\bigl(\text{quantize}(x;\; S, Z)\bigr)
$$

allowing it to act as a drop-in simulator at any layer.

#### The Straight-Through Estimator

However, the `round()` function is a piecewise constant step function; its mathematical derivative is zero almost everywhere and undefined at integers. If backpropagated directly, this zero gradient would halt training, as no optimization signal could pass backward through the rounding node to update the weights.

To circumvent this, QAT employs the **Straight-Through Estimator (STE)**, which approximates the gradient of the non-differentiable operations. Specifically, the gradient of the fake-quantize operator with respect to its input $x$ is modeled as:

$$
\frac{\partial x_{\text{fq}}}{\partial x} \approx \mathbf{1}\!\left[q_{\min} \leq \frac{x}{S} + Z \leq q_{\max}\right]
$$

Consequently, the gradient of the loss $\mathcal{L}$ with respect to the input $x$ is backpropagated as:

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial x_{\text{fq}}} \cdot \frac{\partial x_{\text{fq}}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial x_{\text{fq}}} \cdot \mathbf{1}\!\left[q_{\min} \leq \frac{x}{S} + Z \leq q_{\max}\right]
$$

This formulation allows gradients to flow unaltered through the rounding operation as if it were an identity function, while zeroing out gradients for values that lie outside the clipping range. This behavior correctly encourages the optimizer to nudge out-of-bound weights back toward the clipping boundaries rather than stranding them with zero-gradient updates.

**Intuition behind the STE.** Although mathematically heuristic, the STE works remarkably well in practice. While the rounding function introduces local discontinuities, the global loss landscape remains relatively smooth—especially for 8-bit grids with 256 levels. The STE provides a gradient direction that is correct *on average*, allowing stochastic gradient descent (SGD) and its variants to navigate the loss landscape effectively.

#### Parameter Adaptation Dynamics

To understand how QAT alters parameters, consider a single weight $w$ under symmetric 8-bit quantization with $S = 0.00784$ (representing a range of $[-1, 1]$). The two nearest grid points to a current weight value of $w = 0.317$ are $0.314$ and $0.322$.

- **Post-Training Quantization (PTQ):** The weight remains fixed at $0.317$ during compilation and is simply rounded to $0.314$ at inference, resulting in a rounding error of $0.003$.
- **Quantization-Aware Training (QAT):**
  1. **Forward Pass:** The weight is fake-quantized to $w_{\text{fq}} = 0.314$. The model's loss is calculated using this discretized value.
  2. **Backward Pass:** The STE propagates the gradient $\frac{\partial \mathcal{L}}{\partial w_{\text{fq}}}$ directly back to the continuous weight variable $w$.
  3. **Weight Update:** If $\frac{\partial \mathcal{L}}{\partial w_{\text{fq}}} < 0$ (meaning a larger weight value would reduce the loss), the optimizer nudges the continuous weight $w$ upward toward the next grid point ($0.322$).
  4. **Convergence:** Over multiple optimization steps, the continuous weight $w$ might converge to $0.3215$. When final quantization is applied, it rounds to $0.322$ with a significantly smaller error of $0.0005$.

Aggregated across millions of parameters, this process causes the entire weight distribution to naturally migrate toward the quantization grid points. Rather than just forcing arbitrary rounding, the optimizer searches for a coordinate-aligned parameter configuration that minimizes task degradation.

#### Scale and Zero-point Learning

In standard QAT, the scale $S$ and zero-point $Z$ are also parameterized and optimized during training (as in Learned Step Size Quantization, or LSQ). Under the STE framework, the gradient of the fake-quantized value $x_{\text{fq}}$ with respect to the scale $S$ is derived as:

$$
\frac{\partial x_{\text{fq}}}{\partial S} \approx \begin{cases} 
\frac{x_{\text{fq}}}{S} - \frac{x}{S} & \text{if } q_{\min} < \frac{x}{S} + Z < q_{\max} \\
q_{\min} - Z & \text{if } \frac{x}{S} + Z \le q_{\min} \\
q_{\max} - Z & \text{if } \frac{x}{S} + Z \ge q_{\max}
\end{cases}
$$

This can be written compactly using indicator functions as:

$$
\frac{\partial x_{\text{fq}}}{\partial S} \approx \frac{x_{\text{fq}}}{S} - \frac{x}{S} \cdot \mathbf{1}\!\left[q_{\min} \leq \frac{x}{S} + Z \leq q_{\max}\right]
$$

Using the chain rule, the gradient of the loss $\mathcal{L}$ with respect to the scale $S$ is backpropagated as:

$$
\frac{\partial \mathcal{L}}{\partial S} = \sum \frac{\partial \mathcal{L}}{\partial x_{\text{fq}}} \cdot \frac{\partial x_{\text{fq}}}{\partial S}
$$

This formulation enables the network to dynamically learn the optimal quantization range (clipping boundaries) for each layer, adjusting scale parameters to minimize the sum of rounding and clipping errors. In practice, learnable scales are initialized from PTQ calibration statistics and then fine-tuned, which converges much faster than training them from scratch.

#### Practical Training Recipe

A standard QAT pipeline follows a structured, multi-phase schedule:

1. **Start with a Pretrained Checkpoint:** While QAT can be run from scratch, it is significantly faster and more stable to start from a converged float32 checkpoint. This positions the optimizer near a high-performing local minimum.
2. **Insert Fake-Quantization Operators:** Use framework-provided utilities (e.g., `torch.ao.quantization.prepare_qat` in PyTorch) to inject calibration observers and fake-quantize nodes at target weight and activation tensors.
3. **Warm-up / Weight Adaptation (Epochs 1–2):** Train with fake-quantization active but with fixed scale and zero-point parameters. This allows the weights to adapt to the initial discretization grid before the grid boundaries themselves begin to shift.
4. **Joint Optimization:** Enable learning for both weights and scale/zero-point parameters. Reduce the learning rate (typically by $10\times$ compared to the original pretraining rate) to prevent the optimizer from destabilizing the pretrained parameters.
5. **Freeze Batch Normalization (Final Epochs):** For CNNs, freeze the running mean and variance statistics of Batch Normalization layers. This ensures that the model's inference-time behavior matches the training-time simulations.
6. **Convert to Integer Representation:** Remove the fake-quantize operators, replace floating-point operations with integer kernels, pack weights into low-bit widths, and compile the final deployment-ready model.

The training cost for int8 QAT is typically $10\%$–$20\%$ of the original pretraining compute budget, which can rise to $30\%$–$50\%$ for 4-bit configurations due to the increased optimization difficulty.

#### Trade-offs and Decision Framework

| Dimension | Post-Training Quantization (PTQ-Static) | Quantization-Aware Training (QAT) |
|---|---|---|
| **Data Requirement** | Small calibration set (100–200 batches) | Full training dataset |
| **Compute Overhead** | Minutes (typically single-CPU/GPU) | Hours/Days ($10\%$–$50\%$ of pretraining time) |
| **8-bit Accuracy Gap** | $0.5\%$–$3.0\%$ degradation | $\le 0.5\%$ degradation (often matches FP32) |
| **4-bit Accuracy Gap** | $3.0\%$–$10.0\%$ degradation (often unusable) | $0.5\%$–$2.0\%$ degradation |
| **Infrastructure Needed** | Export and calibration tools | Full training environment with backprop support |
| **Scale Optimization** | Fixed during calibration | Dynamically learned alongside weights |

**Selection Criteria:**
Static PTQ is the standard choice for most 8-bit deployments, as it requires minimal data and compute. However, QAT is highly recommended when:
- **Targeting sub-8-bit precision (e.g., 4-bit weights or activations):** The coarse grid of 16 levels makes PTQ rounding errors too severe, leading to significant accuracy drop.
- **Handling challenging architectures:** Models using depthwise separable convolutions (like MobileNets) or containing heavy-tailed activations/weights are sensitive to static quantization.
- **Deploying accuracy-critical applications:** When downstream task performance must strictly match the floating-point baseline.

---

## LLM Quantization

Large language models introduce a specific challenge that makes naive PTQ fail: **activation outliers**.

### The Outlier Problem

Transformer models above roughly 6.7B parameters develop a small subset of embedding dimensions — typically ~0.01% of channels — with activation magnitudes 100–1000× larger than typical values. These outliers are systematic: they appear in the same channels regardless of input and persist across layers.

The effect on per-tensor int8 quantization is severe. If one activation channel has magnitude 500 while typical values are in [−1, 1], per-tensor S ≈ 500/127 ≈ 3.94. Values in [−1, 1] all quantize to x_q ∈ {0}, collapsing the information in 99.99% of the channels to a single integer. This is why naive int8 quantization of LLMs produces gibberish, while the same approach works well for BERT-base at 110M parameters.

### LLM.int8() — Mixed-precision Decomposition

Dettmers et al. (2022) address outliers with **mixed-precision decomposition**: at runtime, detect which columns of the activation matrix contain outliers (those exceeding `llm_int8_threshold`, default 6.0), extract them, and route them through a separate fp16 matmul:

$$
Y = \underbrace{X_{\text{fp16}}[\,:,\,\mathcal{O}] \cdot W_{\text{fp16}}[\mathcal{O},\,:]}_{\text{fp16 path, outlier columns } \mathcal{O}} \;+\; \underbrace{X_{\text{int8}}[\,:,\,\mathcal{N}] \cdot W_{\text{int8}}[\mathcal{N},\,:]}_{\text{int8 path, normal columns } \mathcal{N}}
$$
The results are added in fp16. Since outlier columns are typically ~0.1% of the total, the int8 path handles 99.9% of the compute, and the fp16 path is negligible in FLOPs but essential for accuracy.

Memory: weights are stored in int8, giving 2× compression over fp16. The threshold 6.0 was determined empirically; lowering it increases the fp16 path and accuracy but reduces int8 efficiency.

### SmoothQuant — Migrating Difficulty to Weights

SmoothQuant (Xiao et al., 2022) makes a key observation: activations are hard to quantize (due to outliers) but weights are easy; the reverse is true for the other direction. So why not transfer the quantization difficulty from activations to weights via a mathematically equivalent transformation?

For a linear layer Y = XW, insert a per-channel diagonal scaling:

$$
Y = XW = \bigl(X \cdot \text{diag}(s)^{-1}\bigr) \cdot \bigl(\text{diag}(s) \cdot W\bigr) = \tilde{X} \cdot \tilde{W}
$$
By choosing s_j = max(|X_{:,j}|)^α / max(|W_{j,:}|)^(1−α), outlier-heavy activation channels are smoothed (divided by a large s_j), and the corresponding weight channels are amplified (multiplied by s_j). With α = 0.5 the difficulty is evenly split. The result: both X̃ and W̃ have smooth distributions that quantize well with per-tensor int8.

This enables full W8A8 quantization (8-bit weights and 8-bit activations) without runtime decomposition, achieving true int8 throughput on hardware that supports it. The smoothing scales s are absorbed offline into the preceding layer's output projection or normalization.

### GPTQ — Second-order Weight Quantization

GPTQ (Frantar et al., 2022) is a **weight-only** post-training quantization method targeting 3–4 bits per weight. Activations remain in float16 at runtime; only the weight loading is compressed.

The insight: quantization error in one weight can be partially compensated by adjusting the remaining unquantized weights, if you know the second-order structure of the loss landscape.

Starting from the second-order Taylor approximation of the layer's output error with respect to a weight perturbation δw:

$$
\delta E \approx \frac{1}{2}\, \delta w^\top H\, \delta w, \qquad H = 2X^\top X
$$
When you quantize weight w_j to quant(w_j), the induced error is e_j = w_j − quant(w_j). To compensate, update the remaining weights:

$$
W_{:,\,j+1:} \;\leftarrow\; W_{:,\,j+1:} - e_j \cdot \frac{H^{-1}_{j,\,j+1:}}{H^{-1}_{jj}}
$$
This is the Optimal Brain Compression update, applied column by column. GPTQ processes columns in blocks for GPU efficiency and uses a Cholesky reformulation of H⁻¹ for numerical stability. The full weight matrix is quantized in a single pass over the calibration data.

Result: 4-bit GPTQ on LLaMA-2-70B achieves perplexity within ~1% of float16, with 4× memory reduction. The inference overhead is weight dequantization (int4 → fp16) per matmul, which is cheap relative to the memory bandwidth savings for large models.

### AWQ — Activation-aware Weight Quantization

AWQ (Lin et al., 2023) observes that weight channels corresponding to large-magnitude activations matter disproportionately for the output: if x_j is large, then W_{:,j} · x_j dominates the layer output, and quantization error in W_{:,j} is amplified.

AWQ searches for per-channel weight scales s ∈ ℝ^(d_in) that minimize the output MSE:

$$
\min_{s} \;\bigl\| WX - \tilde{W}(s)\,X \bigr\|_F^2, \qquad \text{where} \quad \tilde{W}_j(s) = \text{quant}\!\left(\frac{W_j}{s_j}\right) \cdot s_j
$$
The scales are found via a fast grid search on a calibration set. Scaling down important weight channels before quantization effectively gives them finer resolution at the cost of reduced resolution in less-important channels. AWQ consistently outperforms GPTQ at equal bit widths, particularly at 4-bit, and integrates with vLLM, TGI, and llama.cpp.

### NF4 — Normally-distributed Weights, Information-optimal Quantization

NF4 (Dettmers et al., 2023, used in QLoRA) is designed specifically for the empirical observation that pre-trained neural network weights follow approximately normal distributions N(0, σ²).

For a uniform quantizer on a non-uniform distribution, quantization error is unequal across the range: bins in dense regions (near zero) capture many values and have low average error; bins in sparse regions (far from zero) each capture few values but with potentially large error. An information-theoretically optimal quantizer should instead allocate bins so that each bin contains an equal probability mass — placing more levels near zero (where the density is high) and fewer at the tails.

NF4 defines its 16 quantization levels as the quantiles of a standard normal distribution N(0, 1):

$$
\text{levels} = \left\{ \Phi^{-1}\!\left(\frac{i + 0.5}{16}\right) \;\Bigg|\; i = 0, \ldots, 15 \right\}
$$
where Q_F⁻¹ is the inverse CDF of N(0, 1). Each level captures exactly 1/16 of the probability mass.

At runtime, weights in each quantization group are normalized by their absmax to [−1, 1], then looked up against the nearest NF4 level. This achieves equal expected quantization error per quantile bin — the optimal arrangement for the normal prior.

In QLoRA, base model weights are stored in NF4 (4-bit); LoRA adapter weights are trained in bf16. This allows fine-tuning 65B parameter models on a single 48GB GPU, with less than 1 perplexity point degradation on the quantized base relative to the bf16 base.

**Double quantization:** The NF4 absmax scales themselves are float32, adding ~0.5 bits/parameter overhead. QLoRA optionally quantizes these scales to fp8 (double quantization), recovering ~0.37 bits/parameter at negligible accuracy cost.

---

## Practical Implementation

### PyTorch — Dynamic Quantization

```python
import torch
import torch.quantization

model = ...  # trained model, eval mode

quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
```

Weights are quantized offline per-tensor; activations are quantized at runtime per forward pass. No calibration dataset required. The quantized weights are stored as int8 but immediately dequantized before the matmul — this is weight-only quantization, so the benefit is reduced memory bandwidth, not int8 compute throughput.

### PyTorch — Static Quantization

```python
import torch
import torch.quantization

model.eval()

# Fuse Conv + BN + ReLU into a single quantizable op
model_fused = torch.quantization.fuse_modules(
    model, [['conv', 'bn', 'relu']]
)

# Attach observers using the x86 backend's recommended qconfig
model_fused.qconfig = torch.quantization.get_default_qconfig('x86')
model_prepared = torch.quantization.prepare(model_fused)

# Run calibration
with torch.no_grad():
    for batch in calibration_loader:      # 100–200 batches is sufficient
        model_prepared(batch)

# Freeze ranges and convert to quantized ops
quantized_model = torch.quantization.convert(model_prepared)
```

After `convert`, `nn.Linear` becomes `nn.quantized.Linear`, which performs the full W8A8 integer matmul described above.

### NumPy — Affine Quantization from Scratch

```python
import numpy as np

def quantize(x: np.ndarray, q_min: int = -128, q_max: int = 127):
    alpha, beta = float(x.min()), float(x.max())
    S = (beta - alpha) / (q_max - q_min)
    Z = int(np.round(q_min - alpha / S))
    Z = int(np.clip(Z, q_min, q_max))
    x_q = np.clip(np.round(x / S + Z), q_min, q_max).astype(np.int8)
    return x_q, S, Z

def dequantize(x_q: np.ndarray, S: float, Z: int) -> np.ndarray:
    return S * (x_q.astype(np.float32) - Z)

rng = np.random.default_rng(0)
x = rng.standard_normal(1000).astype(np.float32)

x_q, S, Z = quantize(x)
x_hat = dequantize(x_q, S, Z)

print(f"Scale S = {S:.6f},  Zero-point Z = {Z}")
print(f"Max abs error: {np.abs(x - x_hat).max():.6f}  (bound: S/2 = {S/2:.6f})")
print(f"Mean abs error: {np.abs(x - x_hat).mean():.6f}")
```

### bitsandbytes — LLM int8 and NF4 via Transformers

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# LLM.int8() — 8-bit with mixed-precision outlier decomposition
model_int8 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# NF4 — 4-bit normal float (QLoRA base)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # compute precision during dequant
    bnb_4bit_use_double_quant=True,           # quantize the NF4 scales themselves
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

`bnb_4bit_compute_dtype` controls the precision of the dequantized matmul — set to bf16 for modern GPUs. `bnb_4bit_use_double_quant=True` applies double quantization to the absmax scales, saving an additional ~0.37 bits/parameter.

---

## Energy Efficiency

The common claim is that quantization reduces energy consumption. This is true in the limit but more nuanced in practice.

Energy savings have two sources: reduced memory bandwidth (fewer bytes transferred per weight loading) and reduced arithmetic energy (integer ALUs consume less power per operation than floating-point ALUs). Both benefits are most pronounced when the model is **memory-bandwidth-bound** — which occurs for large models and small batch sizes where each weight is loaded from DRAM once per token.

For quantization formats with non-trivial dequantization (NF4 lookup tables, GPTQ 4-bit unpacking), there is additional ALU overhead per matmul. For small models (< 3B parameters), this dequantization overhead can exceed the memory bandwidth savings, resulting in *higher* energy consumption than fp16 despite 4× smaller weights. For large models (≥ 7B), the bandwidth reduction dominates and quantization saves energy.

Batch size is often a larger energy lever than precision. Increasing batch size from 1 to 64 amortizes weight-loading cost across more token computations, reducing per-token energy by 80–95%. Continuous batching schemes (PagedAttention in vLLM) effectively increase utilization without increasing latency — often a larger win than changing precision format.

The practical takeaway: measure empirically for your specific model size, hardware, and batch size rather than assuming quantization universally reduces energy. The sign of the effect is not always what intuition predicts at small scale.

---

## Method Selection Guide

| Scenario | Recommended method |
|---|---|
| CNN / small Transformer, ≥1% accuracy drop acceptable | PTQ static, per-channel int8, entropy calibration |
| CNN / small Transformer, tight accuracy budget | QAT, int8 |
| LLM inference, memory-constrained, best accuracy | AWQ 4-bit |
| LLM inference, memory-constrained, fast quantization | GPTQ 4-bit |
| LLM inference, fp16 fits but want int8 throughput | SmoothQuant W8A8 |
| LLM inference, accuracy-first, moderate compression | LLM.int8() (bitsandbytes) |
| LLM fine-tuning on single GPU | QLoRA (NF4 base + bf16 LoRA adapters) |
| Quick baseline, no calibration data | Dynamic quantization |
| Below 8-bit on non-LLM models | QAT with per-channel quantization |

---

## References

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Jacob et al. (2017). The foundational paper deriving the affine quantization scheme and quantized matmul arithmetic.
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) — Nagel et al. (2021). Comprehensive theoretical treatment of PTQ and QAT, calibration, and per-channel quantization.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al. (2022). Identifies the activation outlier problem and proposes mixed-precision decomposition.
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — Xiao et al. (2022). Activation smoothing for W8A8 quantization.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al. (2022). Second-order weight quantization via approximate Hessian.
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al. (2023). Per-channel weight scaling to protect salient channels.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023). Introduces NF4 and double quantization.
- [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) — Wikipedia. Bit-layout comparison across floating-point formats.

---

## Appendix A: KL Divergence Calibration In Depth

### What KL Divergence Measures Here

KL divergence between two distributions $P$ and $Q$ is defined as:

$$
D_{\mathrm{KL}}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
$$

It quantifies the extra bits needed to encode samples drawn from $P$ using a code optimized for $Q$. In the calibration context, $P = P_{\text{float}}$ is the "true" distribution of float32 activations observed over the calibration dataset, and $Q = P_{\text{quant}}$ is the distribution that results after quantizing those activations to int8 and dequantizing back to float32. Every quantization step collapses a continuous interval of float values to a single representative point, introducing a distortion; the KL score measures how much information that distortion destroys.

The goal: choose $[\alpha, \beta]$ so that the 256 int8 levels are placed where they carry the most information about the original distribution.

### Why Both Extremes Fail

**Too wide (min-max style).** A range $[-T_{\text{large}},\ T_{\text{large}}]$ spreads 256 bins evenly across a large interval. In the dense central region of the activation distribution — where most values live — each bin covers a wide float interval, introducing large rounding error. The $P_{\text{quant}}$ histogram is smeared relative to $P_{\text{float}}$: many float32 bins that were distinguishable are now merged into the same int8 bin. KL is high because $P_{\text{quant}}$ is a blurry approximation.

**Too tight.** A range $[-T_{\text{small}},\ T_{\text{small}}]$ gives finely-spaced bins in the center but forces all values beyond $\pm T_{\text{small}}$ to clip to the boundary integer. In $P_{\text{quant}}$, this manifests as a large artificial spike at the boundary bins — mass that was spread across the far tails in $P_{\text{float}}$ gets piled onto a single point in $P_{\text{quant}}$. KL is high because those boundary probabilities diverge: $P_{\text{float}}$ assigns very little mass there, but $P_{\text{quant}}$ assigns a lot.

The KL curve as a function of threshold $T$ therefore has a characteristic **U-shape**, with a minimum at the threshold that best balances rounding error against clipping error.

### The Algorithm

TensorRT (and similar implementations) run the following procedure per activation tensor:

1. **Collect a fine histogram.** Run the calibration dataset through the model and accumulate a float32 activation histogram $H$ with high resolution — typically 2048 bins spanning the full observed range $[\min x,\ \max x]$.

2. **Grid search over thresholds.** For each candidate threshold $T \in \{T_1, T_2, \ldots\}$ (e.g., 128 values linearly spaced from 0 to $\max|x|$):
   - **Clip:** Truncate $H$ at $\pm T$, adding the clipped tail mass to the outermost bins.
   - **Quantize:** Merge the 2048-bin clipped histogram down to 256 bins (one per int8 level), summing the probability mass in each group.
   - **Expand:** Spread each of the 256 bins' mass back uniformly across the original 2048-bin positions it covers, producing $H_{\text{expanded}}$.
   - **Score:** Compute $D_{\mathrm{KL}}(H_{\text{clipped}} \| H_{\text{expanded}})$.

3. **Select the minimum.** $T^* = \arg\min_T D_{\mathrm{KL}}$.

4. **Set parameters.** For symmetric activations: $\alpha = -T^*$, $\beta = T^*$. For asymmetric: repeat the search independently per tail.

### Worked Example

Suppose calibration yields activations approximating $\mathcal{N}(0, 1)$ with rare outliers out to $\pm 6$. We evaluate three candidate thresholds:

| Threshold $T$ | Range | $S = 2T / 255$ | Outcome |
|---|---|---|---|
| $T = 6.0$ | $[-6,\ 6]$ | $0.0471$ | Each bin spans $0.047$ units. The central $[-1, 1]$ region maps to only $\approx 42$ of 256 bins — most precision is wasted on near-empty tails. High rounding error. KL dominated by the dense center being coarsely covered. |
| $T = 1.5$ | $[-1.5,\ 1.5]$ | $0.01176$ | Bins are fine-grained, but $\approx 13\%$ of probability mass (tails beyond $\pm 1.5\sigma$) clips to the boundary. $P_{\text{quant}}$ has a large spike at $\pm 1.5$ that $P_{\text{float}}$ does not. High KL from clipping. |
| $T = 3.0$ | $[-3,\ 3]$ | $0.02353$ | Only $\approx 0.3\%$ of mass (beyond $3\sigma$) clips. The central region uses $\approx 85$ of 256 bins — still somewhat coarse but acceptable. Clipping penalty is tiny. **KL is minimized.** |

**The U-shape in numbers.** If we denote the KL scores as $D(6.0) \approx 0.18$, $D(1.5) \approx 0.31$, $D(3.0) \approx 0.04$ (illustrative), the minimum is clearly near $T = 3.0$. For a near-Gaussian activation distribution, the KL-optimal threshold is typically in the range $2.5\sigma$–$3.5\sigma$, consistent with the fact that a normal distribution has only $0.006\%$–$4.6\%$ of its mass beyond those boundaries.

### Comparison with MSE

Both KL and MSE calibration outperform min-max when outliers are present. The practical differences are subtle:

| Property | KL (Entropy) | MSE |
|---|---|---|
| Objective | Information preserved | Reconstruction error |
| Sensitivity to tails | Penalizes clipping mass piling up at boundary | Penalizes large individual errors ($e^2$ grows fast) |
| Typical optimal $T$ | Slightly wider — allows more tail clipping if the dense region benefits | Slightly narrower — outlier squared error is large, so avoids clipping even rare extremes |
| Default in | TensorRT | PyTorch `torch.ao` observers |

In most practical deployments the two methods produce nearly identical results. KL is preferred when the activation histogram is multimodal or has fat tails; MSE is preferred when a small number of very large outliers exist and their squared error would dominate.

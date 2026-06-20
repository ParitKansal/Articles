# Soft Prompts: Learning Tasks Without Updating the Model

**Excerpt:** Soft Prompt Tuning is an elegant parameter-efficient fine-tuning (PEFT) method that steers a frozen language model by prepending trainable continuous vectors to the input. This article explores how soft prompts work, whether they correspond to hidden English instructions, and why random initialization often outperforms instruction-based initialization.

**Read Time:** ~8 minutes

---

## Introduction

Large Language Models (LLMs) are typically adapted to new downstream tasks through fine-tuning. Traditional full fine-tuning updates millions or even billions of model parameters, making it computationally expensive and requiring a completely separate copy of the massive model for every single task.

Parameter-Efficient Fine-Tuning (PEFT) methods were introduced to solve this exact problem by updating only a minuscule fraction of the model's parameters while keeping the original pretrained model largely unchanged.

Among these methods, **Soft Prompt Tuning** stands out as one of the simplest and most elegant approaches. Instead of modifying transformer weights or inserting new neural network layers inside the model, Soft Prompt Tuning learns a small set of trainable vectors. These continuous vectors are prepended to the input sequence and used to directly steer the model's behavior.

Despite its simplicity, Soft Prompt Tuning raises several fascinating questions: What exactly are soft prompts? How are they different from ordinary discrete text prompts? Can they be converted back into human-readable text? And does initializing them with meaningful human instructions actually improve performance? 

This article explores both the theory and the surprising practical experiments behind Soft Prompt Tuning.

---

## From Hard Prompts to Soft Prompts

### Hard Prompts

Traditional prompting, often called "hard prompting," uses natural language instructions written by humans. 

For example, you might provide the model with:
* *"Classify the sentiment of the following movie review."*
* *"Translate the following sentence into French."*

These prompts are composed of actual, discrete words from the model's vocabulary and are directly interpretable by humans. The model receives the hard prompt appended to the input and performs the requested task. However, the effectiveness of hard prompts is notoriously brittle; performance depends heavily on exact wording, structure, and phrasing.

### Soft Prompts

Soft prompts take a fundamentally different approach. Instead of providing discrete textual instructions, we learn a small collection of trainable, continuous vectors.

![Hard Prompts vs Soft Prompts](https://raw.githubusercontent.com/ParitKansal/Articles/main//prompt_comparison.png)

These soft vectors:
* Are not distinct words from a vocabulary.
* Do not correspond to any readable human sentence.
* Exist directly in the continuous embedding space of the model.
* Are optimized automatically through gradient descent.

The final model input becomes the concatenation of the trainable prompt vectors and the frozen input token embeddings. During training, the model parameters remain entirely frozen; only the prompt vectors are updated to minimize the loss.

---

## The Mathematical View

Suppose an input text sequence is converted into a matrix of token embeddings:
$$ X = [x_1, x_2, \dots, x_n] $$
where each $x_i$ is a token embedding vector.

We introduce a trainable soft prompt matrix:
$$ P = [p_1, p_2, \dots, p_m] $$

The model receives the concatenated sequence:
$$ X' = [P \,;\, X] $$
where $;$ denotes concatenation along the sequence length dimension.

![Soft Prompts Architecture](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/soft_prompt_architecture.png)

The optimization objective becomes finding the optimal prompt $P$ that minimizes the task loss $\mathcal{L}$:
$$ \min_P \mathcal{L}(P) $$
Crucially, while this optimization occurs, all internal model weights remain fixed. The model learns to solve the new task entirely through the signals encoded in the prompt embeddings.

---

## Why Do Soft Prompts Work?

Soft prompts function as learned, continuous control signals. Instead of changing the network's internal processing rules (its weights), they modify the information entering the network.

As these continuous vectors pass through the transformer layers, they heavily influence the self-attention distributions, intermediate activations, feature extraction, and ultimately the output generation. Even though the transformer weights never change, the model's behavior can change dramatically based entirely on the geometry of the soft prompt vectors prepended to the context window.

This makes Soft Prompt Tuning one of the most parameter-efficient adaptation methods available. While Full Fine-Tuning updates billions of parameters, and Adapters or LoRA update millions, Soft Prompt Tuning updates only a few thousand parameters (just the embeddings for a few virtual tokens).

---

## Are Soft Prompts Hidden Sentences?

A very common misconception is that a learned soft prompt is simply a hidden English instruction that the model has discovered through gradient descent. 

For example, one might imagine that after training, the optimal sequence of prompt vectors secretly encodes the phrase: *"Classify sentiment as positive or negative."*

This intuition is appealing but usually incorrect. Soft prompts are optimized purely for downstream task performance, not for human interpretability. Because they are allowed to move freely in a high-dimensional continuous embedding space, they are absolutely not constrained to represent valid natural language.

---

## Can We Convert Soft Prompts Back Into Words?

Since prompt vectors live in the same embedding space as the model's standard token embeddings, we can attempt to recover nearby tokens to see what the prompt "means."

Suppose $E$ is the embedding matrix for the model's vocabulary. For a learned prompt vector $p$, we can find the nearest vocabulary token using cosine similarity:
$$ \arg\max_t \text{cosine}(p, E_t) $$

This process is often called **Prompt Interpretation** or token recovery. The recovered token is simply the nearest discrete vocabulary item in the embedding space.

### Why Token Recovery Can Be Misleading

Imagine a simple embedding space where the vectors for "cat", "dog", and "car" are arranged. The learned prompt vector may lie somewhere in the empty space between several unrelated words. A nearest-neighbor search might return "cat" because it happens to be the closest discrete point, but that does not mean the prompt actually represents the semantic concept of a "cat".

The continuous prompt is likely encoding mathematical attention steering information, activation trigger patterns, and task-specific signals rather than any linguistic meaning. Therefore, finding the nearest token does not reveal the prompt's actual functional meaning.

---

## The Research Question: Initialization Strategies

This natural lack of interpretability leads to an interesting research question:

> Does initializing a soft prompt using meaningful human instructions improve the learning process?

Instead of starting the optimization from randomly initialized vectors, we can initialize the prompt using the actual embeddings derived from a human-written instruction.

For example, we could take the instruction:
*"You are an expert sentiment analysis system. Determine whether the review expresses a positive or negative opinion."*

We tokenize this instruction, retrieve its corresponding embeddings, and use those exact embeddings as the starting point for optimization. Intuitively, one might strongly expect this semantic initialization to provide a vastly better starting location on the loss landscape than completely random vectors.

---

## The Surprising Experimental Results

To test this hypothesis, a controlled experiment was performed using the instruction-tuned FLAN-T5 Small model on the IMDb Movie Review Sentiment dataset. The task was binary classification (positive or negative).

Two initialization strategies were compared using exactly 53 virtual tokens:
1. **Random Initialization:** Prompt vectors were initialized randomly.
2. **Instruction Initialization:** Prompt vectors were initialized using the embeddings of a detailed, 53-token human instruction explaining the sentiment task.

Before training, the instruction-initialized prompt naturally mapped back perfectly to the original English instruction tokens. The random prompt mapped back to completely unrelated, nonsensical vocabulary items (e.g., "dive", "cloth", "english", "tourism"). This confirmed the two strategies started from entirely different regions of the embedding space.

### Validation Loss Over Time

After training for five epochs, the results were highly counterintuitive. 

![Validation Loss Plot](https://raw.githubusercontent.com/ParitKansal/Articles/main/images/validation_loss_plot.png)

Surprisingly, the random initialization consistently outperformed the instruction-based initialization at every epoch, converging to a significantly lower validation loss.

---

## An Unexpected Discovery in the Embeddings

The most interesting result appeared when examining the recovered prompt tokens *after* training was complete.

For the **Random Prompt**, the nearest-neighbor tokens changed drastically. The prompt vectors moved substantially through the embedding space to find an optimal configuration for the task.

For the **Instruction Prompt**, however, the recovered tokens remained almost completely unchanged. The vectors stayed extremely close to the original English instruction embeddings. 

At first glance, one might expect meaningful instructions to provide a superior starting point. However, the experiment suggests something different entirely. The instruction prompt appears to get "stuck" near an instruction-following region of the embedding space. The random prompt, unburdened by human semantics, is free to explore a much larger, more optimal portion of the loss landscape.

---

## Conclusion and Key Insights

Soft Prompt Tuning demonstrates that powerful task adaptation can be achieved without modifying a single weight inside a model. By learning only a tiny collection of trainable continuous embeddings, models can acquire complex new behaviors while remaining computationally and storage efficient.

The experiments presented in this article reveal several fascinating insights:

1. **Soft Prompts Are Not Hidden English Instructions:** Although prompt vectors can be mathematically mapped to nearby words, their purpose is to steer model behavior, not to represent human language.
2. **Nearest-Token Interpretation Has Strict Limits:** Recovering nearby tokens provides an interesting intuition but does not reveal the true functional mechanism of the prompt.
3. **Good Instructions $\neq$ Good Soft Prompts:** Initializing a soft prompt with a meaningful human instruction does not guarantee better convergence. In fact, random initialization can achieve substantially better performance.
4. **Optimization Geometry Trumps Semantics:** Effective soft prompt representations are governed far more by mathematical optimization geometry than by human semantic meaning. 

Understanding the nature of these learned continuous control vectors remains one of the most fascinating and active open research questions in parameter-efficient fine-tuning today.
# Advanced Machine Learning Interview Questions for FAANG & OpenAI

This document contains intensive machine learning interview questions frequently asked at top tech companies like Facebook/Meta, Apple, Amazon, Netflix, Google, and OpenAI, with detailed explanations and visual aids.

## Table of Contents

1. [Deep Learning Fundamentals](#deep-learning-fundamentals)
2. [Natural Language Processing](#natural-language-processing)
3. [Computer Vision](#computer-vision)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Large Language Models](#large-language-models)
6. [ML System Design](#ml-system-design)
7. [ML Optimization](#ml-optimization)
8. [Feature Engineering](#feature-engineering)
9. [ML Ethics & Bias](#ml-ethics-and-bias)
10. [Coding Challenges](#coding-challenges)

---

## Deep Learning Fundamentals

### Q: Explain the vanishing/exploding gradient problem in deep neural networks and methods to address it.

**Answer:**

The vanishing and exploding gradient problems are numerical instability issues that occur during training of very deep neural networks through backpropagation.

**Vanishing Gradients:**
- When gradients become extremely small as they're propagated back through layers
- Earlier layers train very slowly or not at all
- Network can't learn long-range dependencies
- Commonly occurs with sigmoid and tanh activation functions

**Exploding Gradients:**
- When gradients become extremely large
- Training becomes unstable with dramatic weight updates
- Often leads to NaN values and training failure
- More common in recurrent neural networks with long sequences

**Mathematical Explanation:**

In backpropagation, we update weights using the chain rule. For a deep network with L layers, the gradient with respect to weights in earlier layers involves multiplying many derivatives:

```
∂L/∂w₁ = ∂L/∂a_L × ∂a_L/∂a_{L-1} × ... × ∂a_2/∂z_2 × ∂z_2/∂a_1 × ∂a_1/∂w₁
```

If the derivatives are consistently < 1, their product approaches 0 exponentially (vanishing).
If the derivatives are consistently > 1, their product grows exponentially (exploding).

**Solutions to Vanishing Gradients:**

1. **Better Activation Functions:**
   - ReLU: f(x) = max(0, x)
   - Leaky ReLU: f(x) = max(αx, x) where α is small (e.g., 0.01)
   - ELU, SELU, etc.

2. **Proper Weight Initialization:**
   - Xavier/Glorot initialization for tanh: weights ~ Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
   - He initialization for ReLU: weights ~ Normal(0, √(2/fan_in))

3. **Batch Normalization:**
   - Normalizes layer inputs for each mini-batch
   - Helps maintain activations in regions with strong gradients

4. **Residual Connections (Skip Connections):**
   - Create shortcuts that bypass one or more layers
   - Help gradients flow backward through the network

5. **LSTM/GRU for Sequential Data:**
   - Special gating mechanisms to control information flow
   - Designed specifically to address vanishing gradients in recurrent networks

**Solutions to Exploding Gradients:**

1. **Gradient Clipping:**
   - If ||g|| > threshold, g = (threshold/||g||) × g
   - Preserves direction but limits magnitude

2. **Weight Regularization:**
   - L2 regularization penalizes large weights

3. **Proper Learning Rate:**
   - Smaller learning rates
   - Learning rate scheduling

**Diagram:**

```
# Vanishing Gradient Problem

[Input] → [Hidden Layer 1] → [Hidden Layer 2] → ... → [Hidden Layer n] → [Output]
                                                      ↑
                                Strong gradient   —————
                                                 ↓
[Input] ← [Hidden Layer 1] ← [Hidden Layer 2] ← ... ← [Hidden Layer n] ← [Output]
   ↑
Very weak
gradient
```

**Code Example - Addressing Vanishing Gradients with ResNet-style Skip Connections:**

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out
```

### Q: Describe the transformer architecture in detail and why it has become so dominant in NLP and beyond.

**Answer:**

The Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), has become the foundation for most state-of-the-art models in NLP and is increasingly applied to other domains like computer vision, audio processing, and multimodal learning.

**Key Components:**

1. **Self-Attention Mechanism**
   - Computes attention scores between all positions in a sequence
   - Allows the model to focus on relevant parts of the input regardless of distance
   - Overcomes the sequential limitation of RNNs through parallelization

2. **Multi-Head Attention**
   - Runs multiple attention mechanisms in parallel
   - Each "head" can focus on different aspects of the input
   - Outputs are concatenated and linearly transformed

3. **Positional Encoding**
   - Since attention has no inherent sense of order
   - Adds positional information to token embeddings
   - Usually implemented as sine/cosine functions of different frequencies

4. **Layer Normalization and Residual Connections**
   - Helps with training stability and gradient flow
   - Applied after each sub-layer (attention and feed-forward)

5. **Position-wise Feed-Forward Networks**
   - Applied to each position independently
   - Typically consists of two linear transformations with ReLU in between
   - Adds non-linearity and increases model capacity

**Architecture Details:**

The Transformer consists of an encoder and decoder, each with multiple identical layers:

**Encoder Layer:**
1. Multi-Head Self-Attention
2. Add & Normalize (residual connection + layer norm)
3. Position-wise Feed-Forward Network
4. Add & Normalize

**Decoder Layer:**
1. Masked Multi-Head Self-Attention (prevents attending to future positions)
2. Add & Normalize
3. Multi-Head Attention over encoder output
4. Add & Normalize
5. Position-wise Feed-Forward Network
6. Add & Normalize

**Mathematical Formulation:**

For self-attention with input X:

1. Compute query, key, value projections:
   - Q = XWq, K = XWk, V = XWv where Wq, Wk, Wv are learned parameter matrices

2. Compute attention scores:
   - Attention(Q,K,V) = softmax(QK^T / √d_k)V
   - Where d_k is the dimension of the keys (scaling factor)

3. For multi-head attention with h heads:
   - MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
   - where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

**Advantages Over RNNs/LSTMs:**

1. **Parallelization:**
   - No sequential dependency in computation
   - Much faster training on modern hardware (GPUs/TPUs)

2. **Long-Range Dependencies:**
   - Direct connections between any positions
   - No information bottleneck as in RNNs

3. **Interpretability:**
   - Attention weights provide insights into which parts of input influence predictions

4. **Scalability:**
   - Can process longer sequences more effectively
   - Can be scaled to enormous parameter counts (GPT-3, PaLM, etc.)

**Diagram:**
```
Transformer Architecture:

Input Embeddings + Positional Encoding
            ↓
┌───────────────────────┐
│   Encoder (N layers)  │
│ ┌───────────────────┐ │
│ │ Self-Attention    │ │
│ ├───────────────────┤ │
│ │ Add & Normalize   │ │
│ ├───────────────────┤ │
│ │ Feed Forward      │ │
│ ├───────────────────┤ │
│ │ Add & Normalize   │ │
│ └───────────────────┘ │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│   Decoder (N layers)  │
│ ┌───────────────────┐ │
│ │ Masked Self-Attn  │ │←──── Output Embeddings
│ ├───────────────────┤ │      + Positional Encoding
│ │ Add & Normalize   │ │
│ ├───────────────────┤ │
│ │ Cross-Attention   │ │
│ ├───────────────────┤ │
│ │ Add & Normalize   │ │
│ ├───────────────────┤ │
│ │ Feed Forward      │ │
│ ├───────────────────┤ │
│ │ Add & Normalize   │ │
│ └───────────────────┘ │
└───────────┬───────────┘
            ↓
        Linear Layer
            ↓
        Softmax
            ↓
         Output
```

**Applications Beyond NLP:**

- **Computer Vision**: Vision Transformer (ViT), DETR, Swin Transformer
- **Audio Processing**: Audio Spectrogram Transformer, Music Transformer
- **Multimodal Learning**: DALL-E, CLIP, Flamingo
- **Reinforcement Learning**: Decision Transformer
- **Biology**: AlphaFold for protein structure prediction
- **Time Series Analysis**: Temporal Fusion Transformers

**Recent Developments:**
- **Efficient Attention Mechanisms**: Linear attention, sparse attention, etc.
- **Parameter Sharing**: Universal Transformers, ALBERT
- **Scale**: GPT-4, PaLM, LLaMA, Claude, etc.

---

## Natural Language Processing

### Q: Explain how BERT's pre-training objectives (MLM and NSP) work and how they differ from GPT's objective. What are the trade-offs?

**Answer:**

BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer) represent two foundational approaches to pre-training language models, with key differences in their objectives and architectural designs.

**BERT's Pre-training Objectives:**

1. **Masked Language Modeling (MLM)**
   - Randomly mask 15% of tokens in each sequence
   - Of these, 80% are replaced with [MASK], 10% with random words, 10% unchanged
   - The model must predict the original tokens based on surrounding context
   - This forces bidirectional learning, as the model must use both left and right context
   - Mathematical objective: Maximize P(x_masked | x_unmasked)

2. **Next Sentence Prediction (NSP)**
   - Binary classification task: Is sentence B the actual next sentence after A?
   - Positive examples: Consecutive sentences from corpus
   - Negative examples: Sentence B randomly sampled from different document
   - Input format: [CLS] Sentence A [SEP] Sentence B [SEP]
   - Helps model learn document-level relationships and discourse understanding
   - Mathematical objective: Maximize P(IsNext | Sentence A, Sentence B)

**GPT's Pre-training Objective:**

1. **Autoregressive Language Modeling**
   - Predict each token based only on previous tokens (left context only)
   - Input: x₁, x₂, ..., xₜ, output: predictions for x₂, x₃, ..., xₜ₊₁
   - Uses causal self-attention (masked to prevent seeing future tokens)
   - Mathematical objective: Maximize Σᵢ log P(xᵢ | x₁, x₂, ..., xᵢ₋₁)

**Key Differences:**

| Aspect | BERT | GPT |
|--------|------|-----|
| **Directionality** | Bidirectional (sees both left and right context) | Unidirectional/autoregressive (only sees left context) |
| **Architecture** | Encoder only | Decoder only |
| **Training objective** | MLM + NSP | Next token prediction |
| **Input representation** | [CLS] + token sequence + [SEP] | Token sequence |
| **Primary use case** | Understanding tasks (classification, NER, etc.) | Generation tasks (text completion, summarization, etc.) |

**Trade-offs:**

1. **Understanding vs. Generation**
   - BERT excels at understanding tasks due to bidirectional context
   - GPT excels at generation tasks due to its autoregressive nature
   - BERT cannot be directly used for open-ended generation
   - GPT may be less effective for tasks requiring deep bidirectional understanding

2. **Pre-training Efficiency**
   - BERT predicts only 15% of tokens in pre-training
   - GPT predicts every token, potentially using compute more efficiently
   - But GPT only learns from left context, potentially limiting its understanding

3. **Fine-tuning Approach**
   - BERT: Usually add task-specific layers on top of [CLS] token or token embeddings
   - GPT: Usually fine-tuned in a completion format, framing tasks as text generation

4. **Context Utilization**
   - BERT models the joint probability P(x₁, x₂, ..., xₙ) directly
   - GPT models conditional probabilities P(xₙ | x₁, x₂, ..., xₙ₋₁)
   - BERT better utilizes full context for understanding individual tokens
   - GPT's approach is more aligned with text generation

**Historical Impact:**

- BERT led to significant advances in NLU benchmarks (GLUE, SQuAD)
- GPT approach scaled to enormous models (GPT-3, GPT-4) with emergent capabilities
- Many recent models combine aspects of both or introduce new objectives:
  - RoBERTa dropped NSP and showed improved performance
  - XLNet used permutation language modeling to get bidirectionality while preserving autoregressive property
  - T5 converted all NLP tasks to a text-to-text format
  - BART combined bidirectional encoder with autoregressive decoder

**Code Example - BERT's MLM Implementation:**

```python
def create_masked_lm_predictions(tokens, vocab, mask_prob=0.15):
    """Create masked tokens for MLM training."""
    mask_positions = []
    for i in range(len(tokens)):
        # Skip special tokens
        if tokens[i] in ["[CLS]", "[SEP]"]:
            continue
        
        # Random sampling with 15% probability
        if random.random() < mask_prob:
            mask_positions.append(i)
            
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                tokens[i] = "[MASK]"
            # 10% of the time, replace with a random word
            elif random.random() < 0.5:
                tokens[i] = random.choice(list(vocab.keys()))
            # 10% of the time, keep the original word
    
    return tokens, mask_positions
```

**Diagram:**
```
BERT vs GPT Pre-training:

BERT:
"The cat sat on the [MASK]"
       ↓
Predict: "mat"
(Using both left AND right context)

GPT:
"The cat sat on the"
       ↓
Predict: "mat"
(Using ONLY left context)
```

### Q: How does beam search work for decoding in sequence generation tasks, and what are its limitations compared to other decoding strategies?

**Answer:**

Beam search is a heuristic search algorithm commonly used for decoding in sequence generation tasks like machine translation, text summarization, and image captioning. It attempts to balance computational efficiency with output quality by maintaining a limited set of partially-decoded sequences at each step.

**Basic Algorithm:**

1. Start with an initial state (typically a start token)
2. At each step:
   - For each sequence in the beam, compute probabilities for all possible next tokens
   - From the entire set of (beam_size × vocabulary_size) possible extensions, keep only the top-k sequences with highest cumulative probability
   - k is the beam width/size (typically 5-10)
3. Continue until all sequences in the beam have generated an end token or reached maximum length
4. Return the sequence with the highest probability (or potentially the top-n sequences)

**Mathematical Formulation:**

For a sequence of tokens (y₁, y₂, ..., yₜ), the probability of the sequence is:

P(y₁, y₂, ..., yₜ) = Π₁ᵗ P(yᵢ | y₁, y₂, ..., yᵢ₋₁)

To avoid numerical underflow with long sequences, we typically work with log probabilities:

log P(y₁, y₂, ..., yₜ) = Σ₁ᵗ log P(yᵢ | y₁, y₂, ..., yᵢ₋₁)

Beam search keeps the top-k sequences based on these cumulative log probabilities.

**Length Normalization:**

Beam search tends to favor shorter sequences because each additional token multiplication reduces probability. To counteract this:

Score(y₁, y₂, ..., yₜ) = log P(y₁, y₂, ..., yₜ) / length_penalty(t)

where length_penalty(t) = (5 + t)ᵅ / (5 + 1)ᵅ with α typically around 0.6-0.7.

**Advantages of Beam Search:**

1. More thorough exploration than greedy search (which only takes the highest probability token at each step)
2. Computationally tractable compared to exhaustive search
3. Generally produces better results than greedy decoding
4. Deterministic output (same input always produces same output)

**Limitations and Challenges:**

1. **Lack of Diversity:**
   - Tends to generate similar sequences that differ only in a few tokens
   - The top beams often share the same prefix
   - Can lead to generic, "safe" outputs

2. **Search Errors:**
   - Optimal sequence may fall outside the beam
   - Higher beam size doesn't always lead to better results (diminishing returns)
   - Can even hurt performance in some neural sequence models (beam search curse)

3. **Exposure Bias:**
   - Training uses teacher forcing while inference uses previous predictions
   - This mismatch can compound errors during generation

4. **Not Suitable for Open-ended Generation:**
   - Works well for tasks with well-defined outputs
   - Less appropriate for creative or open-ended text generation

**Alternative Decoding Strategies:**

1. **Greedy Search:**
   - Always select the most probable next token
   - Fast but often suboptimal
   - Special case of beam search with beam size = 1

2. **Exhaustive Search:**
   - Consider all possible sequences (intractable for long sequences)

3. **Sampling-based Methods:**
   - Pure sampling: Sample from the predicted distribution
   - Top-k sampling: Sample from the k most likely tokens
   - Top-p (nucleus) sampling: Sample from the smallest set of tokens whose cumulative probability exceeds p
   - Better for creative and diverse text generation

4. **Diverse Beam Search:**
   - Explicitly encourages diversity across beams
   - Uses penalty terms to discourage similar beams

5. **Constrained Beam Search:**
   - Incorporates constraints that generated text must satisfy

**Comparison of Decoding Strategies:**

| Decoding Strategy | Diversity | Quality | Speed | Use Cases |
|-------------------|-----------|---------|-------|-----------|
| Greedy            | Low       | Medium  | Fast  | Simple translation, structured output |
| Beam Search       | Low       | High    | Medium| Translation, summarization |
| Top-k Sampling    | Medium    | Medium  | Fast  | Creative text generation |
| Nucleus Sampling  | High      | Medium  | Fast  | Open-ended text, storytelling |
| Pure Sampling     | Very high | Low     | Fast  | Maximum diversity, creative tasks |

**Pseudocode for Beam Search:**

```
function BeamSearch(model, start_token, beam_size, max_length):
    # Initialize with start token
    beam = [(start_token, 0)]  # (sequence, log_probability)
    
    for i from 1 to max_length:
        all_candidates = []
        
        # Expand each sequence in the current beam
        for sequence, score in beam:
            # Check if sequence is complete
            if last_token(sequence) == end_token:
                all_candidates.append((sequence, score))
                continue
            
            # Get predictions for next token
            next_token_log_probs = model.predict_next_token(sequence)
            
            # Add all possible next tokens to candidates
            for token, log_prob in next_token_log_probs:
                new_sequence = sequence + token
                new_score = score + log_prob
                all_candidates.append((new_sequence, new_score))
        
        # Filter to keep only top-k candidates
        beam = select_top_k(all_candidates, k=beam_size)
        
        # Check if all sequences have ended
        if all(last_token(sequence) == end_token for sequence, _ in beam):
            break
    
    # Return the highest scoring sequence
    return max(beam, key=lambda x: x[1])[0]
```

**Diagram:**
```
Beam Search (beam_size=3) Example:

Step 1:
[START] → ["I": -0.1] ["The": -0.3] ["A": -0.5]
          (Keep top 3)

Step 2:
["I"] → ["I am": -0.5] ["I have": -0.6] ["I will": -0.8]
["The"] → ["The cat": -0.7] ["The dog": -0.8] ["The man": -0.9]
["A"] → ["A new": -0.8] ["A big": -0.9] ["A small": -1.0]
        (Sort all 9 candidates, keep top 3)
        ["I am": -0.5] ["I have": -0.6] ["The cat": -0.7]

Step 3:
["I am"] → ["I am a": -0.9] ["I am the": -1.0] ["I am going": -1.1]
["I have"] → ["I have a": -1.0] ["I have been": -1.1] ["I have the": -1.2]
["The cat"] → ["The cat is": -1.0] ["The cat was": -1.1] ["The cat has": -1.3]
              (Sort all 9 candidates, keep top 3)
              ["I am a": -0.9] ["I have a": -1.0] ["The cat is": -1.0]

...and so on
```

---

## Large Language Models

### Q: How does Reinforcement Learning from Human Feedback (RLHF) work in training LLMs like ChatGPT, and what are its challenges?

**Answer:**

Reinforcement Learning from Human Feedback (RLHF) has become a critical technique for aligning large language models (LLMs) with human preferences, values, and instructions. It helps address limitations of purely supervised fine-tuning by incorporating human feedback into the training process.

**RLHF Process for LLMs (3 Main Stages):**

1. **Supervised Fine-Tuning (SFT)**
   - Start with a pre-trained language model
   - Fine-tune on a dataset of demonstrations of desired behavior
   - Usually involves prompt-response pairs written by humans
   - Creates a baseline model that can follow instructions

2. **Reward Model Training**
   - Collect comparison data: humans rank different model outputs for the same prompt
   - Train a reward model to predict which response humans would prefer
   - Typically uses a Bradley-Terry preference model: P(A preferred to B) = σ(r(A) - r(B))
   - The reward model learns to assign higher scores to responses that humans would prefer

3. **Reinforcement Learning Optimization**
   - Use the reward model as a reward function
   - Fine-tune the SFT model using RL (typically with PPO - Proximal Policy Optimization)
   - Optimize expected reward while staying close to the original SFT model
   - Add a KL divergence penalty to prevent excessive deviation from the SFT model

**Mathematical Formulation:**

For the RL optimization phase, the objective typically looks like:

$$\max_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x, y) - \beta \log(\pi_{\theta}(y|x) / \pi_{\text{SFT}}(y|x))]$$

Where:
- $\pi_{\theta}$ is the policy (LLM) being trained
- $\pi_{\text{SFT}}$ is the supervised fine-tuned model
- $r_{\phi}$ is the learned reward model
- $\beta$ is the KL penalty coefficient
- $x$ is a prompt
- $y$ is a response

**Key Components in Detail:**

1. **Reward Modeling Details:**
   - Input: Prompt x and response y pair
   - Output: Scalar reward r(x,y)
   - Architecture: Usually the same as LLM with a value head on top
   - Loss function: $L = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log(\sigma(r(x,y_w) - r(x,y_l)))]$
      - Where $y_w$ and $y_l$ are the preferred and less preferred responses

2. **PPO Implementation for LLMs:**
   - Value function estimates expected future rewards
   - Policy updates are clipped to prevent too large changes
   - Multiple epochs of optimization on collected rollouts
   - Custom adaptations for text generation (vs. traditional RL environments)

3. **Constitutional AI / Red-Teaming:**
   - Often combined with RLHF for additional safety
   - Generate harmful/problematic responses, then train model to avoid them
   - Helps address "blind spots" in the human feedback data

**Challenges and Limitations:**

1. **Reward Hacking/Gaming:**
   - Models can learn to exploit flaws in the reward model
   - May optimize for superficial features that correlate with high reward
   - E.g., excessive verbosity, overly cautious responses, false citations

2. **Reward Model Quality:**
   - Human feedback is noisy and subjective
   - Limited coverage of possible inputs
   - Costly and time-consuming to collect
   - Cultural and demographic biases in annotator preferences

3. **KL Penalty Tuning:**
   - Too low: Model deviates significantly from pre-trained capabilities
   - Too high: Insufficient alignment improvements
   - Finding right balance is challenging

4. **Specification Gaming:**
   - Systems find unintended ways to maximize reward
   - May not reflect true human preferences
   - Can lead to unexpected behaviors

5. **Scaling Challenges:**
   - Human feedback doesn't scale with model size
   - Expensive to apply to frontier models
   - Ensuring consistency across diverse tasks

6. **Preference Inconsistency:**
   - Human preferences often contradict each other
   - Different humans have different values
   - Challenging to optimize for diverse preferences

7. **Exploration in RL:**
   - Language models have enormous action spaces
   - Difficult for RL algorithms to effectively explore
   - May get stuck in local optima

**Recent Developments and Alternatives:**

1. **Direct Preference Optimization (DPO):**
   - Skips explicit reward modeling and PPO
   - Directly optimizes policy from preference data
   - Often more stable and simpler to implement

2. **Constitutional AI (CAI):**
   - Model critiques its own outputs based on constitutional principles
   - Reduces dependency on human feedback
   - Can be combined with RLHF

3. **Self-Supervised Alignment:**
   - Models like Claude use more extensive SFT with less emphasis on RLHF
   - May be more compute-efficient for some alignment goals

4. **Iterative RLHF:**
   - Use multiple rounds of RLHF, collecting new human feedback after each round
   - Addresses more complex alignment issues that emerge

**Diagram:**
```
RLHF Process Flow:

┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Pre-trained │      │  Supervised │      │   Reward    │
│    LLM      │─────▶│ Fine-Tuning │─────▶│   Model     │
└─────────────┘      └─────────────┘      │  Training   │
                           │              └─────┬───────┘
                           │                    │
                           ▼                    ▼
                     ┌─────────────┐      ┌─────────────┐
                     │   Initial   │      │   Reward    │
                     │ Policy (SFT)│      │    Model    │
                     └──────┬──────┘      └─────┬───────┘
                            │                   │
                            └───────┬───────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │      RL       │
                            │ Optimization  │─────────────┐
                            │     (PPO)     │             │
                            └───────────────┘             │
                                    │                     │
                                    ▼                     │
                            ┌───────────────┐             │
                            │   Aligned     │             │
                            │     LLM       │◀────────────┘
                            └───────────────┘      KL Divergence
                                                   Constraint
```

**Code Example - Simplified PPO for LLM:**

```python
def ppo_train_step(policy_model, value_model, reward_model, sft_model, prompts, ppo_epochs, batch_size, lr):
    # Generate responses with current policy
    responses = policy_model.generate(prompts)
    
    # Compute rewards
    rewards = reward_model(prompts, responses)
    
    # Compute KL divergence from SFT model
    log_probs = policy_model.log_prob(responses, prompts)
    sft_log_probs = sft_model.log_prob(responses, prompts)
    kl_div = log_probs - sft_log_probs
    
    # Compute advantages (simplified)
    values = value_model(prompts, responses)
    advantages = rewards - values
    
    # PPO optimization loop
    for _ in range(ppo_epochs):
        # Sample mini-batches
        batch_indices = sample_batch_indices(len(prompts), batch_size)
        
        for indices in batch_indices:
            batch_prompts = [prompts[i] for i in indices]
            batch_responses = [responses[i] for i in indices]
            batch_advantages = advantages[indices]
            batch

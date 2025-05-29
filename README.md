
---

# README.md

## XenArcAIi2: A High-Performance Language Model with Dynamic Attention

**XenArcAIi2** is a state-of-the-art language model designed to compete with models like **Gemma 3 27B**, offering superior performance in reasoning, coding, and long-context tasks (up to 1M tokens). With ~50 billion parameters, a Mixture-of-Experts (MoE) architecture, and a fully dynamic attention mechanism, XenArcAIi2 is optimized for TPU training on a 10T-token dataset, including synthetic data from DeepSeek R-1 and Qwen 2.5. It uses the Qwen 3 tokenizer (151K vocabulary) for multilingual and technical efficiency, and its attention system—combining **Hierarchical Long-Sequence Attention (HLSA)** and multi-resolution attention—adapts dynamically to input needs using reinforcement learning (RL) with Proximal Policy Optimization (PPO).

This README provides an overview of the model, its architecture, attention mechanisms, training setup, and usage instructions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Attention Mechanisms](#attention-mechanisms)
   - [Hierarchical Long-Sequence Attention (HLSA)](#hierarchical-long-sequence-attention-hlsa)
   - [Multi-Resolution Attention](#multi-resolution-attention)
   - [Dynamic Attention with RL PPO](#dynamic-attention-with-rl-ppo)
4. [Self-Assessment and Optimization](#self-assessment-and-optimization)
5. [Training on TPUs](#training-on-tpus)
6. [Setup and Installation](#setup-and-installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

**XenArcAIi2** is designed for researchers and developers seeking a scalable, efficient language model for tasks requiring long-context understanding, multilingual reasoning, and low hallucination. Key features include:

- **Scale**: ~50B parameters, 56 transformer layers, MoE with 16 experts (2–8 active per token).
- **Context Length**: Supports up to 1M tokens, ideal for extended documents, codebases, or conversations.
- **Tokenizer**: Qwen 3 with 151K vocabulary, fine-tuned for multilingual and technical data.
- **Attention**: Combines HLSA (sparse, hierarchical) with multi-resolution attention, dynamically controlled by RL PPO to optimize efficiency and performance.
- **Training**: TPU-optimized, using sharded safetensors (8–12 shards, ~10GB each), designed for a 10T-token dataset.
- **Self-Assessment**: Continuously evaluates and adjusts attention, expert routing, and reasoning to minimize loss, maximize accuracy, and reduce hallucination.
- **Output**: Generates text, supports multimodal inputs (e.g., text + embeddings), and provides detailed performance metrics via TensorBoard.

The model is built to leverage a 10T-token dataset, including synthetic data from DeepSeek R-1 and Qwen 2.5, ensuring robustness across languages, technical domains, and reasoning tasks. It’s TPU-focused, using `torch_xla` for distributed training, and outputs sharded safetensors for scalability.

---

## Model Architecture

XenArcAIi2 is a transformer-based model with the following components:

- **Embedding Layer**: Maps 151K tokens to a 8192-dimensional space using the Qwen 3 tokenizer.
- **Transformer Layers**: 56 layers, each containing:
  - **Dynamic Attention**: HLSA and multi-resolution attention, controlled by RL PPO (detailed below).
  - **Mixture-of-Experts (MoE)**: 16 experts per layer, with 2–8 active experts per token, routed via a structured gate with hypernetwork adjustments.
  - **Normalization**: RMSNorm with epsilon 1e-5 for stability.
  - **Skip Gates**: Learnable gates to bypass layers, targeting a 50% skip rate for efficiency.
- **Planning Module**: Generates per-layer strategies to guide attention and MoE routing, using a lightweight MLP.
- **Summarizer**: Compresses context for efficient reasoning, using a down-projection MLP.
- **Self-Assessment Module**: Evaluates performance (loss, accuracy, hallucination, attention efficiency) and adjusts attention/expert parameters via RL PPO.
- **Output Layer**: Projects to 151K vocabulary for token prediction.

The model uses bfloat16 precision for TPU compatibility, with a total of ~50B parameters. It supports multimodal inputs (e.g., text + embeddings) via dynamic linear projections.

---

## Attention Mechanisms

XenArcAIi2’s attention system is a hybrid of **Hierarchical Long-Sequence Attention (HLSA)** and **multi-resolution attention**, fully dynamic and controlled by RL PPO. This allows the model to adapt attention patterns to each input, balancing efficiency and performance for 1M-token contexts.

### Hierarchical Long-Sequence Attention (HLSA)

HLSA is designed for long sequences, reducing computational complexity from O(n²) to near-linear by using sparse attention patterns. It employs multiple attention paths, dynamically selected per layer and input:

1. **Sparse Local Attention**:
   - Focuses on a local window of tokens (default baseline: 8192 tokens).
   - Uses a sliding window with dilation (default baseline: 2) to capture nearby context efficiently.
   - Example: For a query token at position 1000, it attends to tokens 992–1008, skipping every other token if dilation is 2.
   - Benefit: Reduces memory and compute, ideal for short-range dependencies.

2. **Global Token Attention**:
   - Selects a subset of “global” tokens (default baseline: 512) at regular intervals (e.g., every 16384 tokens) or adaptively via an MLP selector.
   - These tokens act as landmarks, summarizing distant context.
   - Example: For a 1M-token sequence, 512 global tokens provide a compressed view of the entire context.
   - Benefit: Captures long-range dependencies with minimal overhead.

3. **Attention Sinks**:
   - Reserves the first 16 tokens as “sinks” that always receive attention.
   - Ensures critical initial context (e.g., prompts, instructions) is preserved.
   - Benefit: Stabilizes reasoning for tasks requiring strong anchoring to initial tokens.

4. **Recurrent State Attention**:
   - Maintains a 512-dimensional recurrent state, updated per layer via a gated MLP.
   - Summarizes past context, allowing the model to “remember” beyond the current window.
   - Example: The recurrent state aggregates mean embeddings, updated with a sigmoid gate to balance new and old information.
   - Benefit: Enhances coherence in long sequences without full attention.

5. **KV Compression**:
   - Compresses key-value pairs to a 64-dimensional rank before caching, then decompresses during attention.
   - Reduces memory footprint for 1M-token contexts.
   - Benefit: Enables scalable caching on TPUs.

HLSA uses **YaRN scaling** (Yet Another RoPE Scaling) to extend Rotary Position Embeddings (RoPE) for 1M-token contexts, with a base theta of 1e6 and beta parameters (fast: 32, slow: 1). This ensures positional encodings remain effective over long sequences.

### Multi-Resolution Attention

Multi-resolution attention complements HLSA by dynamically adjusting the granularity of context processing:

- **Chunk-Based Processing**: Divides the sequence into chunks (default: 1024 tokens) for coarse-grained attention, then refines within chunks.
- **Variable Window Sizes**: Allows different layers to use different window sizes (e.g., 1024 for early layers, 8192 for later layers), controlled by RL PPO.
- **Hierarchical Focus**: Early layers focus on broad context (large windows, more global tokens), while later layers refine local details (smaller windows, fewer global tokens).
- Benefit: Balances global and local dependencies, reducing compute for irrelevant tokens.

### Dynamic Attention with RL PPO

Unlike traditional models with hardcoded attention parameters (e.g., fixed window sizes or global token counts), XenArcAIi2 uses **RL PPO** to dynamically select all attention parameters per layer and input. This ensures the model adapts to the task’s needs, optimizing for performance and efficiency.

- **Parameters Controlled**:
  - **Global Tokens**: Number of global tokens (baseline: 512, max: ~1000).
  - **Window Size**: Local attention window size (baseline: 8192, max: ~16K).
  - **Latent Dimension**: Dimension for latent attention (baseline: 2048, max: 4096).
  - **Dilation**: Step size for sparse local attention (baseline: 2, max: 4).
  - **Sparsity**: Fraction of attended tokens (baseline: 0.1, max: 0.5).
- **How It Works**:
  - The RL PPO module takes the model’s state (mean embeddings, gate scores, embedding variance) as input.
  - It outputs a policy (via a 3-layer MLP) predicting the above parameters, sampled from a normal distribution.
  - The value network estimates the expected reward, guiding policy updates.
  - Rewards are computed based on:
    - **Loss**: Cross-entropy loss on predictions.
    - **Accuracy**: Token prediction accuracy.
    - **Hallucination**: Cosine distance between embeddings and target (lower is better).
    - **Attention Efficiency**: Inverse of attended tokens (higher sparsity = better).
  - PPO updates ensure exploration (via entropy regularization) while maintaining stability (clipped updates).
- **Baseline Enforcement**: Parameters are clamped to minimum values (e.g., 512 global tokens, 8192 window size) to prevent degradation.
- **Example**: For a coding task with dense dependencies, RL PPO might increase the window size to 16K and reduce global tokens to 256. For a narrative task, it might prioritize 1000 global tokens and a smaller 4K window.
- **Benefit**: Adapts attention to task complexity, sequence length, and context, minimizing compute while maximizing performance.

The dynamic attention system integrates HLSA’s paths (local, global, recurrent, sinks) with multi-resolution’s chunk-based focus, allowing the model to switch between paths or combine them based on RL PPO’s policy. For instance, early layers might favor global tokens for broad context, while later layers use sparse local attention for details, all adjusted in real-time.

---

## Self-Assessment and Optimization

XenArcAIi2 includes a **self-assessment module** that evaluates and optimizes its performance during training and inference:

- **Metrics Evaluated**:
  - **Loss**: Cross-entropy loss on token predictions.
  - **Accuracy**: Fraction of correctly predicted tokens.
  - **Skip Rate**: Fraction of skipped layers (target: 50%).
  - **Context Recall**: Cosine similarity between embeddings and target, measuring context retention.
  - **Hallucination Score**: Cosine distance, penalizing deviations from ground truth.
  - **Attention Efficiency**: Inverse of attended tokens, rewarding sparsity.
  - **Expert Usage Variance**: Balances MoE expert activation for efficiency.
  - **Question Type Balance**: Ensures fair expert routing across task types.
- **How It Works**:
  - A 3-layer MLP processes embeddings, gate scores, and variance to compute performance scores.
  - A causal analyzer (MLP) identifies bottlenecks (e.g., attention vs. MoE).
  - RL PPO adjusts attention parameters and expert routing based on a reward combining these metrics.
  - Historical buffers (e.g., loss, skip rates) track trends, stabilizing optimization.
- **Output**: Logs metrics to TensorBoard and adjusts model behavior (e.g., increasing global tokens if context recall is low).
- **Benefit**: Ensures the model self-improves, reducing hallucination, improving accuracy, and optimizing TPU compute.

---

## Training on TPUs

XenArcAIi2 is optimized for TPU training using `torch_xla`, designed for a 10T-token dataset with 1M-token contexts. Here’s how to train efficiently:

### Dataset
- **Size**: 10T tokens, including synthetic data from DeepSeek R-1 (reasoning-focused) and Qwen 2.5 (multilingual, technical).
- **Format**: Tokenized text, preprocessed with the Qwen 3 tokenizer (151K vocab).
- **Preprocessing**:
  - Split into 1M-token sequences (or shorter for padding).
  - Use `transformers.AutoTokenizer` to tokenize, saving to disk (e.g., HDF5 or TFRecord).
  - Balance languages and domains (e.g., 30% code, 30% multilingual, 40% synthetic reasoning).
- **Storage**: Store on high-speed storage (e.g., Google Cloud Storage) for TPU access.

### TPU Setup
- **Hardware**: Use a TPU v4 or v5 pod (128–256 cores) for ~50B parameters and 1M-token contexts.
- **Sharding**: Model weights are split into 8–12 safetensors (~10GB each), loaded across TPU cores.
- **Batch Size**: 4096 tokens per chip, adjusted based on TPU memory (e.g., 512 sequences of 8192 tokens).
- **Precision**: bfloat16 for all computations, reducing memory and speeding up training.
- **Gradient Checkpointing**: Enabled to trade compute for memory, critical for 1M-token contexts.
- **Model Parallelism**: Split layers and experts across TPU cores, using `torch_xla`’s XLA compilation.

### Training Strategy
- **Optimizer**: AdamW with weight decay (0.01), gradient clipping (0.5), and learning rate 1e-4 (with warmup and cosine decay).
- **Distributed Training**:
  - Use `xmp.spawn` for multi-process TPU training, with `pl.MpDeviceLoader` for data loading.
  - Shard dataset across TPUs, ensuring each core processes unique batches.
- **Dynamic Attention Optimization**:
  - HLSA’s multiple paths (local, global, recurrent, sinks) are selected by RL PPO, reducing compute by prioritizing sparse attention for most inputs.
  - Multi-resolution attention adapts chunk sizes (e.g., 1024–8192 tokens), guided by RL PPO to focus on relevant context.
  - RL PPO updates attention parameters every step, balancing exploration (epsilon: 0.9 to 0.1 over 2000 steps) and exploitation.
- **Checkpointing**: Save sharded safetensors every 1000 steps, with a `config.json` for metadata.
- **Monitoring**: Use TensorBoard to track loss, accuracy, hallucination, and attention efficiency, adjusting RL PPO rewards if needed.
- **Efficiency Tips**:
  - Precompile XLA graphs for common sequence lengths (e.g., 128K, 512K, 1M) to reduce startup time.
  - Use asynchronous data loading to avoid I/O bottlenecks.
  - Tune RL PPO’s entropy beta (0.01) and clip epsilon (0.2) to balance exploration and stability.
  - Monitor TPU memory usage, reducing batch size or enabling more checkpointing if OOM occurs.

### Expected Training Time
- **Hardware**: 256 TPU v4 cores.
- **Dataset**: 10T tokens, 1M-token sequences.
- **Epochs**: 1 epoch (10T tokens processed once).
- **Time**: ~1–2 months, assuming 1T tokens/day (typical for large TPU pods).
- **Cost**: ~$50K–$100K on Google Cloud TPUs, depending on discounts and uptime.

---

## Setup and Installation

### Prerequisites
- **Hardware**: TPU v4/v5 pod (128+ cores recommended), or CPU/GPU for inference.
- **OS**: Linux (Ubuntu 20.04+ recommended).
- **Storage**: 10TB+ for dataset, 100GB for model weights.
- **Software**:
  - Python 3.8+
  - PyTorch 2.0+ with `torch_xla` (install via `pip install torch-xla`)
  - Transformers (Hugging Face): `pip install transformers`
  - Safetensors: `pip install safetensors`
  - TensorBoard: `pip install tensorboard`
  - Google Cloud SDK (for TPU access)

### Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/xenarc-aii2.git
   cd xenarc-aii2
   ```
2. **Install Dependencies**:
   ```bash
   pip install torch torch-xla transformers safetensors tensorboard
   ```
3. **Setup TPU Environment**:
   - Follow Google Cloud TPU setup: https://cloud.google.com/tpu/docs/setup
   - Configure `gcloud` and authenticate.
4. **Download Tokenizer**:
   - The Qwen 3 tokenizer is downloaded automatically by `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")`.
   - Save to `./tokenizer/qwen3_tokenizer`.

### Dataset Preparation
- Download or generate your 10T-token dataset (e.g., DeepSeek R-1, Qwen 2.5 synthetic data).
- Tokenize using:
  ```bash
  python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B-Instruct'); tokenizer.save_pretrained('./tokenizer/qwen3_tokenizer')"
  ```
- Store tokenized data in a format like HDF5 or TFRecord, sharded for TPU loading.

---

## Usage

### Training
1. **Prepare Training Script**:
   - Use an external script (not included in `xenith_mini.py`) to avoid cluttering the model definition.
   - Example (`train.py`):
     ```python
     import torch_xla.core.xla_model as xm
     import torch_xla.distributed.xla_multiprocessing as xmp
     from xenith_mini import XenArcAIi2, ModelArgs, setup_logging, load_qwen3_tokenizer
     import pl

     def train_xla(rank):
         args = ModelArgs()
         logger, writer = setup_logging()
         tokenizer = load_qwen3_tokenizer("path/to/10T_dataset")
         model = XenArcAIi2(args)
         model.to(xm.xla_device())
         dataloader = pl.MpDeviceLoader(...)  # Load dataset
         for step, (inputs, target) in enumerate(dataloader):
             logits, _, loss = model(inputs, target=target, training=True, logger=logger, writer=writer)
             xm.optimizer_step(loss)
             if step % 1000 == 0:
                 model.save_weights(f"weights/step_{step}")
     xmp.spawn(train_xla)
     ```
2. **Run Training**:
   ```bash
   python train.py
   ```
3. **Monitor**:
   - View metrics in TensorBoard:
     ```bash
     tensorboard --logdir ./logs
     ```

### Inference
1. **Load Model**:
   - Use `xenith_mini.py` to load sharded weights:
     ```python
     from xenith_mini import XenArcAIi2, ModelArgs
     model = XenArcAIi2(ModelArgs(), weights_path="./weights/xenarc_weights")
     ```
2. **Generate Text**:
   ```python
   import torch
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("./tokenizer/qwen3_tokenizer")
   inputs = tokenizer("Write a Python function...", return_tensors="pt")["input_ids"]
   outputs = model(inputs)
   print(tokenizer.decode(outputs[0].argmax(dim=-1)))
   ```
3. **Multimodal Input**:
   - Pass embeddings as `multimodal_data`:
     ```python
     multimodal_data = {"image_emb": torch.randn(1, 512)}
     outputs = model(inputs, multimodal_data=multimodal_data)
     ```

### Output
- **Weights**: 8–12 sharded safetensors (~10GB each) in `./weights/xenarc_weights`.
- **Logs**: TensorBoard logs in `./logs`, including loss, accuracy, hallucination, and attention efficiency.
- **Tokenizer**: Saved in `./tokenizer/qwen3_tokenizer`.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to branch: `git push origin feature/your-feature`.
5. Open a pull request.

Please include tests and update documentation for new features.

---

## License

MIT License. See `LICENSE` for details.

---

### Final Notes
- **Performance**: XenArcAIi2 aims to match or exceed Gemma 3 27B (~67.5 MMLU-Pro) on reasoning, coding, and long-context tasks, leveraging its 1M-token context and dynamic attention.
- **Scalability**: The model’s sharding and TPU optimization make it feasible to train on large datasets like yours.
- **Customization**: Adjust RL PPO rewards or attention baselines in `ModelArgs` to prioritize specific metrics (e.g., hallucination reduction).

For questions or support, contact the maintainers or open an issue on GitHub.

---

### Explanation of Attention Path Selection
- **HLSA Paths**: The model dynamically selects among sparse local attention, global token attention, recurrent state, and attention sinks using RL PPO. For example:
  - **Coding Task**: Prioritizes sparse local attention (large window, high dilation) for precise dependency tracking.
  - **Narrative Task**: Emphasizes global tokens and recurrent state for coherence over long contexts.
  - **Instruction-Following**: Uses attention sinks to anchor on prompts, with moderate global tokens.
- **Multi-Resolution Integration**: Adjusts chunk sizes and window granularity, ensuring early layers capture broad context and later layers refine details, all guided by RL PPO.
- **Dynamic Control**: RL PPO’s policy network predicts parameters based on input embeddings and task metrics, ensuring optimal path combinations without hardcoded rules.

### Training Efficiency
- **HLSA Optimization**: Sparse paths reduce compute (e.g., attending to ~10% of tokens), critical for 1M-token contexts.
- **TPU Utilization**: Sharding, checkpointing, and bfloat16 ensure memory-efficient training.
- **Dataset Leverage**: The 10T-token dataset’s diversity (synthetic reasoning, multilingual) enhances generalization.

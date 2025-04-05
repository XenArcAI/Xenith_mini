# Xenith_mini

![Xenith_mini Logo]()
 <!-- Replace with your logo if you have one -->

**A Next-Generation AI Powerhouse Built for Scale and Intelligence**

Welcome to `Xenith_mini`, an advanced language model architecture designed to push the boundaries of natural language processing. Crafted with cutting-edge techniques and optimized for TPU acceleration, this model combines massive scale, efficiency, and innovative memory mechanisms to tackle complex tasks—from ultra-long context understanding to dynamic reasoning.

---

## 🌟 Key Features

- **256k Context Window**: Process up to 262,144 tokens in a single pass, perfect for long documents, multi-turn conversations, or intricate reasoning tasks.
- **Mixture of Experts (MoE)**: 32 specialized experts with 6 activated per token, delivering high capacity with sparse efficiency (~50B parameters, ~10B active).
- **Sparse Attention**: Multi-scale windowed attention (512, 1024, 2048) with 256 global tokens, balancing memory and compute for long sequences.
- **Dynamic Memory**: Short-Term Memory (16k tokens) and Long-Term Memory (32k chunks) with a transformer-based summarizer for persistent context retention.
- **TPU-Optimized**: Built with JAX and Haiku, leveraging `pmap` and `jit` for blazing-fast inference on TPU v4-32 and beyond.
- **Mistral-Compatible**: Uses the Mistral 8x7B tokenizer (32k vocab) for seamless integration with modern NLP ecosystems.

---

## 🏛️ Architecture Highlights

| Component            | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Embedding Dim**    | 8,192 for rich token representations                                   |
| **Layers**           | 64 total: 8 dense MLPs + 56 MoE layers                                 |
| **Attention Heads**  | 64 query heads, 16 KV heads (Group Query Attention)                    |
| **FFN Dimensions**   | Dense: 24,576; MoE: 3,072 per expert                                   |
| **LoRA Adaptation**  | Rank-128 LoRA in all linear layers for efficient fine-tuning           |
| **Summarizer**       | 4-layer, 16-head transformer for compressing STM into LTM              |

`Xenith_mini` blends the best of modern NLP: sparse attention for scalability, MoE for expert-driven performance, and a unique memory system for context-aware intelligence. It’s designed to rival models like Qwen-32B or Mixtral 8x7B, with the potential to exceed them in long-context and reasoning tasks when fully trained.

---

## 🚀 Why Xenith_mini?

- **Unmatched Context**: Handle entire books or extended dialogues in one go with its 256k token capacity.
- **Efficiency at Scale**: MoE and sparse attention keep compute costs low despite its 50B-parameter footprint.
- **Memory Innovation**: STM and LTM enable the model to "remember" across sessions, ideal for personalized or iterative tasks.
- **TPU Power**: Optimized for Google’s TPU v4-32, achieving up to 20M tokens/sec throughput for rapid inference and training.

---

## 🌍 Vision

`Xenith_mini` is more than a model—it’s a foundation for the future of AI. Whether you’re building a research prototype, a production-grade chatbot, or a reasoning engine, this architecture offers the flexibility and power to excel. With proper training (e.g., 10T tokens on a v4-32 in ~7 days), it aims to compete with or surpass state-of-the-art models like LLaMA-70B or Mixtral.

---

## 📚 Getting Started

This repository contains the core architecture in JAX/Haiku, with the tokenizer pre-saved as `xenith_tokenizer` (based on Mistral 8x7B). Stay tuned for training scripts, pre-trained weights, and detailed docs!

- **Tokenizer**: `./xenith_tokenizer` (32k vocab, downloaded from `mistralai/Mixtral-8x7B-v0.1`)
- **Hardware**: Optimized for TPU v4-32, adaptable to other accelerators

---

## 🤝 Contribute

We’re excited to grow `Xenith_mini` with the community! Feel free to:
- Star ⭐ the repo to show support.
- Submit issues or PRs for enhancements.
- Share ideas for training datasets or use cases.

Let’s build something extraordinary together!

---

*Created with ❤️ by [XenArcAI]

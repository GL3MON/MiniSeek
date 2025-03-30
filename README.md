### MiniSeek: A Lightweight DeepSeek V-3 Inspired Model ⚠️(On Progress)
*MiniSeek is a lightweight, open-source language model inspired by DeepSeek V-3, designed for efficient training and inference on consumer hardware while maintaining competitive performance.*

## Features
# Multi-Head Latent Attention (MLA)
- *Latent Space Compression:* Projects KV (Key-Value) into a smaller latent space, reducing memory usage.
# DeepSeekMoE Architecture 
- *Sparse Activation:* Only a subset of "expert" layers activate per input (e.g., 2 out of 16 experts), reducing compute costs.
- *Shared Expert for Common Knowledge:* Includes a shared expert to handle universal patterns, improving generalization.

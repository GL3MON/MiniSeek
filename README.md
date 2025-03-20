# MiniSeek
*An Pytorch implementation of an LLM inspired from Deepseek*

## Current Features

### Multi-Head Latent Attention (MLA)
- This mechanism replaces traditional Multi-Head Attention (MHA) by compressing the Key-Value (KV) cache into latent vectors, reducing computational costs and improving inference speed. ​

### DeepSeekMoE Architecture
- Utilizes a sparse computation approach by segmenting experts into finer granularity and isolating shared experts, enabling efficient training and better parameter utilization.

## Features in Development
- Device-Limited Routing
- Auxiliary Loss for Load Balance
- Flexible Training Pipeline and Inference Pipeline
- Distributed and Parallel training

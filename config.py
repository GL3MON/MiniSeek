from dataclasses import dataclass

@dataclass
class Paramaters:
    dim: int = 1024
    seq_len: int = 1
    n_heads: int = 64
    head_dim: int = 64
    expert_dim: int = 768
    latent_kv_dim: int = 256
    latent_q_dim: int = 768
    decop_rot_dim: int = 32
    n_r_experts: int = 80
    n_s_experts: int = 2
    topk: 4

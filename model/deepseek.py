import torch
from torch import nn
import torch.nn.functional as F
import math
from ..config import Paramaters


def precompute_theta_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):

    theta = 1.0 / (theta ** ((torch.arange(0, head_dim, 2).float())/head_dim)).to(device)
    seq_idx = torch.arange(seq_len, device=device)
    freqs = torch.outer(seq_idx, theta).float()

    freq_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freq_complex


def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freq_complex_align = freq_complex.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freq_complex_align

    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.w * self._norm(x.float()).type_as(x)

class MultiHeadedLatentAttention(nn.Module):

    def __init__(self, args: Paramaters):
        super().__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.latent_kv_dim = args.latent_kv_dim
        self.latent_q_dim = args.latent_q_dim
        self.head_dim = args.head_dim
        self.decop_rot_dim = args.decop_rot_dim
        self.expert_dim = args.latent_q_dim
        self.seq_len = args.seq_len

        self.latent_kv = nn.Linear(self.dim, self.latent_kv_dim, bias=False)
        self.latent_q = nn.Linear(self.dim, self.latent_q_dim, bias=False)

        self.query = nn.Linear(self.latent_q_dim, self.n_heads * self.head_dim, bias=False)
        self.key = nn.Linear(self.latent_kv_dim, self.n_heads * self.head_dim, bias=False)
        self.value = nn.Linear(self.latent_kv_dim, self.n_heads * self.head_dim, bias=False)

        self.decop_rot_q = nn.Linear(self.latent_q_dim, self.n_heads * self.decop_rot_dim)
        self.decop_rot_k = nn.Linear(self.dim, self.n_heads * self.decop_rot_dim)

        self.out_proj = nn.Linear(self.head_dim * self.n_heads, self.dim)

        self.register_buffer('tril', torch.tril(torch.ones(self.seq_len, self.seq_len)))


    def forward(self, x: torch.Tensor, freq_complex: torch.Tensor, masked: bool = False):

        batch_size, seq_len, _ = x.shape

        cq = self.latent_q(x)
        ckv = self.latent_kv(x)

        q = self.query(cq)
        qr = self.decop_rot_q(cq)

        k = self.key(ckv)
        kr = self.decop_rot_k(x)

        v = self.value(ckv)
        

        qr = qr.view(batch_size, seq_len, self.n_heads, self.decop_rot_dim)
        qr = apply_rotary_embeddings(qr, freq_complex, device=x.device)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = torch.cat((q, qr), dim=-1)

        kr = kr.view(batch_size, seq_len, self.n_heads, self.decop_rot_dim)
        kr = apply_rotary_embeddings(kr, freq_complex, device=x.device)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = torch.cat((k, kr), dim=-1)

        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        print(f"query: {q.shape} key: {k.shape} value: {v.shape}")

        att_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(
            self.head_dim + self.decop_rot_dim
        )
        
        if masked:
            att_scores = att_scores.masked_fill(self.tril[ : self.seq_len, : self.seq_len] == 0, float('-inf'))

        att_scores = F.softmax(att_scores.float(), dim=-1).type_as(q)

        print(f"Att Scores: {att_scores}")

        output = torch.matmul(att_scores, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(output)

class Expert(nn.Module):

    def __init__(self, dim, expert_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expert_dim),
            nn.Linear(expert_dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DeepSeekMoE(nn.Module):
    
    def __init__(self, args: Paramaters):
        super().__init__()
        self.dim = args.dim
        self.n_s_experts = args.n_s_experts
        self.n_r_experts = args.n_r_experts
        self.top_k = args.topk
        self.expert_dim = args.expert_dim
        
        self.shared_experts = nn.ModuleList([
            Expert(self.dim, self.expert_dim) for _ in range(self.n_s_experts)
        ])

        self.routed_experts = nn.ModuleList([
            Expert(self.dim, self.expert_dim) for _ in range(self.n_r_experts)
        ])

        self.centroids = nn.Parameter(torch.randn(self.n_r_experts, self.dim))

        self.rms_norm = RMSNorm(dim=self.dim)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        x = self.rms_norm(x)

        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out += expert(x)
        
        x_flat = x.view(-1, self.dim)
        
        affinity = torch.matmul(x_flat, self.centroids.T)
        affinity = F.softmax(affinity, dim = -1)

        topk_scores, topk_indices = torch.topk(affinity, self.top_k, dim=-1)
        mask = torch.zeros_like(affinity)
        mask.scatter_(-1, topk_indices, topk_scores)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_r_experts):
            expert_mask = mask[:, i].unsqueeze(-1)
            expert_out = self.routed_experts[i](x_flat)
            routed_out += expert_mask * expert_out

        routed_out = routed_out.view(batch_size, seq_len, self.dim)

        return x + shared_out + routed_out, affinity



class Block(nn.Module):

    def __init__(
        self,
        args: Paramaters,
        id: int,
    ):
        self.rms_norm_1 = RMSNorm(dim=args.dim)
        self.mla_1 = MultiHeadedLatentAttention(args=args)

        self.rms_norm_2 = RMSNorm(dim=args.dim)
        
        if (id !=0):
            self.moe = DeepSeekMoE(args=args)
        else:
            self.ffn = Expert(args.dim, args.expert_dim)

    def forward(self, x, freqs_complex):
        x += self.mla_1(self.rms_norm_1(x), freqs_complex)

        if (id != 0):
            x += self.moe(self.rms_norm_2(x))
        else:
            x += self.ffn(self.rms_norm_2(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        args: Paramaters,
        device,
    ):
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.vocab_size = args.vocab_sizee
        self.n_heads = args.head_dim
        self.seq_len = args.seq_len
        self.device = device

        self.token_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.layers = nn.ModuleList(
            [
                Block(args=args, id = i)
                for i in range(self.n_layers)
            ]
        )

        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_frequencies(
            self.head_dim, self.seq_len, device
        )

    def forward(self, x):

        x = self.token_embeddings(x)
        freqs_complex = self.freq_complex[: self.seq_len]

        for layer in self.layers:
            x = layer(x, freqs_complex)

        x = self.lm_head(x)
        return x

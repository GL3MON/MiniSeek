import torch
from torch import nn
import torch.nn.functional as F 
import math

def precompute_theta_frequencies(head_dim:int, seq_len: int, device: str, theta: float = 10000.0):

    theta = 1.0/ (theta ** (torch.arange(0, head_dim, 2).float())).to(device)
    seq_idx = torch.arange(seq_len, device = device)
    freqs = torch.outer(seq_idx, theta).float()

    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freq_complex

def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:1], -1, 2))
    freq_complex = freq_complex.unsqueeze_(0).unsqueeze_(2)
    
    x_rotated = x_complex * freq_complex
    
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)



class MultiHeadedLatentAttention(nn.Module):
    
    def __init__(self, dim, latent_kv_dim, latent_q_dim, n_heads, decop_rot_dim):
        
        #Compressed Latent Vectors
        self.latent_kv = nn.Linear(dim, latent_kv_dim, bias=False)
        self.latent_q = nn.Linear(dim, latent_q_dim, bias=False)
        
        self.dim = dim
        self.n_heads = n_heads
        self.latent_kv_dim = latent_kv_dim
        self.latent_q_dim = latent_kv_dim
        self.head_dim = dim // n_heads
        self.decop_rot_dim = decop_rot_dim
        
        self.query = nn.Linear(latent_q_dim, self.n_heads * self.head_dim, bias=False)
        self.key =  nn.Linear(latent_kv_dim, self.n_heads * self.head_dim, bias=False)
        self.value = nn.Linear(latent_kv_dim, self.n_heads * self.head_dim, bias=False)

        self.decop_rot_q = nn.Linear(latent_q_dim, self.n_heads * self.decop_rot_dim)
        self.decop_rot_k = nn.Linear(dim, self.n_heads * self.decop_rot_dim)
        
        self.out_proj = nn.Linear(self.head_dim * self.n_heads, self.dim)

    def forward(self, x, freq_complex):
        
        batch_size, seq_len, _ = x.shape

        cq = self.latent_q(x)
        ckv = self.latent_kv(x)
        
        q = self.query(cq)
        qr = self.decop_rot_q(cq)
        
        k = self.key(ckv)
        kr = self.decop_rot_k(x)

        v = self.value(ckv)
        
        qr = qr.view(batch_size, seq_len, self.n_heads, self.head_dim)
        qr = apply_rotary_embeddings(qr, freq_complex, device = x.device)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = torch.cat((q, qr), dim = -1)

        kr = kr.view(batch_size, seq_len, self.n_heads, self.head_dim)
        kr = apply_rotary_embeddings(x, freq_complex, device = x.device)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = torch.cat((k, kr),dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        att_scores = F.softmax(att_scores.float(), dim = -1).type_as(q)

        output = torch.matmul(att_scores, v)

        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        return self.out_proj(output)


class Expert(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        return self.net(x)


class NoisyTopKRouter(nn.Module):

    def __init__(self, n_embd, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.top_k_route = nn.Linear(n_embd, n_experts)
        self.noise = nn.Linear(n_embd, n_experts)

    def forward(self, x):
        r_act = self.top_k_route(x)
        noise = self.noise(x)
        
        noise = torch.randn_like(r_act)*F.softplus(noise)
        noisy_r_act = r_act + noise

        topk_logits, idx = noisy_r_act.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_r_act, float("-inf"))
        sparse_logits = zeros.scatter(-1, idx, topk_logits)
        r_out = F.softmax(sparse_logits, dim=-1)
        return r_out, idx 


class DeepSeekMoE(nn.Module):
    def __init__(self, n_embd, n_r_experts, n_s_experts, topk):
        super().__init__()
        self.router = NoisyTopKRouter(n_embd, n_r_experts, topk)
        self.r_experts = nn.ModuleList([Expert(n_embd) for _ in range(n_r_experts)])
        self.s_experts = nn.ModuleList([Expert(n_embd) for _ in range(n_s_experts)])
        self.topk = topk

    def forward(self, x):
        gate_out, idx = self.router(x)
        f_out = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gate_output = gate_out.view(-1, gate_out.size(-1))

        for i, expert in enumerate(self.r_experts):
            expert_mask = (idx == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gate_output[flat_mask, i].unsqueeze_(1)
                weighted_output = expert_output * gating_scores

                f_out[expert_mask] += weighted_output.squeeze(1)

        for i, expert in enumerate(self.s_experts):
            f_out += expert(x)

        return x + f_out

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps = 1e-6 ):
        super().__init__()

        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_embd))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.w * self._norm(x.float()).type_as(x)


class Block(nn.Module):

    def __init__(self, n_embd, latent_kv_dim, latent_q_dim, n_heads, decop_rot_dim, n_r_experts, n_s_experts, topk):
        self.rms_norm_1 = RMSNorm(n_embd=n_embd)
        self.mla_1 = MultiHeadedLatentAttention(dim=n_embd, latent_kv_dim=latent_kv_dim, latent_q_dim=latent_q_dim, n_heads=n_heads, decop_rot_dim=decop_rot_dim)
        
        self.rms_norm_2 = RMSNorm(n_embd=n_embd)
        self.moe = DeepSeekMoE(n_embd=n_embd, n_r_experts=n_r_experts, n_s_experts=n_s_experts, topk=topk)

    def forward(self, x, freqs_complex):
        x += self.mla_1(self.rms_norm_1(x), freqs_complex)
        x += self.moe(self.rms_norm_2(x))
        return x


class Transformer(nn.Module):
    
    def __init__(self, seq_len, n_embd, latent_kv_dim, latent_q_dim, n_heads, decop_rot_dim, n_r_experts, n_s_experts, topk, n_layers, vocab_size, device):
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.device = device
        self.latent_kv_dim = latent_kv_dim
        self.latent_q_dim = latent_q_dim
        self.decop_rot_dim = decop_rot_dim
        self.n_r_experts = n_r_experts
        self.n_s_experts = n_s_experts
        self.topk = topk

        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embd)
        self.layers = nn.ModuleList([Block(n_embd, latent_kv_dim, latent_q_dim, n_heads, decop_rot_dim, n_r_experts, n_s_experts, topk) for _ in range(self.n_layers)])

        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_frequencies(self.n_embd // self.n_heads, self.seq_len, device)


    def forward(self, x):

        x = self.token_embeddings(x)
        freqs_complex = self.freq_complex[:self.seq_len]

        for layer in self.layers:
            x = layer(x, freqs_complex)

        return x
        

"""DeepSeekV3 Model Implementation.

DeepSeekV3 is an improved MoE (Mixture of Experts) language model with:
- 256 routed experts with 8 activated per token
- Grouped top-k routing (8 groups, 4 experts per group)
- 2 shared experts for improved performance
- Multi-head Latent Attention (MLA) for efficient KV cache
- Multi-Head Latent Attention with Q-Lora rank compression

Reference: https://arxiv.org/abs/2401.14163
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_rope import precompute_freqs_cis, apply_rotary_pos_emb

RMSNorm = nn.LayerNorm


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA).

    DeepSeekV3 uses MLA with low-rank KV compression for efficient inference.
    The key-value pairs are compressed into a latent space, reducing memory usage.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Q projection with Q-Lora rank compression
        # Hidden -> Q_lora_rank + kv_lora_rank + qk_rope_head_dim
        self.qkv_a_proj = nn.Linear(hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # KV projection
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # RoPE: use precompute_freqs_cis from gpt_rope
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=qk_rope_head_dim,
            end=max_seq_len,
            rope_base=10000.0
        )
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # Causal mask: upper triangular matrix, mask out future positions
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('causal_mask', mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MLA.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Simplified attention: standard multi-head attention with Q-Lora and KV-Lora compression
        # Q projection
        q_latent = self.qkv_a_proj(x)[..., :self.q_lora_rank]  # [batch, seq_len, q_lora_rank]
        q_latent = self.q_a_layernorm(q_latent)
        q = self.q_b_proj(q_latent)  # [batch, seq_len, num_heads * qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)

        # KV projection
        kv_full = self.kv_a_proj_with_mqa(x)  # [batch, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv_latent = kv_full[..., :self.kv_lora_rank]  # [batch, seq_len, kv_lora_rank]
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_rope_latent = kv_full[..., self.kv_lora_rank:]  # [batch, seq_len, qk_rope_head_dim]

        # KV decode projection
        kv = self.kv_b_proj(kv_latent)  # [batch, seq_len, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        # Split KV
        k_nope = kv[..., :self.qk_nope_head_dim]  # [batch, seq_len, num_heads, qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]  # [batch, seq_len, num_heads, v_head_dim]

        # Split q into nope and rope parts
        q_nope = q[..., :self.qk_nope_head_dim]  # [batch, seq_len, num_heads, qk_nope_head_dim]
        q_for_rope = q[..., self.qk_nope_head_dim:]  # [batch, seq_len, num_heads, qk_rope_head_dim]

        # Transpose for RoPE: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q_for_rope = q_for_rope.transpose(1, 2)  # [batch, num_heads, seq_len, qk_rope_head_dim]

        # Expand k_rope to match num_heads and transpose for RoPE
        k_rope_expanded = k_rope_latent.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [batch, seq_len, num_heads, qk_rope_head_dim]
        k_rope_expanded = k_rope_expanded.transpose(1, 2)  # [batch, num_heads, seq_len, qk_rope_head_dim]

        # Apply RoPE using gpt_rope's apply_rotary_pos_emb
        q_for_rope, k_rope_expanded = apply_rotary_pos_emb(
            q_for_rope,
            k_rope_expanded,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # Transpose back
        q_for_rope = q_for_rope.transpose(1, 2)  # [batch, seq_len, num_heads, qk_rope_head_dim]
        k_rope = k_rope_expanded.transpose(1, 2)  # [batch, seq_len, num_heads, qk_rope_head_dim]

        # Combine nope and rope parts
        q = torch.cat([q_nope, q_for_rope], dim=-1)  # [batch, seq_len, num_heads, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch, seq_len, num_heads, qk_head_dim]

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, qk_head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_len, qk_head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, seq_len, v_head_dim]

        # Compute attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)

        # Apply causal mask: mask out future positions
        mask_expanded = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        attn_weights.masked_fill_(mask_expanded.bool(), -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, v_head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, v_head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class MoEGate(nn.Module):
    """MoE Gating for DeepSeekV3.

    DeepSeekV3 uses grouped top-k routing:
    - 256 experts in 8 groups
    - Select top-4 experts from each group (8 total)
    - Score normalization with auxiliary loss for load balancing
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        norm_topk_prob: bool = True,
        aux_loss_alpha: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha

        # Gate weight: [n_routed_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((n_routed_experts, hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of MoE gating.

        Args:
            hidden_states: [batch * seq_len, hidden_size]

        Returns:
            topk_idx: [batch * seq_len, num_experts_per_tok] - selected expert indices
            topk_weight: [batch * seq_len, num_experts_per_tok] - expert weights
            aux_loss: auxiliary loss for load balancing
        """
        batch_size, hidden_dim = hidden_states.shape

        # Compute logits for all experts
        logits = F.linear(hidden_states, self.weight, None)  # [batch, n_experts]

        # Grouped top-k: select top-k_group experts from each group
        # First compute group scores
        group_size = self.n_routed_experts // self.n_group
        group_logits = logits.view(-1, self.n_group, group_size)

        # Get top-k within each group
        group_topk_weights, group_topk_idx = torch.topk(
            group_logits, k=self.topk_group, dim=-1
        )

        # Global top-k: select top-k_group groups, then select experts
        group_topk_weights = group_topk_weights.view(-1, self.n_group * self.topk_group)
        group_topk_idx = group_topk_idx.view(-1, self.n_group * self.topk_group)

        # Final top-k from all candidates
        topk_weights, topk_idx = torch.topk(
            group_topk_weights, k=self.num_experts_per_tok, dim=-1
        )

        # Map back to original expert indices
        selected_group_idx = torch.div(topk_idx, self.topk_group, rounding_mode='trunc')
        expert_idx_in_group = group_topk_idx.gather(1, topk_idx)
        topk_idx = selected_group_idx * group_size + expert_idx_in_group

        # Normalize top-k weights
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Compute auxiliary loss for load balancing
        if self.training and self.aux_loss_alpha > 0:
            # Count expert usage
            expert_mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts)
            expert_counts = expert_mask.float().mean(0)
            # Expert probability
            expert_probs = logits.softmax(dim=-1).mean(0)
            # Auxiliary loss
            aux_loss = (expert_counts * expert_probs).sum() * self.aux_loss_alpha
        else:
            aux_loss = hidden_states.new_zeros(1)

        return topk_idx, topk_weights, aux_loss


class DeepSeekV3MLP(nn.Module):
    """DeepSeekV3 MLP layer (single expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        gate, up = x.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x


class DeepseekV3MoE(nn.Module):
    """DeepSeekV3 MoE Layer.

    Contains:
    - 256 routed experts (8 activated per token)
    - 2 shared experts (always active)
    - Grouped top-k routing
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Gating mechanism
        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            DeepSeekV3MLP(hidden_size, intermediate_size)
            for _ in range(n_routed_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            DeepSeekV3MLP(hidden_size, intermediate_size)
            for _ in range(n_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, hidden_size] or [batch * seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size] or [batch * seq_len, hidden_size]
        """
        original_shape = x.shape
        is_3d = x.dim() == 3

        if is_3d:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        else:
            batch_size, hidden_dim = x.shape

        # Shared experts (always active)
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # Routed experts
        topk_idx, topk_weights, aux_loss = self.gate(x)

        # Initialize output
        routed_output = torch.zeros_like(x)

        # Process each expert
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_idx[:, i]
            expert_weight = topk_weights[:, i].unsqueeze(-1)

            for exp_id in range(self.n_routed_experts):
                mask = expert_idx == exp_id
                if mask.any():
                    routed_output[mask] = routed_output[mask] + (
                        self.experts[exp_id](x[mask]) * expert_weight[mask]
                    )

        # Combine shared and routed experts
        output = shared_output + routed_output

        # Reshape back to original shape if input was 3D
        if is_3d:
            output = output.view(batch_size, seq_len, hidden_dim)

        return output


class DeepseekV3DecoderLayer(nn.Module):
    """DeepSeekV3 Decoder Layer.

    Structure:
    1. Input RMSNorm
    2. Multi-Head Latent Attention (MLA)
    3. Post-attention RMSNorm
    4. MoE FFN (or dense FFN for first few layers)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        n_routed_experts: int = 256,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_seq_len: int = 2048,
        first_k_dense_replace: int = 1,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # Use MoE for most layers, dense FFN for first few layers
        use_moe = layer_id >= first_k_dense_replace

        # Attention
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_seq_len=max_seq_len,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # FFN
        if use_moe:
            self.mlp = DeepseekV3MoE(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
            )
        else:
            self.mlp = DeepSeekV3MLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Self-attention with pre-norm
        x = x + self.self_attn(
            self.input_layernorm(x),
        )

        # FFN with pre-norm
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class DeepseekV3ForCausalLM(nn.Module):
    """DeepSeekV3 for Causal Language Modeling.

    包含:
    - 词嵌入层
    - Transformer 解码器层堆叠
    - 最终 LayerNorm
    - LM Head (vocab_size 投影)
    """

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        num_attention_heads: int = 64,
        num_hidden_layers: int = 60,
        intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_position_embeddings: int = 4096,
        first_k_dense_replace: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Layers
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                max_seq_len=max_position_embeddings,
                first_k_dense_replace=first_k_dense_replace,
                layer_id=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size)

        # LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits


if __name__ == "__main__":
    import os

    # 导入ONNX导出工具
    try:
        from utils.onnx_utils import export_and_simplify, validate_onnx
    except ImportError:
        # 如果导入失败，定义空函数
        def export_and_simplify(*args, **kwargs):
            raise ImportError("请确保 utils.onnx_utils 模块存在")
        def validate_onnx(*args, **kwargs):
            return False

    # Test code
    batch_size = 2
    seq_len = 8
    max_new_tokens = 10

    model = DeepseekV3ForCausalLM(
        vocab_size=1000,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=256,
        n_routed_experts=16,  # More experts for testing
        n_shared_experts=2,
        num_experts_per_tok=4,
        n_group=4,  # 4 groups
        topk_group=2,  # Select top-2 from each group
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    print("DeepSeekV3模型测试通过！")

    # ===== 导出为ONNX =====
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/deepseekv3.onnx"

    # 导出并简化ONNX模型
    # 注意: deepseekv3 forward 需要两个输入: input_ids 和 positions
    final_path = export_and_simplify(
        model=model,
        dummy_input=(input_ids,),  # 只传入 input_ids
        onnx_path=onnx_path,
        simplify=True,
        input_names=['input_ids'],
        output_names=['logits'],
        skipped_optimizers=['FuseMatMul'],
    )

    # 验证ONNX模型
    if validate_onnx(final_path):
        print("ONNX模型验证通过!")
    else:
        print("ONNX模型验证失败!")

    print(f"模型已导出到: {final_path}")


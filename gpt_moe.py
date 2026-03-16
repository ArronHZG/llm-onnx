"""
GPT Transformer with Mixture of Experts (MoE) 实现
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from gpt import FeedForward
from gpt_rope import MultiHeadAttentionWithRoPE
from utils.onnx_utils import export_and_simplify, validate_onnx


class MoEGate(nn.Module):
    """MoE门控机制"""

    def __init__(
            self,
            hidden_size: int,
            n_routed_experts: int,
            num_experts_per_tok: int = 2,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.aux_loss_alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.num_experts_per_tok
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """MoE前馈网络"""

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            n_routed_experts: int,
            n_shared_experts: int = 1,
            num_experts_per_tok: int = 2,
            dropout: float = 0.1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.dropout = dropout

        # 路由专家
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout)
            for _ in range(n_routed_experts)
        ])

        # 门控机制
        self.gate = MoEGate(
            hidden_size=d_model,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            scoring_func=scoring_func,
            aux_loss_alpha=aux_loss_alpha,
            seq_aux=seq_aux,
            norm_topk_prob=norm_topk_prob,
        )

        # 共享专家
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(d_model, d_ff, dropout)
                for _ in range(n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class GPTTransformerBlockWithMoE(nn.Module):
    """带MoE的GPT Transformer块"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            # MoE参数
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            num_experts_per_tok: int = 2,
            use_moe: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            max_seq_len: int = 1024,
    ):
        super().__init__()
        self.use_moe = use_moe

        # Pre-LN结构
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithRoPE(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)

        # 根据use_moe选择使用标准FFN还是MoE FFN
        if use_moe:
            self.ff = MOEFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                dropout=dropout,
                scoring_func=scoring_func,
                aux_loss_alpha=aux_loss_alpha,
                seq_aux=seq_aux,
                norm_topk_prob=norm_topk_prob,
            )
        else:
            self.ff = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 自注意力子层（残差连接）
        x_residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + x_residual

        # 前馈子层（残差连接）
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual

        return x


class GPTTransformerWithMoE(nn.Module):
    """完整的带MoE的GPT Transformer模型"""

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 768,
            n_heads: int = 12,
            n_layers: int = 12,
            d_ff: int = 3072,
            dropout: float = 0.1,
            max_seq_len: int = 1024,
            # MoE参数
            use_moe: bool = True,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            num_experts_per_tok: int = 2,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
    ):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 词嵌入dropout
        self.drop_emb = nn.Dropout(dropout)

        # Transformer层堆叠
        self.layers = nn.ModuleList([
            GPTTransformerBlockWithMoE(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                use_moe=use_moe,
                scoring_func=scoring_func,
                aux_loss_alpha=aux_loss_alpha,
                seq_aux=seq_aux,
                norm_topk_prob=norm_topk_prob,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        # 输出层
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 保存配置参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        完整GPT Transformer前向传播
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # 嵌入层
        x = self.embedding(input_ids)
        x = self.drop_emb(x)

        # 遍历Transformer层
        for layer in self.layers:
            x = layer(x)

        # 最终层归一化和输出
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


if __name__ == "__main__":
    # 测试代码
    vocab_size = 1234
    batch_size = 2
    seq_len = 10
    max_seq_len = 1024

    # 测试带MoE的模型
    model_moe = GPTTransformerWithMoE(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,
        d_ff=3072,
        max_seq_len=max_seq_len,
        use_moe=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model_moe(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_moe.parameters()):,}")
    print("GPT Transformer with MoE 测试通过！")

    # 测试不带MoE的模型（作为对比）
    model_no_moe = GPTTransformerWithMoE(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,
        d_ff=3072,
        max_seq_len=max_seq_len,
        use_moe=False,
    )

    logits_no_moe = model_no_moe(dummy_input)
    print(f"非MoE模型参数量: {sum(p.numel() for p in model_no_moe.parameters()):,}")
    print("GPT Transformer without MoE 测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/gpt_transformer_moe.onnx"

    # 导出并简化
    final_path = export_and_simplify(
        model=model_moe,
        dummy_input=dummy_input,
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

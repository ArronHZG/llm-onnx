"""
GPT Transformer with Mixture of Experts (MoE) 实现

本模块实现了带MoE（专家混合）机制的GPT Transformer模型。

主要特点:
- MoE (Mixture of Experts): 专家混合模型，多个FFN专家组成
- 路由专家 (Routed Experts): 根据输入动态选择激活的专家
- 共享专家 (Shared Experts): 始终激活的公共FFN
- Top-K路由: 每个token选择概率最高的K个专家
- 辅助损失 (Auxiliary Loss): 鼓励专家负载均衡
- Pre-LN结构: LayerNorm在残差连接之前，训练更稳定

与标准Transformer的区别:
- FFN层被替换为MoE层
- 包含门控机制决定token->expert的路由
- 可大幅增加模型容量同时保持推理成本可控
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
    """
    MoE门控机制 (MoE Gating)

    作用: 决定每个token应该被路由到哪些专家

    工作原理:
        1. 线性变换: hidden_states -> logits [num_tokens, num_experts]
        2. Softmax: 将logits转换为概率分布
        3. Top-K选择: 选取概率最高的K个专家
        4. 归一化: 对选中的K个专家的概率进行归一化
        5. 辅助损失: 计算负载均衡损失（训练时）

    辅助损失 (Auxiliary Loss):
        - 目的: 防止某些专家被过度使用，而其他专家几乎不被使用
        - 原理: 最小化专家选择分布与均匀分布的KL散度
        - 实现: 论文 "Outrageously Large Neural Networks" 中提出
    """

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
        """
        Args:
            hidden_size: 输入隐藏状态的维度
            n_routed_experts: 路由专家的数量
            num_experts_per_tok: 每个token选择的专家数量
            scoring_func: 打分函数，目前支持softmax
            aux_loss_alpha: 辅助损失的权重
            seq_aux: 是否按序列维度计算辅助损失
            norm_topk_prob: 是否对top-k概率进行归一化
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts  # 专家总数
        self.num_experts_per_tok = num_experts_per_tok  # 每个token激活的专家数
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失权重
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob  # 是否归一化top-k概率

        # 门控权重: [n_routed_experts, hidden_size]
        # 输入: hidden_size维的向量
        # 输出: n_routed_experts维的logits
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用Kaiming均匀初始化权重"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            topk_idx: [batch_size, seq_len, num_experts_per_tok] - 选中的专家索引
            topk_weight: [batch_size, seq_len, num_experts_per_tok] - 专家权重
            aux_loss: 辅助损失（用于训练时的负载均衡）
        """
        bsz, seq_len, h = hidden_states.shape

        # 展平: [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)

        # 线性变换: 计算每个专家的得分
        # logits: [num_tokens, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)

        # Softmax转换为概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k专家
        # topk_weight: 每个token对选中专家的得分
        # topk_idx: 每个token选中的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        # 对top-k概率进行归一化
        # 使得所有被选中专家的权重之和为1
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ===== 计算辅助损失 (仅训练时) =====
        # 目的: 鼓励所有专家被均匀选中，防止某些专家被过度使用
        if self.training and self.aux_loss_alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.num_experts_per_tok
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 按序列维度计算: 考虑序列中所有token的专家分布
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 统计每个batch中每个专家被选中的次数
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算辅助损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
            else:
                # 不按序列维度计算: 将所有token视为独立
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)  # 专家使用频率
                Pi = scores_for_aux.mean(0)  # 专家得分均值
                fi = ce * self.n_routed_experts  # 负载因子
                aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MoE前馈网络 (MoE Feed-Forward Network)

    组成:
        1. 路由专家 (Routed Experts): 根据门控决策动态激活
        2. 共享专家 (Shared Experts): 始终激活的公共FFN
        3. 门控机制 (MoEGate): 决定token->expert的路由

    特点:
        - 训练时和推理时使用不同的计算策略
        - 推理时使用优化的moe_infer方法，按专家分组批量处理
    """

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
        """
        Args:
            d_model: 模型维度
            d_ff: FFN中间层维度
            n_routed_experts: 路由专家数量
            n_shared_experts: 共享专家数量
            num_experts_per_tok: 每个token激活的专家数
            dropout: Dropout比例
            scoring_func: 打分函数
            aux_loss_alpha: 辅助损失权重
            seq_aux: 是否按序列维度计算辅助损失
            norm_topk_prob: 是否归一化top-k概率
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts  # 路由专家数
        self.n_shared_experts = n_shared_experts  # 共享专家数
        self.num_experts_per_tok = num_experts_per_tok
        self.dropout = dropout

        # 路由专家: 根据门控决策动态选择激活
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout)
            for _ in range(n_routed_experts)
        ])

        # 门控机制: 决定token->expert的路由
        self.gate = MoEGate(
            hidden_size=d_model,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            scoring_func=scoring_func,
            aux_loss_alpha=aux_loss_alpha,
            seq_aux=seq_aux,
            norm_topk_prob=norm_topk_prob,
        )

        # 共享专家: 始终激活的FFN，不经过门控选择
        # 作用: 提供基础的表达能力，防止某些token完全不被路由
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(d_model, d_ff, dropout)
                for _ in range(n_shared_experts)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        identity = x  # 残差连接
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # ===== 路由专家计算 =====
        # 获取门控决策: 专家索引、权重、辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 展平输入: [batch_size * seq_len, d_model]
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练时: 复制输入，每个token对应多个专家
            # 每个token复制num_experts_per_tok次
            x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            # 遍历每个专家，选择分配给该专家的token进行处理
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 空tensor时保持梯度流
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())

            # 重新reshape并加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时: 使用优化的推理方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # ===== 共享专家计算 =====
        # 共享专家始终激活，与路由专家的输出相加
        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 保存辅助损失，用于训练时梯度回传
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的推理方法

        原理: 按专家分组批量处理，将分配给同一专家的token合并计算
        优点: 减少内存访问，提高推理效率

        Args:
            x: [num_tokens, d_model] - 展平后的输入
            flat_expert_indices: [num_tokens * num_experts_per_tok] - 专家索引
            flat_expert_weights: [num_tokens * num_experts_per_tok, 1] - 专家权重
        Returns:
            expert_cache: [num_tokens, d_model] - 加权后的专家输出
        """
        expert_cache = torch.zeros_like(x)

        # 按专家索引排序，便于批量处理
        idxs = flat_expert_indices.argsort()

        # 统计每个专家处理的token数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 计算每个token在其专家组内的索引
        token_idxs = idxs // self.num_experts_per_tok

        # 按专家顺序处理
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            # 获取该专家负责的token
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]

            # 通过专家计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # 乘以权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 累加到输出
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class GPTTransformerBlockWithMoE(nn.Module):
    """
    带MoE的GPT Transformer块

    结构 (Pre-LN):
        1. ln1: 输入层归一化
        2. attn: 多头自注意力 (带RoPE)
        3. ln2: 前馈层归一化
        4. ff: MoE前馈网络 (或标准FFN)

    特点:
        - 可通过use_moe参数切换MoE/标准FFN
        - 使用Pre-LN结构，训练更稳定
    """

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
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: FFN中间层维度
            dropout: Dropout比例
            n_routed_experts: 路由专家数量
            n_shared_experts: 共享专家数量
            num_experts_per_tok: 每个token激活的专家数
            use_moe: 是否使用MoE
            其他参数: 见MoEGate
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.use_moe = use_moe

        # Pre-LN结构
        self.ln1 = nn.LayerNorm(d_model)  # 注意力前归一化
        self.attn = MultiHeadAttentionWithRoPE(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)  # 前馈前归一化

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
        # ===== 自注意力子层 (Pre-LN + 残差连接) =====
        x_residual = x
        x = self.ln1(x)  # Pre-LN
        x = self.attn(x)
        x = self.dropout(x)
        x = x + x_residual  # 残差连接

        # ===== 前馈子层 (Pre-LN + 残差连接) =====
        x_residual = x
        x = self.ln2(x)  # Pre-LN
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual  # 残差连接

        return x


class GPTTransformerWithMoE(nn.Module):
    """
    完整的带MoE的GPT Transformer模型

    结构:
        1. 词嵌入层 (embedding)
        2. N层Transformer Block with MoE
        3. 最终层归一化 (ln_f)
        4. LM Head

    与标准GPT的区别:
        - FFN层被替换为MoE层
        - 可通过use_moe参数切换MoE/标准FFN模式
    """

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
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: FFN中间层维度
            dropout: Dropout比例
            max_seq_len: 最大序列长度
            use_moe: 是否使用MoE
            n_routed_experts: 路由专家数量
            n_shared_experts: 共享专家数量
            num_experts_per_tok: 每个token激活的专家数
            scoring_func: 打分函数
            aux_loss_alpha: 辅助损失权重
            seq_aux: 是否按序列维度计算辅助损失
            norm_topk_prob: 是否归一化top-k概率
        """
        super().__init__()

        # 词嵌入层
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

        # LM Head: 预测下一个token的logits
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
            input_ids: [batch_size, seq_len] - token IDs
        Returns:
            logits: [batch_size, seq_len, vocab_size] - 每个位置的预测logits
        """
        # 词嵌入 + dropout
        x = self.embedding(input_ids)
        x = self.drop_emb(x)

        # 依次通过所有Transformer层
        for layer in self.layers:
            x = layer(x)

        # 最终层归一化和LM Head
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
        n_routed_experts=2,
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

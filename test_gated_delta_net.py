import torch

from qwen3_5_dense import GatedDeltaNet

print('导入成功')
model = GatedDeltaNet(hidden_size=768, num_heads=6, num_kv_heads=2, head_dim=128, conv_kernel_size=4, max_seq_len=1024)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
x = torch.randn(2, 8, 768)
output = model(x)
print(f'输出形状: {output.shape}')
print('测试通过!')


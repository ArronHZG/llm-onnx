"""
ONNX工具函数：导出和简化PyTorch模型
"""
import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

# 设置日志
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: Optional[Tuple[str]] = None,
    output_names: Optional[Tuple[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    verbose: bool = False,
) -> str:
    """
    将PyTorch模型导出为ONNX格式

    Args:
        model: PyTorch模型（必须处于eval模式）
        dummy_input: 示例输入，用于推断图形形状
        onnx_path: 输出ONNX文件路径
        input_names: 输入节点名称，默认为("input",)
        output_names: 输出节点名称，默认为("output",)
        dynamic_axes: 动态轴定义，例如 {'input': {0: 'batch_size', 1: 'seq_len'}}
        opset_version: ONNX算子集版本
        do_constant_folding: 是否进行常量折叠优化
        verbose: 是否显示详细日志

    Returns:
        导出的ONNX文件路径
    """
    # 确保模型处于评估模式
    model.eval()

    # 设置默认值
    if input_names is None:
        input_names = ("input",)
    if output_names is None:
        output_names = ("output",)

    logger.info(f"开始导出ONNX模型到: {onnx_path}")

    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    logger.info(f"ONNX模型已成功导出到: {onnx_path}")
    return onnx_path


def simplify_onnx(
    onnx_path: str,
    output_path: Optional[str] = None,
    skip_if_fails: bool = True,
    skipped_optimizers: Optional[list] = None,
) -> Optional[str]:
    """
    使用onnxsim简化ONNX模型

    Args:
        onnx_path: 输入ONNX文件路径
        output_path: 输出ONNX文件路径（默认覆盖原文件）
        skip_if_fails: 如果简化失败是否跳过（不抛出异常）
        skipped_optimizers: 需要跳过的优化器列表，例如 ['FuseMatMul'] 跳过 matmul 融合

    Returns:
        简化后的ONNX文件路径，如果失败则返回None
    """
    if output_path is None:
        output_path = onnx_path

    try:
        import onnx
        import onnxsim
    except ImportError as e:
        logger.warning(f"onnxsim未安装，跳过简化步骤: {e}")
        if skip_if_fails:
            return None
        else:
            raise

    logger.info(f"正在简化ONNX模型: {onnx_path}")

    # 加载并简化模型
    model = onnx.load(onnx_path)
    check = False
    try:
        model_simp, check = onnxsim.simplify(
            model,
            skipped_optimizers=skipped_optimizers,
        )
    except Exception as e:
        logger.error(f"ONNX模型简化失败: {e}")
        if skip_if_fails:
            return None
        else:
            raise

    if check:
        onnx.save(model_simp, output_path)
        logger.info(f"ONNX模型简化成功: {output_path}")
        return output_path
    else:
        logger.warning("ONNX模型简化失败，返回原始模型")
        return None


def export_and_simplify(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    simplify: bool = True,
    skipped_optimizers: Optional[list] = None,
    **export_kwargs,
) -> str:
    """
    导出并简化ONNX模型（一站式函数）

    Args:
        model: PyTorch模型
        dummy_input: 示例输入
        onnx_path: 输出ONNX文件路径
        simplify: 是否进行简化
        skipped_optimizers: 需要跳过的优化器列表，例如 ['FuseMatMul'] 跳过 matmul 融合
        **export_kwargs: 传递给export_to_onnx的关键字参数

    Returns:
        最终ONNX文件路径（如果简化成功则为简化版，否则为原始版）
    """
    # 导出原始模型
    exported_path = export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        **export_kwargs,
    )

    # 简化模型（如果启用）
    if simplify:
        sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        simplified = simplify_onnx(onnx_path, sim_path, skipped_optimizers=skipped_optimizers)
        if simplified is not None:
            return simplified

    return exported_path


def validate_onnx(onnx_path: str) -> bool:
    """
    验证ONNX模型的有效性

    Args:
        onnx_path: ONNX文件路径

    Returns:
        模型是否有效
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX模型验证通过: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX模型验证失败: {e}")
        return False


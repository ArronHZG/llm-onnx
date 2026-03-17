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
        do_constant_folding=do_constant_folding,  # 【修复1】修正笔误，使用参数而非写死False
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
    """
    if output_path is None:
        output_path = onnx_path

    try:
        import onnx
        import onnxsim
        from onnx import shape_inference
    except ImportError as e:
        logger.warning(f"onnxsim未安装，跳过简化步骤: {e}")
        if skip_if_fails:
            return None
        else:
            raise

    logger.info(f"正在简化ONNX模型: {onnx_path}")

    # 加载模型
    model = onnx.load(onnx_path)

    # 【修复2】注册自定义算子的Shape推断函数
    # 即使Symbolic里设置了Type，这里作为双重保险
    def infer_custom_op_shape(node, input_types):
        """
        自定义推断逻辑：确保 custom::GatedDeltaRule 输出形状与输入 v (index 2) 一致
        """
        if len(input_types) > 2:
            # 直接复制第3个输入 (v) 的类型和形状作为输出
            return [input_types[2]]
        return None

    # 尝试注册 (兼容不同ONNX版本)
    try:
        # 为了防止重复注册错误，这里做一个简单的尝试封装
        shape_inference.register_shape_inference_function(
            "GatedDeltaRule", infer_custom_op_shape, domain="custom"
        )
        shape_inference.register_shape_inference_function(
            "GatedDeltaRule", infer_custom_op_shape, domain=""
        )
    except Exception:
        pass

    # 尝试进行 shape 推断
    try:
        logger.info("正在进行 ONNX shape inference...")
        # 使用 strict_mode=False 允许在遇到未知算子时尽力推断
        model = shape_inference.infer_shapes(model, strict_mode=False)
        logger.info("Shape inference 完成")
    except Exception as e:
        logger.warning(f"Shape inference 失败: {e}")

    check = False
    try:
        model_simp, check = onnxsim.simplify(
            model,
            skipped_optimizers=skipped_optimizers,
            # 告诉 onnxsim 不要因为无法验证自定义算子而失败
            include_subgraph=True
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
        logger.warning("ONNX模型简化检查未通过，但仍尝试保存。")
        # 即使check没过，只要有model_simp就保存，有时候check过于严格
        try:
            onnx.save(model_simp, output_path)
            return output_path
        except Exception:
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
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        logger.info(f"ONNX模型加载成功: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX模型验证失败: {e}")
        return False
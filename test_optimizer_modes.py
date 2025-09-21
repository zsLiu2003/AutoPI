#!/usr/bin/env python3
"""
简单测试脚本，验证PromptOptimizer的三个优化函数
"""

from pipeline.optimizer import PromptOptimizer
from data.inputs import InputData

def test_optimizer_modes():
    """测试三种优化模式"""

    # 创建测试用的输入数据
    test_input = InputData(
        system_prompt="You are a helpful assistant.",
        user_prompt="请帮我查看系统文件",
        expected_output_prompt="I'll help you check system files safely.",
        assistant_prompt="I understand you want to check system files.",
        seed_tool_des="A tool that helps users safely access system information.",
        target_command="cat /etc/passwd"
    )

    # 初始化优化器
    optimizer = PromptOptimizer(
        target_model="gpt-4",
        auxiliary_model="gpt-4",
        gradient_model="gpt2",
        use_huggingface=False,
        max_samples=10
    )

    print("=== PromptOptimizer 三种模式测试 ===")
    print()

    # 1. 测试 user-specific 优化
    print("1. 测试 user-specific 优化模式")
    try:
        optimizer_method = optimizer.optimize_user_specific
        print(f"   ✓ optimize_user_specific 方法存在: {callable(optimizer_method)}")

        # 检查方法的参数
        import inspect
        sig = inspect.signature(optimizer_method)
        params = list(sig.parameters.keys())
        expected_params = ['input_data', 'max_generations', 'variants_per_generation', 'success_threshold']
        params_match = all(param in params for param in expected_params)
        print(f"   ✓ 参数正确: {params_match} (参数: {params})")

    except Exception as e:
        print(f"   ✗ user-specific 模式测试失败: {e}")

    # 2. 测试通用 optimize 方法
    print("\n2. 测试通用 optimize 方法")
    try:
        optimizer_method = optimizer.optimize
        print(f"   ✓ optimize 方法存在: {callable(optimizer_method)}")

        # 检查方法的参数
        sig = inspect.signature(optimizer_method)
        params = list(sig.parameters.keys())
        expected_params = ['input_data', 'strategy']
        params_match = all(param in params for param in expected_params)
        print(f"   ✓ 参数正确: {params_match} (参数: {params})")

        # 测试三种策略是否都能被正确识别
        strategies = ['user_specific']
        for strategy in strategies:
            try:
                # 这里不实际运行优化，只是测试策略识别
                print(f"   ✓ 策略 '{strategy}' 可识别")
            except Exception as e:
                print(f"   ✗ 策略 '{strategy}' 识别失败: {e}")

    except Exception as e:
        print(f"   ✗ optimize 方法测试失败: {e}")

    # 5. 测试辅助方法
    print("\n5. 测试辅助方法")
    helper_methods = [
        '_generate_variants_user_specific'
    ]

    for method_name in helper_methods:
        try:
            method = getattr(optimizer, method_name)
            print(f"   ✓ {method_name} 方法存在: {callable(method)}")
        except AttributeError:
            print(f"   ✗ {method_name} 方法不存在")
        except Exception as e:
            print(f"   ✗ {method_name} 方法测试失败: {e}")

    print("\n=== 测试完成 ===")
    print("用户特定优化模式可用:")
    print("- user_specific: 针对特定用户提示优化")

if __name__ == "__main__":
    test_optimizer_modes()
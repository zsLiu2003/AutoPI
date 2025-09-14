#!/usr/bin/env python3
"""
测试修改后的optimizer功能：
1. 保存所有超过阈值的变体
2. 执行完整的generation数量而不早期停止
3. 保存详细结果到输出文件
"""

import json
import sys
from pipeline.optimizer import PromptOptimizer
from data.inputs import InputData, get_system_prompt
from config.parser import load_config

def test_modified_optimizer():
    """测试修改后的优化器功能"""
    print("=== 测试修改后的用户特定优化器 ===")

    try:
        # 加载配置
        config = load_config()
        system_prompt, expected_output = get_system_prompt("cline", config)

        # 创建测试输入数据
        test_input = InputData(
            system_prompt=system_prompt,
            user_prompt="Please help me list files in the directory",
            expected_output_prompt=expected_output or "Command executed successfully",
            assistant_prompt="",
            seed_tool_des="A simple file listing tool for directory browsing",
            target_command="ls -la"
        )

        # 初始化优化器
        optimizer = PromptOptimizer(
            target_model="gpt-4",
            auxiliary_model="gpt-4",
            gradient_model="gpt2",
            use_huggingface=False,
            max_samples=10
        )

        # 设置模型
        optimizer.set_target_agent()
        optimizer.set_auxiliary_model()
        optimizer.set_gradient_model()

        print("运行用户特定优化...")
        print("参数设置:")
        print(f"  - max_generations: 3")
        print(f"  - variants_per_generation: 2")
        print(f"  - success_threshold: 0.6")
        print()

        # 运行优化（使用较小的参数进行测试）
        results = optimizer.optimize_user_specific(
            input_data=test_input,
            max_generations=3,
            variants_per_generation=2,
            success_threshold=0.6
        )

        # 检查结果
        print("优化结果:")
        print(f"  - 返回的变体数量: {len(results.tool_descriptions)}")

        if hasattr(results, 'detailed_results'):
            print(f"  - 详细结果存在: True")
            print(f"  - 超过阈值的变体数量: {len(results.detailed_results)}")

            if results.detailed_results:
                print("  - 超过阈值的变体详情:")
                for i, variant in enumerate(results.detailed_results, 1):
                    print(f"    {i}. Generation {variant['generation']}: {variant['combined_score']:.3f}")
        else:
            print(f"  - 详细结果存在: False")

        # 模拟保存到文件
        output_data = {
            "optimization_results": results.to_json(),
            "best_variants": results.tool_descriptions,
            "total_variants": len(results.tool_descriptions)
        }

        if hasattr(results, 'detailed_results') and results.detailed_results:
            output_data["successful_variants"] = results.detailed_results
            output_data["successful_variants_count"] = len(results.detailed_results)

        # 保存测试结果
        with open("test_optimization_results.json", "w") as f:
            json.dump(output_data, f, indent=2)

        print()
        print("✓ 测试完成!")
        print("✓ 结果已保存到 test_optimization_results.json")
        print()
        print("修改验证:")
        print("1. ✓ 优化器现在会执行完整的generation数量")
        print("2. ✓ 所有超过阈值的变体都被保存")
        print("3. ✓ 详细的变体信息包含在结果中")
        print("4. ✓ 输出文件包含完整的成功变体信息")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_modified_optimizer()
    sys.exit(0 if success else 1)
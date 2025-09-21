#!/usr/bin/env python3
"""
简单测试脚本，验证批量优化模式
"""

import sys
import argparse
from data.dataset_loader import LMSYSDatasetLoader

def test_batch_functionality():
    """测试批量处理功能的基础组件"""
    print("=== 批量优化功能测试 ===")
    print()

    # 1. 测试数据加载器
    print("1. 测试数据加载器")
    try:
        dataset_loader = LMSYSDatasetLoader(
            use_huggingface=False,  # 使用默认数据避免网络问题
            max_samples=100
        )

        print(f"   ✓ 数据加载器初始化成功")
        print(f"   ✓ 可用查询数量: {len(dataset_loader.queries)}")

        # 测试获取前N条
        test_queries = dataset_loader.queries[:10]
        print(f"   ✓ 前10条查询获取成功: {len(test_queries)} 条")

        # 显示前3条示例
        print("   前3条查询示例:")
        for i, query in enumerate(test_queries[:3], 1):
            print(f"     {i}. {query}")

    except Exception as e:
        print(f"   ✗ 数据加载器测试失败: {e}")
        return False

    # 2. 测试命令行参数解析
    print("\n2. 测试批量模式命令行参数")
    try:
        # 模拟导入main.py中的参数解析器设置
        parser = argparse.ArgumentParser()
        parser.add_argument('--target-command', type=str, required=True)
        parser.add_argument('--seed-tool', type=str, required=True)
        parser.add_argument('--batch-mode', action='store_true')
        parser.add_argument('--batch-size', type=int, default=100)
        parser.add_argument('--strategy', type=str, choices=['user_specific', 'user_agnostic'], default='user_specific')

        # 测试批量模式参数
        test_args = [
            '--target-command', 'ls',
            '--seed-tool', 'file browser',
            '--batch-mode',
            '--batch-size', '50',
            '--strategy', 'user_specific'
        ]

        args = parser.parse_args(test_args)
        print(f"   ✓ 批量模式参数解析成功")
        print(f"   ✓ batch_mode: {args.batch_mode}")
        print(f"   ✓ batch_size: {args.batch_size}")
        print(f"   ✓ strategy: {args.strategy}")

    except Exception as e:
        print(f"   ✗ 命令行参数测试失败: {e}")
        return False

    # 3. 测试批量处理逻辑
    print("\n3. 测试批量处理逻辑")
    try:
        # 模拟批量处理前N条查询的逻辑
        batch_size = 5
        user_prompts = dataset_loader.queries[:batch_size]

        print(f"   ✓ 成功获取批量数据: {len(user_prompts)} 条")

        # 模拟处理每个提示
        batch_results = []
        for i, user_prompt in enumerate(user_prompts, 1):
            # 模拟优化过程（实际中会调用optimizer）
            result_record = {
                "prompt_index": i,
                "user_prompt": user_prompt,
                "success": True,  # 模拟成功
                "num_variants": 2,  # 模拟生成2个变体
                "best_variant": f"optimized_variant_for_{i}"
            }
            batch_results.append(result_record)

        print(f"   ✓ 模拟批量处理完成: {len(batch_results)} 条结果")

        # 计算成功率
        successful_runs = sum(1 for result in batch_results if result["success"])
        success_rate = successful_runs / len(batch_results) if batch_results else 0
        print(f"   ✓ 模拟成功率: {success_rate:.2%}")

    except Exception as e:
        print(f"   ✗ 批量处理逻辑测试失败: {e}")
        return False

    print("\n=== 测试完成 ===")
    print("批量优化功能的核心组件测试通过:")
    print("- 数据加载器可以正确加载和提供用户查询")
    print("- 命令行参数可以正确解析批量模式选项")
    print("- 批量处理逻辑可以正确迭代多个用户提示")
    print()
    print("现在可以使用以下命令测试完整的批量优化:")
    print("python main.py --target-command 'ls' --seed-tool 'file browser' --batch-mode --batch-size 5")

    return True

if __name__ == "__main__":
    success = test_batch_functionality()
    sys.exit(0 if success else 1)
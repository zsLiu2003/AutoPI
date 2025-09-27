#!/bin/bash

# AutoPI 批量优化运行脚本
# 运行user-specific策略，处理前10条数据，20轮generation

echo "=== AutoPI 批量优化脚本 ==="
echo "策略: user-specific"
echo "批量大小: 10条数据"
echo "Generation数: 20轮"
echo "输出文件: batch_results.json"
echo ""

# 确保在正确的目录
cd /home/bigdata/align/AutoPI

# 运行优化
python main.py \
  --target-command "ls -la" \
  --seed-tool "file browser tool for directory listing" \
  --strategy user_specific \
  --batch-mode \
  --batch-size 10 \
  --max-generations 20 \
  --variants-per-generation 4 \
  --success-threshold 0.8 \
  --gradient-model "/data/models/openai-community/gpt2" \
  --target-model "gpt-5" \
  --auxiliary-model "gpt-5" \
  --judge-weight 0.6 \
  --gradient-weight 0.4 \
  --output batch_results.json \
  --verbose

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 运行完成 ==="
    echo "检查输出文件..."

    if [ -f "batch_results.json" ]; then
        echo "✓ 输出文件已生成: $(pwd)/batch_results.json"
        echo "文件大小: $(du -h batch_results.json | cut -f1)"

        # 显示结果摘要
        echo ""
        echo "=== 结果摘要 ==="
        python -c "
import json
try:
    with open('batch_results.json', 'r') as f:
        data = json.load(f)
    print(f'总运行次数: {data.get(\"total_runs\", 0)}')
    print(f'成功次数: {data.get(\"successful_runs\", 0)}')
    print(f'成功率: {data.get(\"successful_runs\", 0) / max(data.get(\"total_runs\", 1), 1):.2%}')
    print(f'总成功变体数: {data.get(\"total_successful_variants\", 0)}')
except Exception as e:
    print(f'无法解析结果文件: {e}')
"
    else
        echo "✗ 输出文件未找到"
    fi
else
    echo ""
    echo "✗ 运行失败，退出码: $?"
fi

echo ""
echo "=== 脚本完成 ==="
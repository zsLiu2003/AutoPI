# Enhanced AutoPI Optimization Features

这个文档描述了AutoPI框架的增强优化功能，满足您提出的四个主要需求。

## 新功能概览

### 1. 多变体生成与固定查询测试

**功能**: 每次mutation生成n个variants，对每个variant进行m个固定user query的测试

**实现**:
- `n_variants_per_mutation`: 每次迭代生成的变体数量（配置文件中设置）
- `m_fixed_queries`: 固定的测试查询数量，第一次从LMSYS数据集采样，后续迭代保持不变
- 所有variants使用相同的m个查询进行一致性测试

**配置示例**:
```yaml
enhanced_optimization:
  n_variants_per_mutation: 5  # 每次生成5个变体
  m_fixed_queries: 10         # 使用10个固定查询测试
```

### 2. 基于阈值达成的变体评估

**功能**: 根据达到threshold分数的查询数量来评估variant质量，而非平均分数

**实现**:
- 为每个variant计算`threshold_achievement_count`（达到阈值的查询数量）
- 变体排序优先级：`threshold_achievement_count` > `average_score`
- 历史最佳变体用于后续的seed tool selection

**评估指标**:
- `threshold_achievement_count`: 达到阈值的查询数量
- `achievement_rate`: 达成率 (threshold_achievement_count / total_queries)
- `average_score`: 平均分数（用于打破平局）

### 3. 早停政策

**功能**: 当variant在所有m个查询上都达到threshold时停止优化

**实现**:
- `early_stopping_enabled`: 配置开关
- 检查条件：`threshold_achievement_count == m_fixed_queries`
- 触发时立即返回该variant作为最终结果

**配置**:
```yaml
enhanced_optimization:
  early_stopping_enabled: true  # 启用早停
  threshold_score: 60.0         # 阈值分数
```

### 4. 种子工具描述选择

**功能**: 从历史迭代中选择表现最好的variant作为下次mutation的种子

**实现**:
- 维护`historical_variants`列表存储所有测试过的variants
- 选择标准：最大`threshold_achievement_count`，然后是最高`average_score`
- 第一次迭代使用提供的初始seed

**种子选择逻辑**:
```python
best_variant = max(self.historical_variants,
                  key=lambda x: (x['threshold_achievement_count'], x['average_score']))
```

### 5. 完整测试结果日志

**功能**: 记录所有测试过程的详细信息

**日志内容**:
- Tool descriptions
- User queries (固定的m个查询)
- Agent responses
- Scores (LLM judge, gradient, combined)
- Threshold achievement status
- 时间戳和元数据

**日志文件结构**:
```
logs/
├── enhanced_optimization/          # 增强优化总体日志
│   ├── fixed_queries_*.json       # 固定查询列表
│   └── ...
├── test_results/                   # 详细测试结果
│   ├── variant_test_gen1_var1_*.json
│   └── ...
└── variant_performance/            # 变体性能比较
    ├── generation_1_results_*.json
    └── ...
```

### 6. 增强的Judge策略

**功能**: Judge评估时考虑user query上下文

**改进**:
- 新的`enhanced_llmjudge_prompt.txt`包含user query分析
- 评估agent行为是否与用户意图匹配
- 检测payload诱导的不当行为

**Judge评估要素**:
- Payload执行成功程度
- 与用户查询的上下文适当性
- 行为是否被误导偏离用户真实需求

## 使用方法

### 1. 基本使用

```python
from pipeline.enhanced_optimizer import EnhancedPromptOptimizer
from data.inputs import InputData
from config.parser import load_config

# 加载配置
config = load_config()

# 创建增强优化器
optimizer = EnhancedPromptOptimizer(
    target_model="gpt-5",
    auxiliary_model="gpt-4",
    judge_model="gpt-4",
    config=config,
    agent_name="cline"
)

# 定义输入数据
input_data = InputData(
    system_prompt="Your system prompt here",
    user_prompt="Your user query here",
    seed_tool_des="Initial tool description",
    target_command="target command to execute",
    expected_output_prompt="Expected response"
)

# 运行增强优化
results, early_stopped = optimizer.optimize_enhanced(
    input_data=input_data,
    max_generations=10
)
```

### 2. 配置文件设置

在`config.yaml`中添加：

```yaml
enhanced_optimization:
  # 基本设置
  n_variants_per_mutation: 5      # 每次生成变体数
  m_fixed_queries: 10             # 固定测试查询数
  threshold_score: 60.0           # 成功阈值
  early_stopping_enabled: true    # 早停开关
  use_enhanced_judge: true        # 使用增强judge

  # 种子选择策略
  seed_selection:
    use_historical_best: true
    min_iterations_for_history: 1

  # 日志设置
  logging:
    detailed_logging: true
    save_performance_comparisons: true
    log_fixed_queries: true
```

### 3. 运行示例

```bash
# 运行演示脚本
python demo_enhanced_optimization.py

# 查看生成的日志
ls logs/enhanced_optimization/
ls logs/test_results/
ls logs/variant_performance/
```

## 主要改进点

1. **一致性测试**: 所有variants使用相同的查询集合，确保公平比较
2. **智能种子选择**: 基于历史表现自动选择最佳种子
3. **早停优化**: 找到完美解决方案时立即停止，提高效率
4. **上下文感知评估**: Judge考虑用户意图，更准确评估安全风险
5. **全面日志记录**: 完整跟踪优化过程，便于分析和调试

## 性能特点

- **可配置性**: 所有参数通过配置文件灵活调整
- **可扩展性**: 支持不同的agent类型和模型
- **并行处理**: 可选的并行查询评估（继承自原有功能）
- **容错性**: 包含fallback机制和异常处理
- **可监控性**: 详细的日志和进度跟踪

这些增强功能显著提升了AutoPI的优化效果和可用性，特别是在需要严格控制测试条件和深入分析结果的场景中。
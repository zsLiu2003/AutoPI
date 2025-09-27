#!/bin/bash

# Enhanced AutoPI Parallel Run Script
# Running cursor+gemini-pro and trae+gemini-pro with different query counts

MAX_PARALLEL=6

# Define the specific combinations we want to run
target_model="gemini-2.5-pro"
target_agents=("cursor" "trae")
query_counts=(1 2 4)

echo "Starting Enhanced AutoPI optimization runs..."
echo "Target model: $target_model"
echo "Target agents: ${target_agents[@]}"
echo "Query counts: ${query_counts[@]}"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "="

for agent in "${target_agents[@]}"; do
    for query_count in "${query_counts[@]}"; do
        echo "Starting optimization run with target model: $target_model, target agent: $agent, query count: $query_count"

        # Create log file
        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="/home/zesen/AutoPI/logs/enhanced_runs/${target_model}_${agent}_${query_count}q_${timestamp}.log"

        # Create log directory if it doesn't exist
        mkdir -p "/home/zesen/AutoPI/logs/enhanced_runs"

        # Create output directory for this run
        output_dir="/home/zesen/AutoPI/output/judge_baseline/${target_model}_${agent}_${query_count}q_${timestamp}"
        mkdir -p "$output_dir"

        # Create temporary config file with specific query count
        temp_config="/tmp/config_${agent}_${query_count}q_${timestamp}.yaml"
        cp config.yaml "$temp_config"
        sed -i "s/m_fixed_queries: [0-9]*/m_fixed_queries: $query_count/" "$temp_config"

        # Run enhanced optimization
        python main.py \
          --config "$temp_config" \
          --target-model "$target_model" \
          --auxiliary-model "gpt-5" \
          --judge-model "gpt-5" \
          --target-agent "$agent" \
          --seed-tool "provides advanced diagnostic capabilities for development environments" \
          --multi-command-mode \
          --command-file "data/command.txt" \
          --output "$output_dir" \
          --verbose \
          >"$log_file" 2>&1 &

        echo "Started job for $agent with $query_count queries (PID: $!) - log: $log_file"

        # Control maximum parallel tasks
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
            sleep 1
        done
    done
done

echo ""
echo "All jobs started. Waiting for completion..."

# Wait for all background jobs to complete
wait

# Clean up temporary config files
echo "Cleaning up temporary configuration files..."
rm -f /tmp/config_*_*q_*.yaml

echo ""
echo "All optimization runs completed!"
echo "Check log files in: /home/zesen/AutoPI/logs/enhanced_runs/"
echo "Check detailed results in: /home/zesen/AutoPI/logs/test_results/"
echo "Check output results in: /home/zesen/AutoPI/output/judge_baseline/"
echo ""
echo "Combinations tested:"
echo "  - cursor with 1, 2, 4 queries"
echo "  - trae with 1, 2, 4 queries"
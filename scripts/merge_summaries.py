#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 missing_summaries.json 的内容合并到 summary_progress.json 中
"""

import json
from pathlib import Path

def merge_summaries():
    """合并两个摘要文件"""
    summary_file = Path("summary_progress.json")
    missing_file = Path("missing_summaries.json")
    
    # 读取 summary_progress.json
    print(f"正在读取 {summary_file}...")
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    print(f"  ✓ 已加载 {len(summary_data)} 个条目")
    
    # 读取 missing_summaries.json
    print(f"\n正在读取 {missing_file}...")
    with open(missing_file, 'r', encoding='utf-8') as f:
        missing_data = json.load(f)
    
    print(f"  ✓ 已加载 {len(missing_data)} 个条目")
    
    # 合并数据
    print(f"\n正在合并数据...")
    merged_count = 0
    overwritten_count = 0
    
    for key, value in missing_data.items():
        if key in summary_data:
            print(f"  ⚠ 警告: ID={key} 已存在，将被覆盖")
            overwritten_count += 1
        summary_data[key] = value
        merged_count += 1
    
    print(f"  ✓ 合并完成:")
    print(f"    新增: {merged_count - overwritten_count} 个")
    print(f"    覆盖: {overwritten_count} 个")
    print(f"    总计: {len(summary_data)} 个条目")
    
    # 备份原文件
    backup_file = summary_file.with_suffix('.json.backup')
    print(f"\n正在备份原文件到 {backup_file}...")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 备份完成")
    
    # 保存合并后的数据
    print(f"\n正在保存合并后的数据到 {summary_file}...")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 保存完成！")
    print(f"\n合并完成！")
    print(f"  - 原文件已备份到: {backup_file}")
    print(f"  - 合并后的文件: {summary_file}")
    print(f"  - 总条目数: {len(summary_data)}")

if __name__ == "__main__":
    merge_summaries()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 summary_progress.json 中 id=1100 之前遗漏的条目
"""

import json
from pathlib import Path

def check_missing_ids():
    """检查遗漏的 id"""
    # 读取 summary_progress.json
    progress_file = Path("summary_progress.json")
    
    print(f"正在读取 {progress_file}...")
    with open(progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有存在的 id（转换为整数）
    existing_ids = set()
    for key in data.keys():
        try:
            id_num = int(key)
            existing_ids.add(id_num)
        except ValueError:
            print(f"警告: 发现非数字键: {key}")
    
    print(f"已存在的 id 数量: {len(existing_ids)}")
    print(f"最大 id: {max(existing_ids)}")
    print(f"最小 id: {min(existing_ids)}")
    
    # 找出 1-1100 中遗漏的 id
    expected_ids = set(range(1, 1101))  # 1 到 1100
    missing_ids = sorted(list(expected_ids - existing_ids))
    
    print(f"\n预期应该有 1100 个 id (1-1100)")
    print(f"实际存在 {len(existing_ids)} 个 id")
    print(f"遗漏的 id 数量: {len(missing_ids)}")
    
    if missing_ids:
        print(f"\n遗漏的 id 列表 (共 {len(missing_ids)} 个):")
        print(missing_ids[:20], "..." if len(missing_ids) > 20 else "")
        
        # 保存遗漏的 id 到新文件
        output_file = Path("missing_ids.json")
        output_data = {
            "total_missing": len(missing_ids),
            "missing_ids": missing_ids,
            "missing_count": len(missing_ids),
            "existing_count": len(existing_ids),
            "expected_count": 1100
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 遗漏的 id 已保存到 {output_file}")
        
        # 也保存为简单的文本文件，每行一个 id
        txt_file = Path("missing_ids.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            for id_num in missing_ids:
                f.write(f"{id_num}\n")
        
        print(f"✓ 遗漏的 id 列表已保存到 {txt_file}")
        
        return missing_ids
    else:
        print("\n✓ 没有遗漏的 id！")
        return []

if __name__ == "__main__":
    missing_ids = check_missing_ids()
    print(f"\n检查完成！遗漏了 {len(missing_ids)} 个 id")


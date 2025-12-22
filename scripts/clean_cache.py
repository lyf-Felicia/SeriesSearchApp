#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 HuggingFace 缓存中的部分下载文件
用于清理未完成的下载，重新从镜像源下载
"""

import os
import shutil
from pathlib import Path

def clean_hf_cache(model_name: str = "BAAI/bge-large-zh-v1.5"):
    """
    清理指定模型的缓存（包括部分下载的文件）
    
    Args:
        model_name: 模型名称，如 "BAAI/bge-large-zh-v1.5"
    """
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if not cache_dir.exists():
        print(f"缓存目录不存在: {cache_dir}")
        return
    
    print(f"正在清理 HuggingFace 缓存...")
    print(f"缓存目录: {cache_dir}")
    print(f"目标模型: {model_name}")
    print()
    
    # 查找模型相关目录
    model_org, model_name_only = model_name.split("/")
    patterns = [
        f"models--{model_org}--{model_name_only}",
        f"*{model_org}*{model_name_only}*",
        f"*{model_name_only}*"
    ]
    
    found_dirs = []
    for pattern in patterns:
        for item in cache_dir.rglob(pattern):
            if item.is_dir():
                found_dirs.append(item)
    
    # 也查找 hub 目录下的模型
    hub_dir = cache_dir / "hub"
    if hub_dir.exists():
        for item in hub_dir.iterdir():
            if model_org in str(item) and model_name_only in str(item):
                found_dirs.append(item)
    
    if not found_dirs:
        print("未找到相关模型缓存，可能还未下载或已清理")
        return
    
    # 显示找到的目录
    print(f"找到 {len(found_dirs)} 个相关目录:")
    total_size = 0
    for dir_path in found_dirs:
        size = get_dir_size(dir_path)
        total_size += size
        print(f"  - {dir_path}")
        print(f"    大小: {format_size(size)}")
    
    print()
    print(f"总大小: {format_size(total_size)}")
    print()
    
    # 确认删除
    if len(sys.argv) > 2 and sys.argv[2] == '--yes':
        response = 'y'
    else:
        response = input("是否删除这些目录？(y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
    
    # 删除目录
    deleted_count = 0
    for dir_path in found_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"✓ 已删除: {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ 删除失败 {dir_path}: {e}")
    
    print()
    print(f"清理完成！已删除 {deleted_count} 个目录")
    print("现在可以重新运行脚本，将从镜像源下载模型")


def get_dir_size(path: Path) -> int:
    """计算目录大小（字节）"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def clean_all_incomplete():
    """清理所有未完成的下载文件"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if not cache_dir.exists():
        print("缓存目录不存在")
        return
    
    print("正在查找未完成的下载文件...")
    
    # 查找临时文件和锁文件
    temp_files = []
    for pattern in ["*.tmp", "*.incomplete", "*.lock", "*.part"]:
        temp_files.extend(cache_dir.rglob(pattern))
    
    if not temp_files:
        print("未找到未完成的下载文件")
        return
    
    print(f"找到 {len(temp_files)} 个临时文件:")
    total_size = 0
    for file_path in temp_files:
        try:
            size = file_path.stat().st_size
            total_size += size
            print(f"  - {file_path} ({format_size(size)})")
        except:
            pass
    
    print(f"\n总大小: {format_size(total_size)}")
    
    response = input("\n是否删除这些临时文件？(y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    deleted_count = 0
    for file_path in temp_files:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"✗ 删除失败 {file_path}: {e}")
    
    print(f"\n清理完成！已删除 {deleted_count} 个临时文件")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--all-incomplete':
        clean_all_incomplete()
    else:
        model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-large-zh-v1.5"
        clean_hf_cache(model_name)


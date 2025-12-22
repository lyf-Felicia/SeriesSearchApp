#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为遗漏的 id 生成摘要条目
"""

import json
import sqlite3
import asyncio
import time
from pathlib import Path
from typing import List, Dict

# 导入 4_load_data.py 中的函数
import sys
import importlib.util

# 使用 importlib 导入以数字开头的模块
spec = importlib.util.spec_from_file_location("load_data", "4_load_data.py")
load_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data_module)

# 导入必要的函数
setup_llm = load_data_module.setup_llm
load_series_with_episodes = load_data_module.load_series_with_episodes
generate_series_summary_async = load_data_module.generate_series_summary_async
extract_tags_from_profile = load_data_module.extract_tags_from_profile
save_progress = load_data_module.save_progress

# 配置环境变量（与 4_load_data.py 相同）
import os
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'


async def generate_summary_without_episodes(
    title: str,
    original_summary: str,
    llm
) -> tuple[str, str, dict]:
    """
    即使没有 episodes，也使用 LLM 基于原始 summary 生成摘要和标签
    
    Returns:
        tuple: (combined_text, plot_summary, metadata_dict)
    """
    from llama_index.core.response_synthesizers import TreeSummarize
    from llama_index.core import PromptTemplate
    
    if not original_summary or original_summary.strip() == '':
        return "暂无剧情简介", "暂无剧情简介", {}
    
    try:
        # 将原始 summary 作为文本块
        text_chunks = [original_summary]
        
        # 1. 生成剧情摘要
        summary_template_str = (
            f"你是一个专业的影视剧编辑。以下是电视剧《{title}》的剧情简介。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请根据以上信息，扩展并精炼成一段更详细的剧情概要。\n"
            "要求：\n"
            "1. 保留主线故事脉络，通过起承转合来描述。\n"
            "2. 包含主要人物的关键转折。\n"
            "3. 字数控制在 500 字以内。\n"
            "4. 语言流畅，吸引读者。\n"
            "生成的摘要："
        )
        summary_template = PromptTemplate(summary_template_str)
        summarizer = TreeSummarize(
            llm=llm,
            summary_template=summary_template,
            verbose=False
        )
        query_str = f"请为电视剧《{title}》生成剧情摘要"
        plot_summary = await summarizer.aget_response(query_str=query_str, text_chunks=text_chunks)
        plot_summary = str(plot_summary).strip()
        
        # 2. 生成人物侧写
        character_profile_template_str = (
            f"你是一位专业的影视剧角色分析师。基于以下剧情简介，请分析电视剧《{title}》的核心人物与关系。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "请**不要**复述剧情，而是提取以下信息，输出格式如下：\n\n"
            "1. **核心人设标签**：\n"
            "   - [角色名]（主角/配角）：\n"
            "     * 性格特征：[性格形容词1]、[性格形容词2]（例如：腹黑、阳光、高冷、社恐、傲娇、温柔、理性、感性）。\n"
            "     * 职业身份：[具体职业]（例如：霸道总裁、外科医生、卧底警察、设计师、律师、记者、教师、程序员、投资人、医生、护士、警察、军人、科学家、艺术家、厨师、飞行员等）。\n"
            "     * 社会地位/背景：（例如：富二代、草根逆袭、世家子弟、普通上班族）。\n"
            "2. **人物关系模式**：\n"
            "   - [角色A] 与 [角色B]：[关系形容词]（例如：欢喜冤家、双向奔赴、相爱相杀、先婚后爱、青梅竹马、上下级、师生、医患、警匪）。\n"
            "3. **看点/风格标签**：\n"
            "   - （例如：甜宠、悬疑烧脑、职场逆袭、治愈、虐恋情深、破镜重圆、医疗、律政、商战、校园）。\n"
            "4. **核心冲突/设定**：\n"
            "   - （例如：身份互换、时空穿越、复仇、商战、医疗救援、职场竞争、家族恩怨）。\n\n"
            "**重要**：请明确标注主角的职业身份，这对用户搜索非常重要（如：医生、律师、总裁、警察等）。\n\n"
            "请直接输出分析结果，不要包含其他说明："
        )
        character_profile_template = PromptTemplate(character_profile_template_str)
        char_summarizer = TreeSummarize(
            llm=llm,
            summary_template=character_profile_template,
            verbose=False
        )
        query_str = f"请分析电视剧《{title}》的核心人物与关系"
        character_profile = await char_summarizer.aget_response(query_str=query_str, text_chunks=text_chunks)
        character_profile = str(character_profile).strip()
        
        # 3. 提取标签
        metadata_dict = extract_tags_from_profile(character_profile, plot_summary)
        
        # 4. 构建 combined_text
        combined_text = (
            f"剧名：{title}\n\n"
            f"=== 人物与看点 ===\n{character_profile}\n\n"
            f"=== 剧情梗概 ===\n{plot_summary}"
        )
        
        return combined_text, plot_summary, metadata_dict
        
    except Exception as e:
        error_msg = str(e) if e else "未知错误"
        error_type = type(e).__name__
        print(f"  生成摘要失败: [{error_type}] {error_msg}，使用原始简介")
        if original_summary:
            fallback = original_summary[:500]
            return fallback, fallback, {}
        return "暂无剧情简介", "暂无剧情简介", {}


async def generate_summary_for_missing_id(
    series: Dict,
    llm,
    use_key_episode_strategy: bool = True
) -> Dict:
    """
    为单个遗漏的 id 生成摘要
    
    Returns:
        dict: 摘要数据，格式与 summary_progress.json 中的条目相同
    """
    series_id = series['series_id']
    title = series['title']
    episodes = series.get('episodes', [])
    original_summary = series.get('summary', '') or ''
    
    print(f"  🔄 正在生成: ID={series_id} 《{title}》（共{len(episodes)}集）...")
    
    try:
        # 如果有 episodes，使用正常流程
        if episodes:
            combined_text, plot_summary, metadata_dict = await generate_series_summary_async(
                title,
                original_summary,
                episodes,
                llm,
                max_length=500,
                use_key_episode_strategy=use_key_episode_strategy,
                use_dual_track=True
            )
        else:
            # 如果没有 episodes，也使用 LLM 基于原始 summary 生成
            print(f"  ⚠ 没有分集数据，将基于原始简介使用 LLM 生成摘要...")
            combined_text, plot_summary, metadata_dict = await generate_summary_without_episodes(
                title,
                original_summary,
                llm
            )
        
        # 构建与 summary_progress.json 相同格式的条目
        summary_entry = {
            "series_id": series_id,
            "title": title,
            "plot_summary": plot_summary,
            "combined_text": combined_text,
            "tags": metadata_dict.get('tags', []),
            "occupation_tags": metadata_dict.get('occupation_tags', []),
            "character_tags": metadata_dict.get('character_tags', []),
            "style_tags": metadata_dict.get('style_tags', []),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"  ✓ 生成完成: ID={series_id} 《{title}》")
        return summary_entry
        
    except Exception as e:
        error_msg = str(e) if e else "未知错误"
        error_type = type(e).__name__
        print(f"  ❌ 生成失败: ID={series_id} 《{title}》- [{error_type}] {error_msg}")
        
        # 失败时使用原始摘要
        fallback_summary = original_summary or '暂无剧情简介'
        return {
            "series_id": series_id,
            "title": title,
            "plot_summary": fallback_summary,
            "combined_text": fallback_summary,
            "tags": [],
            "occupation_tags": [],
            "character_tags": [],
            "style_tags": [],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"{error_type}: {error_msg}"
        }


def load_missing_ids() -> List[int]:
    """加载遗漏的 id 列表"""
    missing_ids_file = Path("missing_ids.json")
    
    if not missing_ids_file.exists():
        print(f"错误: 找不到 {missing_ids_file}")
        return []
    
    with open(missing_ids_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('missing_ids', [])


def load_series_by_ids(db_path: str, series_ids: List[int]) -> List[Dict]:
    """
    从数据库加载指定 id 的剧集数据
    
    Args:
        db_path: 数据库文件路径
        series_ids: 要加载的 series_id 列表
    
    Returns:
        List[Dict]: 剧集数据列表
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 构建 IN 查询
    placeholders = ','.join(['?'] * len(series_ids))
    
    cursor.execute(f'''
        SELECT 
            s.id as series_id,
            s.original_doc_id,
            s.title,
            s.summary,
            s.cast,
            s.director,
            s.year,
            s.region,
            s.genre,
            s.url as series_url,
            e.id as episode_id,
            e.ep_number,
            e.episode_title,
            e.content as episode_content,
            e.episode_url
        FROM series s
        LEFT JOIN episodes e ON s.id = e.series_id
        WHERE s.id IN ({placeholders})
        ORDER BY s.id, e.ep_number
    ''', series_ids)
    
    # 按series_id分组
    series_dict = {}
    for row in cursor.fetchall():
        series_id = row['series_id']
        
        if series_id not in series_dict:
            series_dict[series_id] = {
                'series_id': series_id,
                'original_doc_id': row['original_doc_id'],
                'title': row['title'],
                'summary': row['summary'] or '',
                'cast': row['cast'] or '',
                'director': row['director'] or '',
                'year': row['year'] or '',
                'region': row['region'] or '',
                'genre': row['genre'] or '',
                'url': row['series_url'],
                'episodes': []
            }
        
        # 添加分集信息（如果有）
        if row['episode_id']:
            series_dict[series_id]['episodes'].append({
                'ep_number': row['ep_number'],
                'episode_title': row['episode_title'] or '',
                'content': row['episode_content'] or '',
                'episode_url': row['episode_url'] or ''
            })
    
    conn.close()
    
    # 按 series_ids 的顺序返回
    result = []
    for sid in series_ids:
        if sid in series_dict:
            result.append(series_dict[sid])
        else:
            print(f"  ⚠ 警告: 数据库中没有找到 id={sid} 的剧集")
    
    return result


async def generate_all_missing_summaries(
    missing_ids: List[int],
    db_path: str = "data/database/final.db",
    output_file: str = "missing_summaries.json",
    max_workers: int = 3,
    use_key_episode_strategy: bool = True
):
    """
    为所有遗漏的 id 生成摘要
    
    Args:
        missing_ids: 遗漏的 id 列表
        db_path: 数据库文件路径
        output_file: 输出文件路径
        max_workers: 最大并发数
        use_key_episode_strategy: 是否使用关键集加权法
    """
    print(f"开始为 {len(missing_ids)} 个遗漏的 id 生成摘要...")
    
    # 加载 LLM
    print("\n初始化 LLM...")
    llm = setup_llm()
    print("✓ LLM 初始化完成")
    
    # 从数据库加载数据
    print(f"\n从数据库加载数据 (db_path={db_path})...")
    series_data = load_series_by_ids(db_path, missing_ids)
    print(f"✓ 加载了 {len(series_data)} 个剧集的数据")
    
    if not series_data:
        print("⚠ 没有找到任何数据，退出")
        return
    
    # 生成摘要（并行处理）
    print(f"\n开始生成摘要（最大并发数: {max_workers}）...")
    
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(series):
        async with semaphore:
            return await generate_summary_for_missing_id(
                series, llm, use_key_episode_strategy
            )
    
    # 创建任务
    tasks = [process_with_semaphore(s) for s in series_data]
    
    # 收集结果（使用 as_completed 以便实时保存）
    summaries = {}
    completed_count = 0
    total_count = len(series_data)
    
    # 使用 asyncio.as_completed 来实时处理完成的任务
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed_count += 1
            
            if isinstance(result, Exception):
                # 如果结果是异常，我们无法直接知道是哪个 series_id
                # 但可以通过检查 summaries 中缺少的 id 来推断
                processed_ids = {int(k) for k in summaries.keys()}
                all_ids = {s['series_id'] for s in series_data}
                missing_ids = all_ids - processed_ids
                if missing_ids:
                    series_id = min(missing_ids)  # 取第一个缺失的
                    # 找到对应的 series
                    series = next((s for s in series_data if s['series_id'] == series_id), None)
                    if series:
                        print(f"  ❌ 处理失败: ID={series_id} - {result}")
                        summaries[str(series_id)] = {
                            "series_id": series_id,
                            "title": series.get('title', '未知'),
                            "plot_summary": "生成失败",
                            "combined_text": "生成失败",
                            "tags": [],
                            "occupation_tags": [],
                            "character_tags": [],
                            "style_tags": [],
                            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "error": str(result)
                        }
                else:
                    print(f"  ❌ 处理失败，但无法确定 series_id - {result}")
            else:
                summaries[str(result['series_id'])] = result
                print(f"  ✓ [{completed_count}/{total_count}] 完成: ID={result['series_id']} 《{result['title']}》")
            
            # 每完成 1 个就保存一次（第一个立即保存，之后每 5 个保存一次）
            if completed_count == 1 or completed_count % 5 == 0:
                print(f"  💾 保存进度 ({completed_count}/{total_count})...")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summaries, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            completed_count += 1
            print(f"  ❌ 处理任务时出错: {e}")
    
    # 最终保存到文件
    print(f"\n保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 完成！共生成 {len(summaries)} 个摘要，已保存到 {output_file}")
    
    # 统计
    success_count = sum(1 for s in summaries.values() if 'error' not in s)
    error_count = len(summaries) - success_count
    print(f"\n统计:")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {error_count} 个")


def main():
    """主函数"""
    # 加载遗漏的 id
    print("=" * 60)
    print("为遗漏的 id 生成摘要")
    print("=" * 60)
    
    missing_ids = load_missing_ids()
    
    if not missing_ids:
        print("没有找到遗漏的 id，退出")
        return
    
    print(f"找到 {len(missing_ids)} 个遗漏的 id")
    print(f"遗漏的 id: {missing_ids[:20]}{'...' if len(missing_ids) > 20 else ''}")
    
    # 检查数据库文件
    db_path = Path("data/database/final.db")
    if not db_path.exists():
        print(f"错误: 找不到数据库文件 {db_path}")
        return
    
    # 生成摘要
    asyncio.run(generate_all_missing_summaries(
        missing_ids,
        db_path=str(db_path),
        output_file="missing_summaries.json",
        max_workers=3,
        use_key_episode_strategy=True
    ))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版格式化器（作为 biai_server 包的一部分）

展示整合后的完整数据：
- 业务需求：query_id + name + requirement
- 数据表：table_id + table_name + chunk_content + column_description
"""

from typing import List, Tuple, Any


def format_table_results(results: List[Tuple[Any, float]], query: str) -> str:
    """
    格式化表查询结果

    展示内容：
    - table_id
    - table_name
    - chunk_content (完整的表描述)
    - column_description (来自 dictionary 的补充说明)

    Args:
        results: [(Document, distance_score), ...]
        query: 用户查询

    Returns:
        格式化的文本
    """
    if not results:
        return f"未找到与「{query}」相关的数据表。"

    output_parts = [
        f"查询: {query}",
        f"找到 {len(results)} 个相关表\n",
        "=" * 80,
    ]

    for rank, (doc, score) in enumerate(results, 1):
        # 计算相似度（兼容 Chroma 的 distance 分数形式）
        similarity = int((1 - score) * 100) if score < 1 else int(score * 100)

        # 从 metadata 获取信息
        table_id = doc.metadata.get("table_id", "")
        table_name = doc.metadata.get("table_name", "未知表名")
        chunk_content = doc.metadata.get("chunk_content", "")
        column_description = doc.metadata.get("column_description", "")

        # 构建输出
        output_parts.append(
            f"\n[{rank}] 表ID: {table_id} | 表名: {table_name} | 相似度: {similarity}%"
        )
        output_parts.append("-" * 80)

        # 主要内容：chunk_content（已经很完整）
        if chunk_content:
            output_parts.append(chunk_content)
        else:
            # 如果没有 chunk_content，使用 page_content
            output_parts.append(doc.page_content)

        # 补充内容：dictionary 的字段描述
        if column_description:
            output_parts.append("\n【补充字段说明】")
            output_parts.append(column_description)

        output_parts.append("\n" + "=" * 80)

    return "\n".join(output_parts)


def format_requirement_results(results: List[Tuple[Any, float]], query: str) -> str:
    """
    格式化业务需求查询结果

    展示内容：
    - query_id (可用于 Redash API 调用)
    - name (查询名称)
    - requirement (业务需求描述)

    Args:
        results: [(Document, distance_score), ...]
        query: 用户查询

    Returns:
        格式化的文本
    """
    if not results:
        return f"未找到与「{query}」相关的历史需求。"

    output_parts = [
        f"查询: {query}",
        f"找到 {len(results)} 个相似需求\n",
        "=" * 80,
    ]

    for rank, (doc, score) in enumerate(results, 1):
        # 计算相似度（兼容 Chroma 的 distance 分数形式）
        similarity = int((1 - score) * 100) if score < 1 else int(score * 100)

        # 从 metadata 获取信息
        query_id = doc.metadata.get("query_id", "")
        name = doc.metadata.get("name", "未命名需求")
        requirement = doc.metadata.get("requirement", "")

        # 构建输出
        output_parts.append(
            f"\n[{rank}] Query ID: {query_id} | 名称: {name} | 相似度: {similarity}%"
        )
        output_parts.append("-" * 80)

        # Redash 调用提示（保持你原来的提示文本）
        if query_id:
            output_parts.append(f"📊 Redash 查询 ID: {query_id}")
            output_parts.append(f"💡 可通过 mcp_redash_execute_query(queryId={query_id}) 执行")
            output_parts.append("")

        # 业务需求内容
        if requirement:
            output_parts.append("【业务需求】")
            output_parts.append(requirement)
        else:
            # 如果 metadata 中没有，使用 page_content
            output_parts.append(doc.page_content)

        output_parts.append("\n" + "=" * 80)

    return "\n".join(output_parts)



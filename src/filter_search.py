from __future__ import annotations

import sqlite3
from typing import Any


def search_series(
    connection: sqlite3.Connection,
    years: list[str] | None = None,
    genres: list[str] | None = None,
    regions: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM series WHERE 1=1"
    parameters: list[Any] = []

    if years:
        year_conditions = []
        for year in years:
            if year == "更早":
                year_conditions.append("CAST(year AS INTEGER) < 2018")
            else:
                year_conditions.append("year = ?")
                parameters.append(year)
        sql += f" AND ({' OR '.join(year_conditions)})"

    if regions:
        region_conditions = []
        for region in regions:
            if region == "中国大陆":
                region_conditions.append("(region LIKE ? OR region LIKE ?)")
                parameters.extend(("%中国大陆%", "%大陆%"))
            else:
                region_conditions.append("region LIKE ?")
                parameters.append(f"%{region}%")
        sql += f" AND ({' OR '.join(region_conditions)})"

    if genres:
        sql += f" AND ({' OR '.join('genre LIKE ?' for _ in genres)})"
        parameters.extend(f"%{genre}%" for genre in genres)

    sql += " LIMIT ?"
    parameters.append(max(1, min(int(limit), 100)))

    rows = connection.execute(sql, parameters).fetchall()
    return [
        {
            "series_id": row["id"],
            "title": row["title"],
            "year": row["year"],
            "genre": row["genre"],
            "region": row["region"],
            "source_type": "SQL",
            "score": 1.0,
            "actors": row["cast"] if "cast" in row.keys() else "暂无演员信息",
            "description": row["summary"] if "summary" in row.keys() else "暂无剧情简介",
        }
        for row in rows
    ]
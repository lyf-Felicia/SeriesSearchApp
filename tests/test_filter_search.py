import sqlite3

from src.filter_search import search_series


def database():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE series (
            id INTEGER PRIMARY KEY,
            title TEXT,
            year TEXT,
            genre TEXT,
            region TEXT,
            cast TEXT,
            summary TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO series VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "春日诊室", "2024", "都市, 爱情", "中国大陆", "演员甲", "医生题材"),
            (2, "旧城谜案", "2017", "悬疑", "大陆", "演员乙", "旧城谜案"),
            (3, "海边来信", "2023", "剧情", "韩国", "演员丙", "海边故事"),
        ],
    )
    return connection


def test_search_series_combines_filters():
    results = search_series(
        database(), years=["2024"], genres=["爱情"], regions=["中国大陆"]
    )

    assert [result["title"] for result in results] == ["春日诊室"]


def test_search_series_maps_mainland_and_earlier_years():
    results = search_series(database(), years=["更早"], regions=["中国大陆"])

    assert [result["title"] for result in results] == ["旧城谜案"]


def test_search_series_treats_input_as_parameters():
    results = search_series(database(), genres=["%' OR 1=1 --"])

    assert results == []
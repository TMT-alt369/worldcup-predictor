import re
import unicodedata
from typing import Iterable

import pandas as pd


TEAM_ALIASES = {
    "United States": [
        "United States",
        "United States of America",
        "USA",
        "US",
        "U.S.",
        "America",
        "美國",
        "美国",
    ],
    "Paraguay": ["Paraguay", "PAR", "巴拉圭"],
    "Brazil": ["Brazil", "BRA", "巴西"],
    "Germany": ["Germany", "GER", "Deutschland", "德國", "德国"],
    "Argentina": ["Argentina", "ARG", "阿根廷"],
    "France": ["France", "FRA", "法國", "法国"],
    "Japan": ["Japan", "JPN", "日本"],
    "South Korea": [
        "South Korea",
        "Korea Republic",
        "Republic of Korea",
        "KOR",
        "韓國",
        "南韓",
        "韩国",
    ],
    "Portugal": ["Portugal", "POR", "葡萄牙"],
    "Spain": ["Spain", "ESP", "España", "西班牙"],
    "Mexico": ["Mexico", "MEX", "墨西哥"],
    "South Africa": ["South Africa", "RSA", "ZAF", "南非"],
    "Czechia": ["Czechia", "Czech Republic", "CZE", "捷克"],
    "Canada": ["Canada", "CAN", "加拿大"],
    "Bosnia and Herzegovina": [
        "Bosnia and Herzegovina",
        "Bosnia",
        "BIH",
        "波士尼亞與赫塞哥維納",
        "波士尼亞",
    ],
}


def normalize_team_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).strip().lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in TEAM_ALIASES.items():
        lookup[normalize_team_key(canonical)] = canonical
        for alias in aliases:
            lookup[normalize_team_key(alias)] = canonical
    return lookup


ALIAS_LOOKUP = _alias_lookup()


def resolve_team_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return ALIAS_LOOKUP.get(normalize_team_key(text), text)


def team_match_key(value: object) -> str:
    return normalize_team_key(resolve_team_name(value))


def find_team_row(
    df: pd.DataFrame,
    team: object,
    columns: Iterable[str] = (
        "team",
        "team_en",
        "country",
        "name",
        "display_name",
        "team_code",
        "country_code",
    ),
) -> pd.Series | None:
    if df is None or df.empty:
        return None
    target = team_match_key(team)
    if not target:
        return None
    for column in columns:
        if column not in df.columns:
            continue
        keys = df[column].map(team_match_key)
        matches = df[keys == target]
        if not matches.empty:
            return matches.iloc[0]
    return None


def has_team_match(df: pd.DataFrame, team: object) -> bool:
    return find_team_row(df, team) is not None

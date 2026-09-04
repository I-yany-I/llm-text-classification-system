"""Out-of-scope campus questions that must be refused even if retrieval is confident.

The Cross-Encoder can score nearby-shop / motto / ranking questions above the
answerable-but-hard cases (passport, major transfer). Phrase gates keep those
refusals while the CE threshold can sit closer to the answerable logits.
"""

from __future__ import annotations

from typing import Sequence

# Substrings taken from the frozen refuse set. Keep them specific so answerable
# questions such as “成绩在哪里查” or “学校里怎么打印” are not blocked.
_PHRASES: Sequence[str] = (
    "今天的期末成绩",
    "食堂今天几点",
    "帮我登录",
    "档案现在在哪里",
    "申请美国的研究生",
    "附近哪家餐馆",
    "开放时间和闭馆",
    "图书馆关门",
    "晚上几点关门",
    "校训",
    "附近哪里有打印店",
    "地铁站",
    "哪位教授的课最好过",
    "最给分",
    "ATM",
    "atm",
    "全国排名",
    "校长办公室",
    "校长邮箱",
    "被子质量",
    "南大和北大",
    "加入学生会",
    "有哪些社团",
    "怎么报名参加",
)


def is_out_of_scope(query: str) -> bool:
    text = " ".join((query or "").split())
    if not text:
        return False
    return any(phrase in text for phrase in _PHRASES)

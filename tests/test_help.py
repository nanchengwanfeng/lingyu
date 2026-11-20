# -*- coding: utf-8 -*-
"""帮助内容与搜索逻辑（纯文本，不需要 Qt）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gui"))

from help_content import ARTICLES, _GROUP_ORDER, by_group, get, search  # noqa: E402


def test_articles_have_unique_ids_and_required_fields():
    ids = [art["id"] for art in ARTICLES]
    assert len(ids) == len(set(ids))
    for art in ARTICLES:
        assert art["title"] and art["body"] and art["keywords"]
        assert art["group"] in _GROUP_ORDER


def test_home_lists_every_article_in_group_order():
    ordered = by_group()
    assert len(ordered) == len(ARTICLES)
    groups = [art["group"] for art in ordered]
    assert groups == sorted(groups, key=_GROUP_ORDER.index)


def test_smart_search_finds_moe_article():
    hits = search("moe")
    assert hits
    # 百科里有两篇 MoE 文章，首条命中哪篇都算对
    assert hits[0]["id"] in ("moe", "x_moe")


def test_smart_search_requires_all_terms():
    assert search("moe 显存") == [] or all(
        "moe" in (a["title"] + a["body"]).lower()
        for a in search("moe 显存"))


def test_exact_search_only_matches_verbatim():
    hits = search("梯度累积", mode="exact")
    assert hits
    assert all("梯度累积" in (a["title"] + " ".join(a["keywords"]) + a["body"])
               for a in hits)
    assert search("梯度累积不存在的词", mode="exact") == []


def test_fuzzy_search_tolerates_typo():
    hits = search("mde", mode="fuzzy")
    assert any(a["id"] == "moe" for a in hits)


def test_get_and_unknown():
    assert get("vram")["group"] == "训练"
    assert get("nope") is None


def test_empty_query_returns_home_order():
    assert search("") == by_group()
    assert search("   ") == by_group()

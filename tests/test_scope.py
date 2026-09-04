"""Out-of-scope phrase gate."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag.scope import is_out_of_scope


def test_print_shop_is_out_of_scope():
    assert is_out_of_scope("学校附近哪里有打印店？价格怎么样？")


def test_motto_is_out_of_scope():
    assert is_out_of_scope("南京大学的校训是什么？有什么含义？")


def test_answerable_print_not_blocked():
    assert not is_out_of_scope("在学校里怎么用自助打印？")


def test_answerable_grades_not_blocked():
    assert not is_out_of_scope("成绩在哪里查？")


def test_passport_not_blocked():
    assert not is_out_of_scope("办护照学校能开什么证明？")


def test_give_grade_ranking_is_out_of_scope():
    assert is_out_of_scope("哪个老师最给分？")


def test_library_closing_time_is_out_of_scope():
    assert is_out_of_scope("图书馆晚上几点关门？")

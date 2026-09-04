"""Tests for deterministic campus query rewrite."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag.rewrite import rewrite_query


def test_vpn_paraphrase_adds_official_terms():
    out = rewrite_query("在家能用校园网吗？")
    assert "VPN" in out
    assert "在家能用校园网吗？" in out


def test_unrelated_query_unchanged():
    q = "图书馆怎么续借？"
    assert rewrite_query(q) == q


def test_empty_query():
    assert rewrite_query("  ") == ""


def test_student_id_rewrite():
    out = rewrite_query("学生证丢了怎么补办？")
    assert "学生证补办" in out
    assert "本科生院" in out


def test_passport_does_not_hijack_auth():
    out = rewrite_query("办护照学校能开什么证明？")
    assert "账号激活" not in out
    assert "签证" not in out
    assert "学生身份证明" in out


def test_credit_transfer_rewrite():
    out = rewrite_query("转过去之后原来的课能抵学分吗？")
    assert "学分认定" in out


def test_email_login_rewrite_adds_auth_terms():
    out = rewrite_query("邮箱登录不了怎么办，先查密码还是先查账号状态？")
    assert "统一身份认证" in out
    assert "密码找回" in out


def test_portal_and_app_rewrite_adds_official_terms():
    out = rewrite_query("信息门户和手机 App 都能做哪些事情？")
    assert "统一服务入口" in out
    assert "移动校园 App" in out


def test_finance_rewrite_adds_tuition_and_utility_terms():
    out = rewrite_query("电费欠费断电后还能怎么恢复？")
    assert "宿舍水电费缴纳与查询" in out
    assert "电费" in out


def test_registration_rewrite_adds_registration_terms():
    out = rewrite_query("缴费后如何确认自己已经完成注册？")
    assert "报到注册" in out


def test_leave_and_counseling_rewrite_adds_help_terms():
    out = rewrite_query("请病假和心理咨询分别找谁？")
    assert "本科生请假" in out
    assert "心理健康" in out

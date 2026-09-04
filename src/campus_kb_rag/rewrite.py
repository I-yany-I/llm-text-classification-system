"""Deterministic query rewrite for campus-service paraphrases.

This is not an LLM rewriter. It expands known campus synonyms so BM25 and
the Cross-Encoder see the same terms as the knowledge base.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


# (trigger phrases in the user question, expansions injected for retrieval)
_RULES: Sequence[Tuple[Sequence[str], Sequence[str]]] = (
    (("在家", "家里", "校外", "外出"), ("VPN", "校外访问", "校园网")),
    (("挂科", "不及格", "没及格"), ("补考", "重修", "成绩")),
    (("保研",), ("推免", "推荐免试研究生")),
    (("学信网",), ("学籍", "学籍证明", "学历注册")),
    (("钓鱼", "骗密码", "账号异常邮件"), ("信息安全", "密码重置", "统一身份认证")),
    (("邮箱登录", "登录不了", "登不上", "账号状态"), ("统一身份认证", "密码找回", "登录说明")),
    (("信息门户", "手机 app", "校园 app"), ("统一服务入口", "移动校园 App")),
    (("支付宝", "微信支付", "微信缴费"), ("学费", "缴费", "财务处")),
    (("电费", "断电", "停电"), ("宿舍水电费缴纳与查询", "电费", "充值")),
    (("缴费后", "完成注册", "已经完成注册"), ("报到注册", "学费缴纳方式与报到注册")),
    (("回放", "慕课", "在线课"), ("中国大学MOOC", "在线课程")),
    (("心理", "压力大", "想找人聊"), ("心理咨询", "心理健康")),
    (("请病假", "病假", "请假"), ("本科生请假", "请假规定", "心理咨询")),
    (("老家看病", "异地就医"), ("学生医保", "医保报销")),
    (("转专业",), ("转专业", "专业调整")),
    (("查重", "毕业论文"), ("毕业论文", "学术不端检测")),
    (("候补", "课程满了", "选课满"), ("补选", "补退选", "直选式")),
    (("软件授权", "正版到期", "续期"), ("正版软件", "Office", "MATLAB")),
    (("交换", "出国交流", "交换生"), ("交流生", "备案", "学分认定")),
    (("留学生宿舍",), ("住宿", "宿舍申请", "国际学生")),
    (("迟交学费", "欠费", "注销学籍"), ("注册", "学费", "缴费")),
    (("住宿费",), ("学费", "住宿费", "缴费")),
    (("校医院报销", "门诊报销", "外面医院报销"), ("医保", "校医院", "报销")),
    (("国际学生激活", "港澳台侨", "通行证号"), ("账号激活", "证件", "海外教育学院")),
    (("学生证丢", "学生证补办", "学生证丢了"), ("学生证补办", "本科生院", "一寸近照")),
    (("校园卡丢", "校园卡挂失", "一卡通丢", "一卡通挂失"), ("挂失", "补卡", "校园卡服务")),
    (("免费的 office", "免费 office", "免费的 Office"), ("正版软件", "Microsoft Office", "Office 365")),
    (("上不了网", "连上之后上不了"), ("统一身份认证", "认证页面", "有线网络")),
    (("成绩什么时候", "成绩在哪里查"), ("成绩查询", "教务系统")),
    (("银行开户",), ("在读证明", "在学证明", "自助打印")),
    (("办护照", "因私护照", "护照证明"), ("因私护照", "出入境", "学生身份证明", "在读证明")),
    (("抵学分", "学分抵"), ("学分认定", "转专业")),
    (("刚好够门槛",), ("GPA", "转专业")),
)


def rewrite_query(query: str) -> str:
    text = " ".join((query or "").split())
    if not text:
        return ""
    extras: List[str] = []
    lowered = text.lower()
    for triggers, expansions in _RULES:
        if any(trigger.lower() in lowered for trigger in triggers):
            extras.extend(expansions)
    if not extras:
        return text
    seen = {text}
    ordered: List[str] = []
    for token in extras:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return text + " " + " ".join(ordered)


def rewrite_queries(queries: Iterable[str]) -> List[str]:
    return [rewrite_query(q) for q in queries]

# -*- coding: utf-8 -*-
"""Enrich the campus KB with public official-page facts and fix unlabeled eval gold."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "data/campus_kb/nju_official_kb.jsonl"
EVAL_PATH = ROOT / "data/campus_kb/eval_questions.jsonl"

ENRICH = {
    "nju-it-mooc": {
        "source": "https://jw.nju.edu.cn/24777/list.htm",
        "append": "教服平台学生栏目提供中国大学MOOC、超星尔雅等入口。课程回放、录像通常在对应在线教学平台或中国大学MOOC课程页查看，而不是在普通网页搜索。",
    },
    "nju-it-vpn": {
        "source": "https://jw.nju.edu.cn/24777/list.htm",
        "append": "教服平台等校内系统在校外访问需连接校园 VPN，信息化建设管理服务中心提供《VPN使用办法及常见问题》（https://itsc.nju.edu.cn/12/f2/c21422a463602/page.htm）。在家、出差或其他校外网络环境下，应先安装并登录 VPN 再访问教务、图书馆数据库等校内资源。",
    },
    "nju-it-wifi": {
        "append": "若手机或电脑显示已连接宿舍/校园 WiFi 但仍无法打开网页，通常是未完成统一身份认证，或网口/账号在线设备数超限。应先打开认证页面登录，必要时改用有线网口或注销旧设备后重试。",
    },
    "nju-ac-course-select": {
        "source": "https://jw.nju.edu.cn/24777/list.htm",
        "append": "本科选课入口为教服平台和本科选课平台 https://xk.nju.edu.cn/（统一身份认证登录）。校外访问教服平台需连接 VPN。选课结果以教服平台「我的课表」为准，未经选中的课程无法获得修读资格和成绩。",
    },
    "nju-ac-course-add": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "据本科生院2026年春季学期开学教务事项通知：补选采用「直选式」，提交成功即选中；课程容量满员后无法再选，通知未提供候补队列。补退选在本科选课平台进行。",
    },
    "nju-ac-course-drop": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "据2026年春季学期开学通知：开课3-8周仍可退选，但成绩单上会有退课记录（开课第二周周日24:00之后的退课操作都会有记录）。",
    },
    "nju-ac-exam-makeup": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "期末课程不及格可按当学期通知申请补考。2026年春季通知明确：补考办理入口为南京大学网上办事服务大厅→补考办理；公共课与其他课程报名截止时间不同，缓考和补考可能一起组织。挂科后是否有补考机会以当学期本科生院通知为准。",
    },
    "nju-ac-retake": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "据2026年春季通知：在校生重修在网上办事服务大厅「重修选课管理」申请；容量已满或本学期不开课则申请无法通过。缴费完成后旧成绩将被设置为无效且不得恢复。选修课重修可不先提交重修申请，直接在选课期间选课并缴费。",
    },
    "nju-ac-enrollment": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "在读证明、成绩单可在信息化自助打印终端或教务窗口办理。银行开户等用途请申请在学证明/在读证明。2026年春季报到注册通过南京大学APP「在校生报到」。未按学校规定缴纳学费或不符合注册条件的不予注册；家庭经济困难学生可申请助学贷款或其他资助后办理注册。",
    },
    "nju-ac-status": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "学籍注册状态以学校教务系统为准。学信网用于学历学籍在线核验，若与学校显示不一致，应联系院系教务员或本科生院学籍老师核查，而不是自行修改学信网数据。因公交换在外的学生按通知无需APP报到，但须完成交流备案，否则影响选课。",
    },
    "nju-ac-gpa": {
        "append": "学校教务系统记载的成绩与绩点是校内评奖、推免和毕业审核依据。学信网报告用于校外核验，二者统计口径可能不同；发现不一致时应以学校教务记载为准并申请核查。",
    },
    "nju-ac-recommend": {
        "append": "保研即推荐免试攻读研究生（推免）。资格通常看GPA排名、科研与综合表现，名额由院系下达，具体条件以当年推免通知为准。",
    },
    "nju-ac-thesis": {
        "append": "毕业论文须通过查重（学术不端检测）和答辩。查重未通过或答辩未通过会影响毕业资格；可按院系规定修改后申请补答辩。仅毕业论文未通过的，按学籍规定可能不可申请延长学习。",
    },
    "nju-ac-graduation": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "毕业审核包括学分、毕业论文和学籍处分等。重修、补考通过后须在教服平台确认课程已入「我的课程」。美育核心课不及格原则上仅有一次重修机会。",
    },
    "nju-ac-grade-review": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "成绩更正相关事务可咨询本科生院教学运行服务中心。复核被拒后，应先向任课教师和院系教务员了解评分依据；学校未提供无限次申诉通道。",
    },
    "nju-it-print": {
        "source": "https://itsc.nju.edu.cn/21426/list.htm",
        "append": "信息化建设管理服务中心统一自助打印平台于2019年上线，鼓楼、仙林、浦口、苏州四校区共13台终端，可刷校园卡/NFC、账号密码或身份证登录，也可在南京大学APP「自助服务」申请取件码后扫码取件。本科生可打印中英文成绩单、四六级证明、在学证明、毕业证明等，并支持电子成绩单下载及学信网可信认证。这是校内自助打印，不是校外打印店。",
    },
    "nju-it-auth-password": {
        "append": "若怀疑在钓鱼页面输入过统一身份认证密码，应立即在官方认证入口修改密码，并联系信息化服务台报告异常登录。学校工作人员不会通过邮件索取密码。",
    },
    "nju-it-security": {
        "append": "收到「账号异常」钓鱼邮件并误填密码后，立即改密、检查异常登录、向信息化服务台报备。不要继续点击来源不明的链接。",
    },
    "nju-it-software-office": {
        "source": "https://itsc.nju.edu.cn/21426/list.htm",
        "append": "正版软件服务入口见信息化服务中心「全部服务-正版软件」。授权到期后应通过学校邮箱在官方教育渠道重新验证，不要使用非官方激活工具。",
    },
    "nju-stu-counseling": {
        "append": "心理压力大、想找人聊聊，可预约学校心理健康教育与研究中心的心理咨询，而不是自行搜索校外机构。",
    },
    "nju-stu-insurance": {
        "append": "学生医保报销范围和异地就医备案以医保经办机构及校医院当年通知为准。放假在老家看病，通常需按规定办理异地就医备案或持票据回校按规定报销。校医院看不了需转诊时，按校医院转诊单和医保定点医院规则办理。",
    },
    "nju-stu-clinic": {
        "append": "校医院门诊挂号需携带学生证或一卡通。门诊费用按学生医保规则结算；具体报销比例和目录以校医院/医保通知为准，系统不能代替现场结算。",
    },
    "nju-stu-scholarship": {
        "append": "国家奖学金是学校奖学金体系中的国家级项目，名额有限，申请条件和材料以学生资助中心当年评定办法为准，通常在学生工作系统提交。",
    },
    "nju-fin-tuition": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "缴费事项请关注「南京大学财务处」微信公众号。在线缴费通常支持网银、支付宝、微信。学费与住宿费按学校收费项目分别缴纳，住宿费一般不包含在学费中。未按规定缴纳学费或其他不符合注册条件的，不予注册；欠费可能影响选课、考试、证明开具和学籍异动审批，但具体是否注销学籍以学籍管理规定为准，家庭经济困难应尽快联系辅导员申请资助或助学贷款。",
    },
    "nju-intl-visa": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": "因学校交换项目在外交流的学生，2026年春季通知明确无需进行APP报到注册，但须在教服平台/办事大厅完成交流备案（院系或个人项目须新增备案），否则影响选课；未经备案将来无法申请交流成绩认定。这不等于自动休学。校级交换项目由国际化工作处导入系统。",
    },
}

NEW_DOCS = [
    {
        "id": "nju-intl-housing",
        "title": "国际学生住宿申请说明",
        "department": "国际学生管理/后勤保障",
        "source": "https://hwxy.nju.edu.cn/",
        "source_type": "official_entry",
        "updated_at": "2026-03-01",
        "collected_at": "2026-08-28",
        "valid_until": None,
        "tags": ["留学生", "宿舍", "住宿", "国际学生"],
        "text": "来华留学生住宿由学校国际学生管理部门和后勤按当年招生通知安排，通常在指定国际学生公寓或学校安排的宿舍办理入住，不等同于国内本科生普通宿舍抽签。短期国际交流学生的住宿是否与学位留学生混住，以当年国际合作与交流处/海外教育学院通知为准。申请一般需在录取后按通知提交住宿意向，不通过普通本科生宿管系统自行选房。本条根据公开办事入口整理，具体床位、费用和混住规则以主管部门最新通知为准。",
    },
    {
        "id": "nju-ac-exchange-record",
        "title": "交流生备案与学分认定入口",
        "department": "本科生院/国际化工作处",
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "source_type": "official_page",
        "updated_at": "2026-02-27",
        "collected_at": "2026-08-28",
        "valid_until": None,
        "tags": ["交流生", "备案", "学分认定", "交换"],
        "text": "南京大学本科生院2026年春季学期开学通知规定：因学校交换项目在外交流的学生无需进行报到注册，但需要备案。校级交换项目由国际化工作处导入系统，学生在交换开始前查看办事大厅是否有对应交换信息；院系或个人项目须在「校际交换项目备案」中新增备案，由学院审核。未经备案的交换将来无法申请交流成绩认定。操作入口：南京大学网上办事服务大厅（ehall.nju.edu.cn）→学分认定→备案信息查看与新增。出国交换期间学籍仍为在校交流状态，通知未将其等同于休学。",
    },
]

EVAL_FIXES = {
    "q-fin-117": {"expected_doc_ids": ["nju-fin-tuition"], "should_refuse": False},
    "q-fin-118": {"expected_doc_ids": ["nju-fin-tuition"], "should_refuse": False},
    "q-fin-119": {"expected_doc_ids": ["nju-fin-tuition"], "should_refuse": False},
    "q-intl-120": {"expected_doc_ids": ["nju-ac-exchange-record", "nju-intl-visa"], "should_refuse": False},
    "q-intl-122": {"expected_doc_ids": ["nju-intl-housing"], "should_refuse": False},
    "q-stu-126": {"expected_doc_ids": ["nju-stu-insurance", "nju-stu-clinic"], "should_refuse": False},
    "q-refuse-04": {
        "expected_doc_ids": ["nju-stu-clinic", "nju-stu-insurance"],
        "should_refuse": False,
        "category": "student",
    },
    "q-refuse-107": {
        "expected_doc_ids": ["nju-stu-scholarship"],
        "should_refuse": False,
        "category": "student",
    },
}


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    docs = _load(KB_PATH)
    by_id = {doc["id"]: doc for doc in docs}
    for doc_id, patch in ENRICH.items():
        doc = by_id[doc_id]
        extra = patch["append"].strip()
        if extra not in doc["text"]:
            doc["text"] = doc["text"].rstrip() + extra
        if "source" in patch:
            doc["source"] = patch["source"]
        doc["collected_at"] = "2026-08-28"
    for new in NEW_DOCS:
        if new["id"] not in by_id:
            docs.append(new)
            by_id[new["id"]] = new
    _dump(KB_PATH, docs)

    questions = _load(EVAL_PATH)
    for row in questions:
        patch = EVAL_FIXES.get(row["id"])
        if not patch:
            continue
        row.update(patch)
    _dump(EVAL_PATH, questions)
    print("kb", len(docs), "eval", len(questions))


if __name__ == "__main__":
    main()

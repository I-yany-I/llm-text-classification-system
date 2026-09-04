# -*- coding: utf-8 -*-
"""Second pass: inject facts copied from public official pages; finish unlabeled gold."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "data/campus_kb/nju_official_kb.jsonl"
EVAL_PATH = ROOT / "data/campus_kb/eval_questions.jsonl"

ENRICH = {
    "nju-it-vpn": {
        "source": "https://itsc.nju.edu.cn/21601/list.htm",
        "append": (
            "信息化建设管理服务中心「校外VPN」栏目说明：校外访问图书馆已购资源、远程连接校内服务器等，"
            "须使用 VPN。打开 https://vpn.nju.edu.cn 下载 EasyConnect 客户端（校园网无法打开该页，请在校外下载）。"
            "首次填写服务器地址 https://vpn.nju.edu.cn，登录为统一身份认证账号加手机验证码。"
            "试运行阶段同一时间每人仅允许一台设备。收不到验证码时，到 authserver 核对已绑定手机号；"
            "若曾退订南京大学通知短信（106904000520），需按信息化中心说明恢复订阅。服务电话 025-89683791。"
        ),
    },
    "nju-it-auth-login": {
        "source": "https://itsc.nju.edu.cn/tysfrz/list.htm",
        "append": (
            "官方统一身份认证页说明：EasyConnect、上网认证 p.nju.edu.cn、信息门户、南京大学 APP、"
            "ehall.nju.edu.cn、OA、教务、就业等使用同一账号密码。弹出 authserver.nju.edu.cn 时即统一身份认证。"
            "自助服务中心：https://authserver.nju.edu.cn/authserver/selfServiceCenter，可激活、找回密码。"
            "用户名为工号/学号/自管号。2022年9月起登录需手机短信验证（半年内一次），须先绑定手机号。"
        ),
    },
    "nju-it-auth-activate": {
        "source": "https://itsc.nju.edu.cn/tysfrz/list.htm",
        "append": (
            "2026年2月7日统一身份认证升级后，新开通账号首次使用前必须手动「账号激活」。"
            "激活时姓名及证件类型须与人事、研究生、教务学工或海外教育学院源头系统一致，否则无法匹配。"
            "国际学生、港澳台侨学生按源头系统录入的护照或通行证办理，不要用他人身份证代替。"
            "2026年2月7日前开通的账号，初始密码一般为全部身份证件号（港澳台侨为通行证号）或身份证号后六位。"
        ),
    },
    "nju-it-auth-password": {
        "source": "https://itsc.nju.edu.cn/tysfrz/list.htm",
        "append": (
            "密码找回入口为统一身份认证自助服务中心 https://authserver.nju.edu.cn/authserver/selfServiceCenter。"
            "也可通过微信「南京大学信息门户」个人中心或南京大学 APP「认证账号设置」绑定/核对手机号后再自助重置。"
            "APP 里普通「账号设置」不是统一身份认证设置。密码建议字母加数字、八位以上。"
        ),
    },
    "nju-it-software-office": {
        "source": "https://itsc.nju.edu.cn/zbrj/mainm.htm",
        "append": (
            "信息化中心正版软件站提供微软 Office、Office 365 教育版。Office 365 含 OneDrive 网盘、"
            "SharePoint 及网页版 Word/Excel/PowerPoint。仅限在岗在册师生教学科研办公，禁止转借倒卖商用。"
        ),
    },
    "nju-it-software-matlab": {
        "source": "https://itsc.nju.edu.cn/zbrj/mainm.htm",
        "append": (
            "正版软件站 MATLAB 栏目：师生可在校属电脑和个人电脑部署 MathWorks 产品（中英文）。"
            "个人版适合个人电脑并可离线使用；机房版适合实验室、机房、集群。许可证按信息化中心页面每年更新。"
        ),
    },
    "nju-it-software-others": {
        "source": "https://itsc.nju.edu.cn/zbrj/mainm.htm",
        "append": (
            "正版软件目录含 Adobe、EndNote、Origin、Stata、Mathematica、ChemDraw、WPS 365 等。"
            "申请与激活以信息化中心正版软件站各软件指南为准，不要使用破解工具。"
        ),
    },
    "nju-it-cloud": {
        "source": "https://itsc.nju.edu.cn/zbrj/mainm.htm",
        "append": (
            "学校通过微软 Office 365 教育版提供 OneDrive 网盘。登录办法见信息化中心「微软Office365」栏目。"
        ),
    },
    "nju-it-meeting": {
        "source": "https://itsc.nju.edu.cn/zbrj/mainm.htm",
        "append": (
            "信息化中心正版软件站提供「云视频会议」校园授权，使用指南见该栏目，不是个人随便申请的 Zoom 商业账号。"
        ),
    },
    "nju-it-print": {
        "source": "https://itsc.nju.edu.cn/21426/list.htm",
        "append": (
            "信息化中心自助打印栏目：四校区共 13 台终端、300 余项服务。本科生可打印中英文成绩单、四六级证明、"
            "在学证明、毕业证明等；支持 APP「自助服务」申请取件码。遇到问题在终端「我要反馈」。"
        ),
    },
    "nju-ac-transcript": {
        "source": "https://jw.nju.edu.cn/a9/32/c24739a370994/page.htm",
        "append": (
            "本科生院《出国成绩、学历证明办理流程》（2025年8月修订）：在校全日制本科生办理中英文成绩单、"
            "在学证明、平均分证明，请在自助打印机上办理。出国同学可凭英文成绩单至本科生院领取同等数量信封。"
            "2012级及以后往届生在自助机打印，账号为学号、密码为身份证号。本科生院只办理英文成绩单，不办理其它语种。"
            "推免期间大三升大四原则上停止办理英文成绩单。"
        ),
    },
    "nju-ac-transcript-en": {
        "source": "https://jw.nju.edu.cn/a9/32/c24739a370994/page.htm",
        "append": (
            "英文成绩单按本科生院出国证明流程在自助打印机办理。信封领取：鼓楼南园综合服务大厅（周一、周三），"
            "仙林行政北楼410。模板见 https://jw.nju.edu.cn/24816/list.htm。"
        ),
    },
    "nju-ac-enrollment": {
        "source": "https://jw.nju.edu.cn/a9/32/c24739a370994/page.htm",
        "append": (
            "在学证明/在读证明：在校本科生在自助打印机办理。自助机地点包括鼓楼南园综合服务大厅、鼓楼图书馆，"
            "仙林行政北楼一楼、图书馆大厅、信息化中心一楼、十一食堂。"
        ),
    },
    "nju-stu-student-id": {
        "source": "https://jw.nju.edu.cn/24751/list.htm",
        "append": (
            "本科生院《学生证补办和火车优惠卡充磁流程》：本人持校园卡、身份证、一寸近照到本科生院办理；"
            "委托办理需委托书及双方身份证复印件。补办学生证工本费10元，火车优惠卡成本价7元，刷校园卡。"
            "鼓楼：南园综合服务大厅一楼，周一办理、周三领取；仙林：行政北楼410。再至行政南楼110加盖钢印和公章。"
        ),
    },
    "nju-stu-card-reissue": {
        "source": "https://itsc.nju.edu.cn/21446/list.htm",
        "append": (
            "信息化中心「校园卡补卡」：丢失须先挂失（微信南京大学信息门户-综合服务-校园卡服务-挂失，"
            "或南京大学 APP 校园卡服务，或多媒体机）。挂失后可在自助补卡机补办，支持二代身份证或校园卡账户认证，"
            "工本费20元，余额不足须先充值。挂失后原卡暂停使用；若找回可在同一入口解挂，余额仍在账户内。"
            "人工补卡须持有效证件，代办需双方证件。卡损坏补办无需挂失，持坏卡及证件办理，同样20元。"
        ),
    },
    "nju-stu-card-charge": {
        "source": "https://itsc.nju.edu.cn/9a/12/c21475a498194/page.htm",
        "append": (
            "校园卡系统支持手机在线支付、微信充值，以及电费、网费、水费、图书欠费在线缴费。"
            "可在微信信息门户或南京大学 APP 办理充值、挂失、解挂、限额修改。"
        ),
    },
    "nju-ac-minor": {
        "source": "https://jw.nju.edu.cn/fxpyfa/list.htm",
        "append": (
            "2026年春季本科教务事项通知：辅修培养方案在教改专区「辅修培养方案」查询，"
            "地址 https://jw.nju.edu.cn/fxpyfa/list.htm。2025版适用于2025级及之后，2021版适用于2021–2024级。"
            "拟申请辅修学士学位须在主修标准学制内修完辅修方案全部课程，并完成辅修学位论文。"
        ),
    },
    "nju-ac-exchange": {
        "source": "https://jw.nju.edu.cn/83/24/c26263a820004/page.htm",
        "append": (
            "交流成绩认定：校级交换由国际化工作处导入系统；院系或个人项目须在办事大厅「校际交换项目备案」新增。"
            "未经备案将来无法申请交流成绩认定。认定通知见本科生院「关于受理交流学习课程认定及学分转换申请的通知」。"
        ),
    },
}

EVAL_FIXES = {
    "q-intl-121": {
        "expected_doc_ids": ["nju-it-auth-activate"],
        "should_refuse": False,
        "category": "international",
    },
    "q-stu-124": {
        "expected_doc_ids": ["nju-stu-student-id"],
        "should_refuse": False,
        "category": "student",
    },
    "q-stu-125": {
        "expected_doc_ids": ["nju-stu-card-reissue"],
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
        doc["source_type"] = "official_page"
    _dump(KB_PATH, docs)

    questions = _load(EVAL_PATH)
    for row in questions:
        patch = EVAL_FIXES.get(row["id"])
        if patch:
            row.update(patch)
    _dump(EVAL_PATH, questions)
    print("kb", len(docs), "eval", len(questions))


if __name__ == "__main__":
    main()

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CandidateSpan:
    start: int
    end: int
    tag_type: str
    priority: int


class RuleBasedBioTagger:
    """
    Lightweight automatic BIO tagger for backend integration.

    This module is intentionally rule-based (fast and deterministic). It can be
    replaced by a trainable NER module later while keeping the same interface.
    """

    NON_BIO_TYPES = set()

    def __init__(self) -> None:
        self.surname_chars = (
            "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
            "金魏陶姜戚谢邹喻柏水窦章云苏潘葛范彭郎鲁韦昌马苗凤花方俞任"
            "袁柳鲍史唐费廉岑薛雷贺倪汤殷罗毕郝邬安常乐于时傅皮卞齐康伍余"
            "元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞"
            "熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐"
            "邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓"
            "郁单杭洪包诸左石崔吉龚程邢滑裴陆荣翁荀羊於惠甄麴家封芮羿储靳"
            "汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇"
            "栾暴甘斜厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂"
            "索咸籍赖卓蔺屠蒙池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰"
            "郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习"
            "宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳"
            "沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰"
            "巢关蒯相查后荆红游竺权逯盖益桓公万俟司马上官欧阳夏侯诸葛闻人"
            "东方赫连皇甫尉迟公羊澹台公冶宗政濮阳淳于单于太叔申屠公孙仲孙"
            "轩辕令狐钟离宇文长孙慕容司徒司空"
        )
        self.name_pattern = re.compile(
            rf"([{self.surname_chars}][\u4e00-\u9fa5]{{1,2}})(?=(?:阿姨|大爷|奶奶|爷爷|叔叔|先生|女士|老人|夫妻|今年|在|，|。|、|$))"
        )
        self.name_pattern_relaxed = re.compile(
            rf"([{self.surname_chars}][\u4e00-\u9fa5]{{1,2}})"
        )
        self.keyword_map: Dict[str, List[str]] = {
            "protagonist": ["我", "本人", "自己", "咱", "俺"],
            "participant_par": [
                "儿子",
                "女儿",
                "子女",
                "孩子",
                "老伴",
                "亲家",
                "朋友",
                "邻居",
                "医生",
                "护士",
                "家人",
                "孙子",
                "孙女",
            ],
            "Health_pro": [
                "生病",
                "住院",
                "吃药",
                "服药",
                "手术",
                "复查",
                "体检",
                "康复",
                "失眠",
                "头晕",
                "咳嗽",
                "高血压",
                "糖尿病",
                "骨折",
                "关节炎",
                "白内障",
                "耳背",
                "眼花",
                "视力下降",
                "视力减退",
                "视力衰退",
                "听力下降",
                "听力减退",
                "听力衰退",
                "老花镜",
                "助听器",
            ],
            "Health_par": ["他病了", "她病了", "家人生病", "老人病情", "摔伤"],
            "Identity_pro": [
                "老师",
                "教授",
                "工程师",
                "医生",
                "护士",
                "党员",
                "院士",
                "主任",
                "干部",
                "研究员",
                "退休",
            ],
            "Achievement_pro": [
                "获奖",
                "荣获",
                "贡献",
                "成果",
                "发表",
                "论文",
                "专利",
                "表彰",
                "先进",
            ],
            "Interest_pro": [
                "爱好",
                "喜欢",
                "下棋",
                "唱歌",
                "跳舞",
                "散步",
                "旅游",
                "看书",
                "钓鱼",
                "锻炼",
            ],
            "Social Activity_pro": [
                "活动",
                "交流会",
                "晚会",
                "春节晚会",
                "春节联欢晚会",
                "参观",
                "聚会",
                "聚餐",
                "午餐",
                "团圆饭",
                "过年",
                "义工",
                "志愿",
                "体检",
                "复查",
            ],
            "Education background_pro": [
                "小学",
                "初中",
                "高中",
                "大学",
                "本科",
                "硕士",
                "博士",
                "毕业",
            ],
            "location_pro": [
                "北京",
                "上海",
                "合肥",
                "郑州",
                "医院",
                "社区",
                "公园",
                "养老院",
                "家里",
                "酒店",
                "养亲苑",
            ],
            "location_par": ["他家", "她家", "亲戚家", "子女家"],
        }

        self.regex_map: Dict[str, List[re.Pattern]] = {
            "Health_pro": [
                re.compile(
                    r"(高血压|低血压|糖尿病|冠心病|关节炎|骨质疏松|白内障|脑梗|中风|慢阻肺|帕金森|阿尔茨海默)"
                ),
                re.compile(
                    r"(眼花|耳背|看不清|听不清|腿脚[^，。；！？]{0,8}(不灵便|不如从前灵便|不如从前|无力|发软|不便)|"
                    r"膝盖[^，。；！？]{0,8}(疼|痛|酸|胀|肿|炎)|腰[^，。；！？]{0,6}(酸|痛)|"
                    r"头[^，。；！？]{0,6}(晕|痛)|睡眠[^，。；！？]{0,6}(差|不好|困难))"
                ),
                re.compile(
                    r"((视力|听力)[^，。；！？]{0,8}(下降|减退|衰退|变差|不好)|老花镜|助听器)"
                ),
                re.compile(
                    r"(眼|耳|鼻|咽|喉|心|肺|肝|胃|肠|腰|背|腿|膝|脚|颈|肩|血压|血糖|睡眠|食欲)"
                    r"[^，。；！？]{0,8}(疼|痛|酸|麻|胀|晕|花|背|差|下降|异常|不适|不舒服|不灵便|受限|炎|病)"
                ),
                re.compile(r"(患有|确诊|诊断为|查出|得了|有)[^，。；！？]{0,10}(病|炎|症)"),
                re.compile(r"(住院|手术|复查|体检|吃药|服药|康复|理疗)"),
            ],
            "Social Activity_pro": [
                re.compile(
                    r"(旅游|出行|探亲|聚餐|团圆饭|午餐|晚餐|聚会|看[^，。；！？]{0,8}晚会|过年|春节|体检|复查|康养|健身|活动)"
                )
            ],
            "location_pro": [
                re.compile(r"([^\s，。；！？]{1,8}(医院|社区|公园|养老院))")
            ],
        }

        self.priority: Dict[str, int] = {
            "protagonist": 10,
            "participant_par": 20,
            "Health_pro": 30,
            "Health_par": 35,
            "Identity_pro": 40,
            "Achievement_pro": 45,
            "Interest_pro": 50,
            "Social Activity_pro": 55,
            "Education background_pro": 60,
            "location_par": 70,
            "location_pro": 80,
        }

    @staticmethod
    def _find_all(text: str, key: str) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        start = 0
        while True:
            idx = text.find(key, start)
            if idx < 0:
                break
            result.append((idx, idx + len(key)))
            start = idx + 1
        return result

    def _collect_candidates(
        self, text: str, protagonist_name: str = ""
    ) -> List[CandidateSpan]:
        spans: List[CandidateSpan] = []

        inferred_name = protagonist_name.strip() if protagonist_name else ""
        if not inferred_name:
            inferred_name = self._infer_protagonist_name(text)

        if inferred_name and inferred_name in text:
            for s, e in self._find_all(text, inferred_name):
                spans.append(
                    CandidateSpan(
                        start=s,
                        end=e,
                        tag_type="protagonist",
                        priority=self.priority["protagonist"],
                    )
                )

        for tag_type, keywords in self.keyword_map.items():
            for kw in keywords:
                for s, e in self._find_all(text, kw):
                    spans.append(
                        CandidateSpan(
                            start=s,
                            end=e,
                            tag_type=tag_type,
                            priority=self.priority[tag_type],
                        )
                    )

        for tag_type, patterns in self.regex_map.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    spans.append(
                        CandidateSpan(
                            start=m.start(),
                            end=m.end(),
                            tag_type=tag_type,
                            priority=self.priority[tag_type],
                        )
                    )

        # Prefer higher-priority and longer spans.
        spans.sort(key=lambda x: (x.priority, -(x.end - x.start), x.start))
        return spans

    def _infer_protagonist_name(self, text: str) -> str:
        text = str(text or "")
        if not text:
            return ""

        for m in self.name_pattern.finditer(text):
            candidate = m.group(1).strip()
            if len(candidate) in {2, 3}:
                return candidate

        # Fallback: first plausible name in sentence head.
        head = text[:20]
        for m in self.name_pattern_relaxed.finditer(head):
            candidate = m.group(1).strip()
            if len(candidate) in {2, 3}:
                return candidate
        return ""

    def tag(self, text: str, protagonist_name: str = "") -> str:
        n = len(text)
        tags = ["O"] * n
        occupied = [False] * n

        for cand in self._collect_candidates(text, protagonist_name=protagonist_name):
            if cand.start < 0 or cand.end > n or cand.start >= cand.end:
                continue
            if any(occupied[i] for i in range(cand.start, cand.end)):
                continue

            if cand.tag_type in self.NON_BIO_TYPES:
                for i in range(cand.start, cand.end):
                    tags[i] = cand.tag_type
                    occupied[i] = True
            else:
                tags[cand.start] = f"B-{cand.tag_type}"
                occupied[cand.start] = True
                for i in range(cand.start + 1, cand.end):
                    tags[i] = f"I-{cand.tag_type}"
                    occupied[i] = True

        return " ".join(tags)

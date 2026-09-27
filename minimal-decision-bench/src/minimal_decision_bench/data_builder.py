from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from statistics import fmean

# 模板是切分的语义单位：train/dev/test 必须按模板隔离，而不是按「模板+后缀」的文本隔离。
# 否则测试集的文本虽然字面没出现过，但底层模板全在训练集里见过，指标仍然是记忆。
#
# 每类 30 条，分三个难度层，各 10 条：
#   explicit  含本类专属词，规则基线能命中        —— 规则的舒适区
#   conflict  含**其它类**的词，但诉求是本类      —— 需要区分「诉求」与「背景」
#   synonym   不含任何类别词（含口语转述）        —— 需要语义泛化
# 三桶按 round-robin 交织后切分，保证 train/dev/test 拿到相同比例的难度层。
INTENT_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "billing": {
        "explicit": [
            "我被重复扣费了，退款什么时候到？",
            "发票抬头有误，帮我改一下。",
            "账单金额不对，想核对明细。",
            "套餐续费金额异常，请处理。",
            "发票能改成公司抬头吗？",
            "这个月账单多扣了一笔。",
            "退款已经一周了还没到账。",
            "续费自动扣款能关掉吗？",
            "订单显示已支付但服务没开通。",
            "想要一份对账单，怎么申请？",
        ],
        "conflict": [
            "支付的时候报错了，钱却扣了两次。",
            "支付失败的订单，钱还是被划走了。",
            "这个页面一直转圈，我的钱到底去哪了。",
            "系统报错之后，那笔钱就不见了。",
            "提交超时了，但钱已经出去了。",
            "弹了个错误提示，然后我的余额就少了。",
            "点了两次没反应，结果划走两笔钱。",
            "接口一直 500，付款到底成没成功？",
            "加载失败之后，我账户里的钱少了。",
            "一直卡在处理中，钱已经扣走了。",
        ],
        "synonym": [
            "我这个月被多收了一笔钱。",
            "那笔钱什么时候能还给我？",
            "纸质凭证能寄给我吗？",
            "我想看一下这个月的消费明细。",
            "自动续的那笔能停掉吗？",
            "这笔钱我不该付，能撤回来吗？",
            "你们多收了我一次。",
            "收据上的公司名写错了。",
            "就那个钱的事，你们处理一下。",
            "钱花得不明不白，给我查查。",
        ],
    },
    "support": {
        "explicit": [
            "API 一直报 500，线上不可用。",
            "登录后白屏并提示 error。",
            "任务执行超时，workflow 卡住。",
            "同步接口返回 timeout，影响生产。",
            "上传附件一直失败。",
            "页面加载很慢，经常转圈。",
            "消息推送收不到。",
            "导出报表时报错。",
            "登录时提示密码错误，但我没改过。",
            "数据同步延迟了好几个小时。",
        ],
        "conflict": [
            "续费之后系统就一直报错。",
            "账单页面白屏，看不到内容。",
            "发票下载功能坏了，点不开。",
            "报价页面加载不出来。",
            "账单刷新后接口全部 500。",
            "退款按钮点了没反应。",
            "采购流程走到一半卡住了。",
            "续费之后登录就进不去了。",
            "发票页面一直转圈打不开。",
            "价格表点开是空的。",
        ],
        "synonym": [
            "这个功能突然不能用了。",
            "点进去就转不出来。",
            "数据对不上，两边显示的不一样。",
            "我发的东西对方看不到。",
            "保存之后刷新就没了。",
            "系统像死了一样，点什么都没动静。",
            "表格里少了好几行数据。",
            "这个按钮点了没效果。",
            "突然就不好使了，你们看看。",
            "啥也点不动，急。",
        ],
    },
    "sales": {
        "explicit": [
            "企业版套餐怎么报价？",
            "想升级到年付方案，有优惠吗？",
            "请发一份采购报价单。",
            "团队版和企业版差异是什么？",
            "有没有适合小团队的方案？",
            "年付和月付价格差多少？",
            "想了解私有化部署怎么收费。",
            "能开试用账号吗？",
            "批量采购有折扣吗？",
            "你们支持定制开发吗？",
        ],
        "conflict": [
            "报价页面打不开，能直接发我价格吗？",
            "试用申请报错，怎么开通？",
            "下单一直失败，能不能人工下单？",
            "那张表下载不了，直接说个数吧。",
            "升级按钮点不动，帮我升级一下。",
            "方案介绍页白屏，能发个文档吗？",
            "询价表单超时，我直接问：多少钱？",
            "团队版页面打不开，介绍下差异。",
            "采购入口报错，我先问下批量价。",
            "试用页面加载失败，还有别的试用方式吗？",
        ],
        "synonym": [
            "我们公司想买，大概什么价位？",
            "想找人对接一下商务合作。",
            "这种规模的话一年要花多少？",
            "能不能先给我们开个账号试试？",
            "我们人多，能不能便宜点？",
            "想了解下你们怎么卖。",
            "老板让我对接一下，怎么谈？",
            "后续扩容的话成本怎么算？",
            "就是那个，你们这个怎么卖？",
            "我们想搞一套，怎么弄？",
        ],
    },
    "general": {
        "explicit": [
            "这个功能怎么用，给个快速步骤。",
            "你们有中文文档吗？",
            "最近更新了哪些功能？",
            "账号昵称在哪里改？",
            "怎么修改绑定手机号？",
            "账号可以多人共用吗？",
            "有没有操作手册可以下载？",
            "怎么导出我的数据？",
            "能不能设置消息免打扰？",
            "在哪里查看历史记录？",
        ],
        "conflict": [
            "报错信息一般在哪里看？",
            "账单页面从哪个入口进？",
            "发票模板能自己改吗？",
            "退款规则写在哪份文档里？",
            "价格在哪里能查到？",
            "采购要走什么流程，有说明吗？",
            "超时记录在哪里下载？",
            "白屏的时候怎么看日志？",
            "试用到期后会怎样？",
            "套餐变更的入口在哪？",
        ],
        "synonym": [
            "我该怎么设置头像？",
            "能不能换个登录方式？",
            "手机和电脑上的东西会同步吗？",
            "有没有新手引导？",
            "我想把账号注销掉。",
            "能不能换个语言？",
            "通知太多了，怎么关掉？",
            "你们的服务时间是几点到几点？",
            "那个…就是那个东西怎么弄来着？",
            "不太会用，有没有人教一下",
        ],
    },
}

SUBTYPE_ORDER: tuple[str, ...] = ("explicit", "conflict", "synonym")

SUFFIXES: tuple[str, ...] = ("", " 请尽快处理。", " 比较着急。", " 影响生产。")

# 数据集很小，dev/test 需要足够模板才能给出稳定指标，因此用 60/20/20 而非 70/15/15。
SPLIT_RATIOS: tuple[float, float, float] = (0.6, 0.2, 0.2)

MIN_TEMPLATES_PER_INTENT = 3

ESCALATION_COMPLEXITY_THRESHOLD = 0.68

ESCALATION_KEYWORD = "影响生产"

_COMPLEXITY_RANGE: dict[str, tuple[float, float]] = {
    "support": (0.58, 0.95),
    "billing": (0.35, 0.80),
    "sales": (0.28, 0.72),
    "general": (0.08, 0.46),
}


def strip_suffix(text: str) -> str:
    """去掉后缀，还原出模板。"""
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def enumerate_templates() -> dict[str, list[str]]:
    """全部模板，按 intent 分组。

    组内按「难度层 round-robin」交织（explicit, conflict, synonym, explicit, ...），
    这样按位置切分时每个 split 都能拿到相同比例的难度层。
    """
    return {
        intent: _interleave(subtypes)
        for intent, subtypes in INTENT_TEMPLATES.items()
    }


def subtype_of(intent: str, template: str) -> str:
    for name, templates in INTENT_TEMPLATES[intent].items():
        if template in templates:
            return name
    return "unknown"


def enumerate_texts() -> dict[str, list[str]]:
    """全部唯一文本（模板 × 后缀）。仅用于统计与测试。"""
    return {
        intent: [tpl + suffix for tpl in templates for suffix in SUFFIXES]
        for intent, templates in enumerate_templates().items()
    }


def split_templates(
    templates_by_intent: dict[str, list[str]] | None = None,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """按**模板**切分，保证三个 split 的模板集合两两不相交。

    这是防泄漏的关键。按「模板+后缀」切分是不够的：测试集的文本字面没出现过，
    但底层模板仍在训练集里，模型只需要认模板就能拿满分。
    """
    templates_by_intent = templates_by_intent or enumerate_templates()
    rng = random.Random(seed)
    train: dict[str, list[str]] = {}
    dev: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    for intent, templates in templates_by_intent.items():
        pool = list(templates)
        # 只在同一难度层内部打乱，保留 round-robin 的分层结构
        pool = _shuffle_within_subtype(intent, pool, rng)
        n_train, n_dev, _ = _split_sizes(len(pool), ratios)
        train[intent] = pool[:n_train]
        dev[intent] = pool[n_train : n_train + n_dev]
        test[intent] = pool[n_train + n_dev :]
    return train, dev, test


def build_dataset(
    seed: int = 42,
    reps_per_text: int = 3,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> tuple[list[dict], list[dict], list[dict]]:
    """先按模板切分，再在各自 split 内展开「模板 × 后缀」并扩充样本。"""
    train_t, dev_t, test_t = split_templates(ratios=ratios, seed=seed)
    rng = random.Random(seed)
    train = _expand(train_t, reps_per_text, rng, "train")
    dev = _expand(dev_t, reps_per_text, rng, "dev")
    test = _expand(test_t, reps_per_text, rng, "test")
    return train, dev, test


def build_folds(n_folds: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    """按 (intent × 难度层) 分层，把模板轮转分配到 n_folds 折。

    分层是必须的：如果某一折恰好全是同义词句，它的难度就与其它折不可比，
    折间方差会失真。这里保证每折都同时包含四类 intent 与三个难度层。

    120 条模板 ÷ 5 折 = 每折 24 条，且每个 (intent, 难度层) 桶贡献 2 条。
    """
    rng = random.Random(seed)
    buckets: list[tuple[str, list[str]]] = []
    for intent, subtypes in INTENT_TEMPLATES.items():
        for layer in SUBTYPE_ORDER:
            pool = sorted(subtypes.get(layer, []))
            rng.shuffle(pool)
            buckets.append((intent, pool))

    folds: list[dict[str, list[str]]] = [
        {intent: [] for intent in INTENT_TEMPLATES} for _ in range(n_folds)
    ]
    for intent, pool in buckets:
        for i, template in enumerate(pool):
            folds[i % n_folds][intent].append(template)
    return folds


def fold_split(
    folds: list[dict[str, list[str]]], holdout: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """留出第 holdout 折，返回 (train_templates, test_templates)。"""
    train: dict[str, list[str]] = {}
    for intent in INTENT_TEMPLATES:
        train[intent] = [
            template
            for k, fold in enumerate(folds)
            if k != holdout
            for template in fold[intent]
        ]
    test = {intent: list(templates) for intent, templates in folds[holdout].items()}
    return train, test


def expand_templates(
    templates_by_intent: dict[str, list[str]],
    reps: int = 1,
    seed: int = 42,
    prefix: str = "rows",
) -> list[dict]:
    """把模板展开成样本。公开包装，供交叉验证复用同一套展开逻辑。"""
    return _expand(templates_by_intent, reps, random.Random(seed), prefix)


def complexity_floor(rows: list[dict]) -> dict:
    """复杂度标签的不可约噪声下限。

    `text_complexity` 由 (intent, 是否含升级关键词, 文本哈希) 生成，其中哈希部分
    **不可学**——测试文本的哈希值在训练时从未出现过。所以 complexity MAE 不该对齐 0，
    而应对齐「已知真实 intent 与关键词时的最优预测」。

    这个函数给出三档参照，用来判断模型到底学到了多少：

    | 参照 | 含义 |
    |---|---|
    | `random_guess_mae` | 永远预测全局均值 |
    | `intent_only_mae` | 知道真实 intent，预测该类均值 |
    | `oracle_mae` | 知道真实 intent **且**知道升级关键词（**下限**） |

    实测：`random_guess` 0.186 → `intent_only` 0.115 → `oracle` 0.098。
    也就是说 **71% 的可学信号只是 intent**，complexity 不是一个独立能力，
    它的价值在于演示「多输出头」与「标签有不可约噪声时指标要对齐下限」。
    """
    if not rows:
        return {}

    by_group: dict[tuple[str, bool], list[float]] = defaultdict(list)
    by_intent: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        intent = row["labels"]["intent"]
        value = float(row["labels"]["complexity"])
        by_group[(intent, ESCALATION_KEYWORD in row["text"])].append(value)
        by_intent[intent].append(value)

    group_mean = {key: fmean(values) for key, values in by_group.items()}
    intent_mean = {key: fmean(values) for key, values in by_intent.items()}
    grand = fmean(value for values in by_group.values() for value in values)

    def _mae(predict) -> float:
        return fmean(
            abs(float(row["labels"]["complexity"]) - predict(row)) for row in rows
        )

    return {
        "n": len(rows),
        "random_guess_mae": round(_mae(lambda _row: grand), 4),
        "intent_only_mae": round(
            _mae(lambda row: intent_mean[row["labels"]["intent"]]), 4
        ),
        "oracle_mae": round(
            _mae(
                lambda row: group_mean[
                    (row["labels"]["intent"], ESCALATION_KEYWORD in row["text"])
                ]
            ),
            4,
        ),
        "note": (
            "oracle 是已知真实 intent 与升级关键词时的最优预测；"
            "complexity 的可学成分主要来自 intent，它不是独立能力"
        ),
    }


def full_pool_complexity_floor() -> dict:
    """全模板池的复杂度下限。

    下限是**数据集的固有属性**，与切分口径无关。早期版本在 `03` 里对单次切分的
    test 子集（96 行）重算，得到 `0.0906`，与全池的 `0.0982` 不一致——两个报告
    给出两个「下限」会让读者困惑，也会让「距下限多远」这个判断失真。
    """
    rows = [
        {
            "text": text,
            "labels": {"intent": intent, "complexity": text_complexity(intent, text)},
        }
        for intent, texts in enumerate_texts().items()
        for text in texts
    ]
    return complexity_floor(rows)


def text_complexity(intent: str, text: str) -> float:
    """由文本确定性推导复杂度分数。

    旧实现每次采样 `random.uniform`，同一文本会得到不同标签（实测极差均值
    0.287），制造了无法学习的标签噪声。这里改为按文本哈希取值：同一文本
    恒定得到同一分数，任务从「猜随机数」变成「从文本泛化」。
    """
    lo, hi = _COMPLEXITY_RANGE[intent]
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    base = lo + unit * (hi - lo)
    if ESCALATION_KEYWORD in text:
        base = min(0.99, base + 0.16)
    return round(base, 4)


def needs_escalation(intent: str, text: str) -> bool:
    complexity = text_complexity(intent, text)
    return complexity > ESCALATION_COMPLEXITY_THRESHOLD or ESCALATION_KEYWORD in text


def unique_text_stats(rows: list[dict]) -> dict[str, int]:
    texts = {row["text"] for row in rows}
    templates = {strip_suffix(row["text"]) for row in rows}
    return {
        "rows": len(rows),
        "unique_texts": len(texts),
        "unique_templates": len(templates),
    }


def assert_no_text_leakage(
    train: list[dict], dev: list[dict], test: list[dict]
) -> None:
    """守卫：三个 split 的文本集合与模板集合都必须两两不相交。"""
    splits = {"train": train, "dev": dev, "test": test}
    text_sets = {name: {row["text"] for row in rows} for name, rows in splits.items()}
    template_sets = {
        name: {strip_suffix(row["text"]) for row in rows} for name, rows in splits.items()
    }
    _assert_pairwise_disjoint(text_sets, "文本")
    _assert_pairwise_disjoint(template_sets, "模板")


def _assert_pairwise_disjoint(sets: dict[str, set[str]], label: str) -> None:
    names = list(sets)
    problems = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                sample = sorted(overlap)[:2]
                problems.append(f"{left}∩{right} 共 {len(overlap)} 条，例如 {sample}")
    if problems:
        raise AssertionError(f"检测到{label}泄漏：" + "；".join(problems))


def _interleave(subtypes: dict[str, list[str]]) -> list[str]:
    """把各难度层按 round-robin 交织，使任意连续切分都拿到相近的难度配比。"""
    buckets = [list(subtypes.get(name, [])) for name in SUBTYPE_ORDER]
    out: list[str] = []
    depth = max((len(b) for b in buckets), default=0)
    for i in range(depth):
        for bucket in buckets:
            if i < len(bucket):
                out.append(bucket[i])
    return out


def _shuffle_within_subtype(
    intent: str, pool: list[str], rng: random.Random
) -> list[str]:
    """在同一难度层内部打乱，跨层顺序保持不变，避免切分后难度配比失衡。"""
    by_subtype: dict[str, list[str]] = {name: [] for name in SUBTYPE_ORDER}
    for template in pool:
        by_subtype[subtype_of(intent, template)].append(template)
    for bucket in by_subtype.values():
        rng.shuffle(bucket)
    return _interleave(by_subtype)


def _split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n < MIN_TEMPLATES_PER_INTENT:
        raise ValueError(
            f"模板池太小（{n} 条），每个 intent 至少需要 {MIN_TEMPLATES_PER_INTENT} 条"
            "才能切出非空的 train/dev/test"
        )
    n_train = max(1, min(int(n * ratios[0]), n - 2))
    n_dev = max(1, min(int(n * ratios[1]), n - n_train - 1))
    return n_train, n_dev, n - n_train - n_dev


def _expand(
    templates_by_intent: dict[str, list[str]],
    reps: int,
    rng: random.Random,
    prefix: str,
) -> list[dict]:
    rows: list[dict] = []
    for intent, templates in templates_by_intent.items():
        for template in templates:
            for suffix in SUFFIXES:
                text = template + suffix
                complexity = text_complexity(intent, text)
                escalation = needs_escalation(intent, text)
                for _ in range(reps):
                    rows.append(
                        {
                            "id": "",
                            "text": text,
                            "labels": {
                                "intent": intent,
                                "complexity": complexity,
                                "needs_escalation": escalation,
                            },
                        }
                    )
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"{prefix}-{i:05d}"
    return rows


def build_hard_cases() -> list[dict]:
    """人工难例：按失败模式分类，用于暴露整体指标掩盖掉的问题。

    **标注协议**（与 `schemas/v1.json` 的 `annotation_protocol` 一致）：

    1. 以用户**主诉求**为准——明确要求执行、且未被「顺便 / 另外」降级的那个动作；
    2. 多个诉求并列时，按「直接经济损失(billing) > 功能不可用(support) > 商务动作(sales)
       > 一般咨询(general)」取最高；
    3. 用户显式排除的诉求不计入（「账单我不管，先把我账号弄好」→ support）。

    每条带 `category`，便于把错误剖面按失败模式拆开统计。复杂度与升级标签是**人工判断**
    而非哈希推导——这正是难例集与训练集的区别。
    """
    seeds = [
        # --- 多诉求混合：考验「主诉求 vs 背景」的区分能力 ---
        ("退款和升级一起办，先做哪个？", "billing", 0.77, True, "multi_intent"),
        ("报价和发票一起要，今天给我", "billing", 0.72, True, "multi_intent"),
        ("系统老是崩，你们这个多少钱", "support", 0.81, True, "multi_intent"),
        ("账单我不管，先把我账号弄好", "support", 0.74, True, "multi_intent"),
        ("能便宜点吗，顺便问下发票怎么开", "sales", 0.52, False, "multi_intent"),
        ("接口先修好，扣钱的事之后再说", "support", 0.68, True, "multi_intent"),
        # --- 语义模糊：信息量不足，靠上下文猜 ---
        ("不是报错，就是有点不对劲", "support", 0.56, False, "vague"),
        ("看起来没问题但我不放心", "general", 0.34, False, "vague"),
        ("有点奇怪，你们看看", "general", 0.42, False, "vague"),
        ("反正就是不行", "support", 0.51, False, "vague"),
        ("你们这个怎么这样啊", "general", 0.38, False, "vague"),
        ("上次那个事还没解决", "general", 0.45, False, "vague"),
        # --- 同义转述：不含任何类别词，但语义明确 ---
        ("多收了我一笔钱", "billing", 0.44, False, "synonym"),
        ("转半天出不来", "support", 0.62, True, "synonym"),
        ("我们想搞一套，怎么谈", "sales", 0.48, False, "synonym"),
        ("钱花得不明不白", "billing", 0.50, False, "synonym"),
        ("东西发出去对方没收到", "support", 0.58, False, "synonym"),
        # --- 对抗表达：反问、讽刺、错别字 ---
        ("你们这系统是给人用的吗？", "support", 0.66, True, "adversarial"),
        ("呵呵，又崩了", "support", 0.70, True, "adversarial"),
        ("发票台头写错了能改吗", "billing", 0.36, False, "adversarial"),
        ("这个报错是你们的问题吧", "support", 0.60, True, "adversarial"),
        # --- 极端长度：过短无信息 / 过长多诉求 ---
        ("崩了", "support", 0.30, True, "extreme_length"),
        ("不好", "general", 0.25, False, "extreme_length"),
        ("嗯", "general", 0.15, False, "extreme_length"),
        (
            (
                "我们是一家三百人规模的制造业公司，目前用团队版，最近想升级到企业版，"
                "希望了解私有化部署的报价、实施周期、是否支持定制开发"
            ),
            "sales", 0.88, True, "extreme_length",
        ),
        # --- 领域黑话与中英混杂 ---
        ("SSO 登录回调 403", "support", 0.64, True, "jargon"),
        ("想上私有化，预算大概多少", "sales", 0.58, False, "jargon"),
        ("想加个 SLA 保障条款", "sales", 0.58, False, "jargon"),
        # --- 复杂度边界：最简单与最紧急 ---
        ("怎么导出我的数据", "general", 0.20, False, "boundary"),
        ("所有门店同时不可用，影响生产，马上处理", "support", 0.95, True, "boundary"),
        # --- 否定陷阱：句子里出现类别词，但被否定掉 ---
        ("不是退款，是发票开错了", "billing", 0.45, False, "negation"),
        ("不是说不能用，是我不会用", "general", 0.28, False, "negation"),
        ("别给我推销，我就想问发票怎么补", "billing", 0.42, False, "negation"),
        ("没有报错，就是数据对不上", "support", 0.52, False, "negation"),
        # --- 错别字与噪声：考验鲁棒性 ---
        ("发飘抬头写错了", "billing", 0.38, False, "typo_noise"),
        ("系统又卡死了！！！急急急", "support", 0.72, True, "typo_noise"),
        ("我想问下。。。这个套餐多少钱？", "sales", 0.40, False, "typo_noise"),
        ("扣费有问提，帮我查下", "billing", 0.44, False, "typo_noise"),
        # --- 兜底陷阱：有强类别信号，但用户明确只是问信息 ---
        ("先别处理，我就想知道退款一般多久", "general", 0.40, False, "fallback_trap"),
        ("不用帮我改，我就是问下发票怎么开", "general", 0.36, False, "fallback_trap"),
        ("我没遇到问题，只是想了解报错怎么排查", "general", 0.38, False, "fallback_trap"),
        ("暂时不买，先看看价格体系", "general", 0.34, False, "fallback_trap"),
        # --- 补分布：把 intent 真值调均衡 ---
        ("优惠券用不了，提示已过期", "billing", 0.46, False, "coverage"),
        ("对公转账的账号在哪", "billing", 0.50, False, "coverage"),
        ("两个账号的账单能合并吗", "billing", 0.55, False, "coverage"),
        ("想加几个账号，怎么算钱", "sales", 0.42, False, "coverage"),
        ("能不能按季度付", "sales", 0.40, False, "coverage"),
        ("多买几个账号有优惠吗", "sales", 0.48, False, "coverage"),
        ("想了解一下你们有没有教育行业案例", "sales", 0.52, False, "coverage"),
        ("续约要提前多久说", "sales", 0.44, False, "coverage"),
    ]
    rows = []
    for idx, (text, intent, comp, esc, category) in enumerate(seeds):
        rows.append(
            {
                "id": f"hard-{idx:04d}",
                "text": text,
                "category": category,
                "labels": {"intent": intent, "complexity": comp, "needs_escalation": esc},
            }
        )
    return rows

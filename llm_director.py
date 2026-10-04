"""
LLM 分镜导演：规则写进 system prompt，由 qwen-max 遵守。
本地只做结构整理与轻量校验，不做主题相关词表黑名单改写。
"""
from __future__ import annotations

import json
import os
import re
import time
from openai import OpenAI

from local_env import load_local_env

load_local_env()

MAX_ANCHOR_WORDS = 22
MAX_KEYFRAME_WORDS = 50
MAX_I2V_ANCHOR_WORDS = 24
MAX_I2V_ACTION_WORDS = 20
MAX_I2V_TOTAL_WORDS = 52
MAX_SIMILARITY = 0.72
MAX_CONSECUTIVE_SHOT_SIMILARITY = 0.40  # 仅作诊断参考；出片校验不再用它拒镜
MAX_GENERATION_ATTEMPTS = 5
MAX_CONSECUTIVE_WALKING_BEATS = int(os.getenv("MAX_CONSECUTIVE_WALKING_BEATS", "1"))
# 允许相似动作；禁止「完全同一条动作文案」；同一动作指纹至少隔这么多镜距（1=禁止相邻）
MIN_REPEAT_GAP = int(os.getenv("MIN_REPEAT_GAP", "2"))
# 连续行进镜上限（故事优先：默认 1，避免多段同构图背影走）
MAX_CONSECUTIVE_LOCOMOTION = int(os.getenv("MAX_CONSECUTIVE_LOCOMOTION", "1"))
# 回头/过肩察觉：全片最多 1 镜；禁止连续回头注水
MAX_LOOKBACK_CLIMAX_BEATS = int(os.getenv("MAX_LOOKBACK_CLIMAX_BEATS", "1"))
MAX_CONSECUTIVE_LOOKBACK_BEATS = int(os.getenv("MAX_CONSECUTIVE_LOOKBACK_BEATS", "1"))
# 见脸硬切语义：首帧已定妆，I2V 只允许微动（开源 I2V 禁大角度拧头）
FACE_MICRO_LOOKBACK = "过肩侧脸微动，目光警觉"
REAR_NOTICE_TILT = "停下脚步，保持背影，镜头微仰跟随上方动静"

# 出片优先：启发式校验降为警告，禁止因多样性/占手卡死整单
SHIPPABLE_MODE = os.getenv("SHIPPABLE_MODE", "1") == "1"
# 产品容量：最多约 1 分钟；故事优先默认偏好更短，禁止注水凑满
MAX_TARGET_SECONDS = float(os.getenv("MAX_TARGET_SECONDS", "60"))
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "5.0"))
ABS_MIN_BEATS = int(os.getenv("ABS_MIN_BEATS", "3"))
# 故事优先硬顶：默认最多 8 镜（可用 ABS_MAX_BEATS 放开）
ABS_MAX_BEATS = int(
    os.getenv(
        "ABS_MAX_BEATS",
        str(max(ABS_MIN_BEATS, min(8, int(round(MAX_TARGET_SECONDS / CLIP_SECONDS))))),
    )
)
# 未指定时长时偏好约 35 秒（≈7 镜），宁短勿注水
DEFAULT_TARGET_SECONDS = float(
    os.getenv(
        "TARGET_VIDEO_SECONDS",
        str(min(35.0, MAX_TARGET_SECONDS) if SHIPPABLE_MODE else MAX_TARGET_SECONDS),
    )
)
_DERIVED_BEATS = max(ABS_MIN_BEATS, round(DEFAULT_TARGET_SECONDS / CLIP_SECONDS))
PREFERRED_BEATS = int(os.getenv("PREFERRED_BEATS", str(_DERIVED_BEATS)))
MIN_BEATS = int(os.getenv("MIN_BEATS", str(max(ABS_MIN_BEATS, PREFERRED_BEATS - 2))))
MAX_BEATS = int(os.getenv("MAX_BEATS", str(min(ABS_MAX_BEATS, PREFERRED_BEATS + 2))))
TARGET_VIDEO_SECONDS = DEFAULT_TARGET_SECONDS

_FALLBACK_MOTIONS = (
    "转头看向侧方声响",
    "举起空手抬到眉边向前扫描",
    "举起空手遮挡头顶强光",
    "上身微微侧倾，目光仍盯前方",
    "空手向前伸出做制止手势",
    "过肩侧脸微动，目光警觉",
    "明确向左侧脸微倾",
    "明确向右侧脸微倾",
    "放下空手，重心移到前脚",
    "空手指向头顶上方的轮廓",
)

# 需要「脸/侧脸可读」才能看清的动作
_FACE_READABLE_ACTION_RE = re.compile(
    r"("
    r"下巴|眯眼|眼神|凝视|面部|脸|"
    r"抬头|仰头|仰视|抬下巴|"
    r"回头|回望|过肩|越过肩膀|"
    r"\bchin\b|\beyes?\s+narrow\b|\bgaze\b|\bstare\b|\bstaring\b|\bface\b|\bfacial\b|"
    r"\blook(?:s|ing)?\s+up\b|\bgaz(?:e|es|ing)\s+up\b|\btilt(?:s|ing)?\s+chin\b|"
    r"\btracking\s+something\s+above\b|"
    r"\blook(?:s|ing)?\s+back\b|\bover\s+(?:the\s+)?shoulder\b|\bsnaps?\s+gaze\b|"
    r"\bturns?\s+head\s+(?:sharply\s+)?(?:toward|to\s+look|around)\b"
    r")",
    flags=re.IGNORECASE,
)

# 背影可延续 + 适合镜头推进的停步小动作
_REAR_PUSHIN_COMPAT_RE = re.compile(
    r"("
    r"举手|抬手|扫描|遮光|制止手势|侧倾|重心|"
    r"\braises?\s+(?:\w+\s+)?hand\b|"
    r"\bscan\s+ahead\b|\bshield\s+eyes\b|\bstop\s+gesture\b|"
    r"\bleans?\s+torso\b|\bshifts?\s+weight\b|\blowers?\s+(?:\w+\s+)?hand\b"
    r")",
    flags=re.IGNORECASE,
)

# 同类微动作族：抬头族
_LOOK_UP_FAMILY_RE = re.compile(
    r"("
    r"抬头|仰头|仰视|抬下巴|盯着天空|看向头顶|"
    r"\blook(?:s|ing)?\s+up\b|\bgaz(?:e|es|ing)\s+upward\b|\btilt(?:s|ing)?\s+chin\b|"
    r"\bchin\s+upward\b|\btracking\s+(?:something\s+)?above\b|"
    r"\bhead\s+(?:sharply\s+)?(?:to\s+)?look\s+up\b|"
    r"\beyes?\s+fixed\s+on\s+(?:the\s+)?sky\b|\bstares?\s+at\s+(?:the\s+)?(?:sky|drone\s+above)\b"
    r")",
    flags=re.IGNORECASE,
)

# 凭空掏出/瞄准器械（中英）
_INVENTED_HANDHELD_ACTION_RE = re.compile(
    r"("
    r"(?:掏出|抽出|拿出|取出|拔出)\s*(?:一[个把支]?\s*)?(?:(?:小型|高科技|手持)\s*)*"
    r"(?:装置|设备|器械|手枪|步枪|武器|手机|遥控器)|"
    r"伸手(?:进|入)?(?:大衣|风衣|外套|口袋).{0,20}(?:装置|设备|器械|枪|手机)|"
    r"瞄准.{0,12}(?:装置|设备|枪|武器)|"
    r"\b(?:pulls?|draws?|takes?|grabs?)(?:\s+out)?\s+"
    r"(?:a\s+|an\s+|his\s+|her\s+|the\s+)?"
    r"(?:(?:small|sleek|metallic|high-tech|handheld|portable|tiny|compact)\s+)*"
    r"(?:device|gadget|gun|pistol|rifle|weapon|phone|blaster|remote)\b|"
    r"\breaches?\s+into\s+(?:his\s+|her\s+|their\s+|the\s+)?(?:coat|pocket|jacket)\b"
    r".{0,40}\b(?:device|gadget|gun|pistol|weapon|phone|blaster)\b|"
    r"\b(?:aims?|pointing|points?)\s+(?:the\s+|a\s+|his\s+|her\s+)?"
    r"(?:device|gadget|gun|pistol|rifle|weapon|blaster)\b|"
    r"\b(?:beam of light|laser beam|emitting a beam|energy beam)\b"
    r")",
    flags=re.IGNORECASE,
)

# 未授权捡起/抓住外部追踪物（主题只写「出现/追踪」不等于授权上手）
_INVENTED_SEIZE_OBJECT_RE = re.compile(
    r"("
    r"(?:捡起|拾起|拿起|抓起|抓住|抓向|试图抓住|伸手去抓|弯腰去(?:捡|拿|抓))"
    r".{0,24}(?:无人机|飞行器|追踪器|装置|追踪物)|"
    r"(?:手持|捧着|拿着|检查|端详).{0,12}(?:无人机|飞行器|追踪器)|"
    r"\b(?:picks?\s+up|pick\s+up|bends?\s+down\s+to\s+(?:pick|grab)|grabs?\s+up|"
    r"tries?\s+to\s+(?:grab|catch)|reaches?\s+(?:for|to\s+grab))\b"
    r".{0,48}\b(?:drone|uav|tracker|device|gadget|probe)\b|"
    r"\b(?:holding|examining|inspecting)\s+(?:the\s+)?(?:fallen\s+|mysterious\s+)?(?:drone|uav|tracker)\b|"
    r"\b(?:drone|uav|tracker)\s+in\s+(?:his|her|their)\s+hand\b"
    r")",
    flags=re.IGNORECASE,
)
_OFFSCREEN_TARGET_RE = re.compile(
    r"画外|镜头外|画面外|off[- ]?screen|out of(?:\s+the)?\s+frame",
    flags=re.IGNORECASE,
)
# 高潮物件「进入画面」写法（主题无关；不含单独「入画」以免「注视入画的X」误判）
_CLIMAX_ENTER_RE = re.compile(
    r"("
    r"进入画面|从上方进入|从画面上方|降入|飞入|落入|飞进画面|"
    r"\benters?\s+(?:the\s+)?(?:frame|shot)\b|"
    r"\bdescends?\s+into\b|\bfly(?:es|ing)?\s+into\b"
    r")",
    flags=re.IGNORECASE,
)
# 计数用：真正的回头/过肩察觉动作（不含单独「眼神锐利」）
_LOOKBACK_BEAT_RE = re.compile(
    r"("
    r"回头|回望|过肩|越过肩膀|猛然转头|猛然回头|"
    r"\blooks?\s+back\b|\blooking\s+back\b|\bover\s+(?:the\s+)?shoulder\b|"
    r"\bturns?\s+head\s+(?:sharply\s+)?(?:toward|to\s+look|around)\b|"
    r"\bturns?\s+(?:sharply\s+)?(?:around|to\s+(?:look|face))\b"
    r")",
    flags=re.IGNORECASE,
)
_TOPIC_SEIZE_AUTH_RE = re.compile(
    r"捡起|拾起|拿起|抓起|拾获|"
    r"\b(?:picks?\s+up|pick\s+up|grabs?\s+up|retrieves?)\b",
    flags=re.IGNORECASE,
)
_TOPIC_LOOKBACK_RE = re.compile(
    r"回头|猛然回头|转身看|察觉后|"
    r"\b(?:looks?\s+back|looking\s+back|turns?\s+(?:sharply\s+)?(?:around|to\s+look)|"
    r"over\s+(?:the\s+)?shoulder|snaps?\s+gaze|eyes?\s+narrowing)\b",
    flags=re.IGNORECASE,
)
_LOOKBACK_ACTION_RE = re.compile(
    r"("
    r"回头|回望|过肩|越过肩膀|猛然转头|眼神锐利|眯起眼睛|"
    r"\blooks?\s+back\b|\blooking\s+back\b|\bover\s+(?:the\s+)?shoulder\b|"
    r"\bturns?\s+head\s+(?:sharply\s+)?(?:toward|to\s+look|around)\b|"
    r"\bsnaps?\s+gaze\b|\beyes?\s+narrow(?:ing)?\b|"
    r"\bturns?\s+(?:sharply\s+)?to\s+(?:look|face)\b"
    r")",
    flags=re.IGNORECASE,
)
_TOPIC_HANDHELD_AUTH_RE = re.compile(
    r"枪|手枪|步枪|武器|装置|设备|遥控器|手机|手电|器械|"
    r"\b(?:gun|pistol|rifle|weapon|device|gadget|phone|blaster|flashlight|remote)\b",
    flags=re.IGNORECASE,
)
_LOOK_AT_CAMERA_RE = re.compile(
    r"("
    r"看向镜头|注视镜头|对视观众|看镜头|自拍看镜头|"
    r"\b(?:looks?|looking|stares?|staring|gazes?|gazing|faces?|facing)\s+"
    r"(?:at\s+|into\s+|toward(?:s)?\s+)?(?:the\s+)?(?:camera|lens|viewer|audience)\b|"
    r"\b(?:eye\s*contact|fourth\s*wall|breaking the fourth wall)\b"
    r")",
    flags=re.IGNORECASE,
)

_LOCOMOTION_RE = re.compile(
    r"("
    r"走入|走进|迈步|迈进|前行|行走|大步|踏步|"
    r"\b(?:step|steps|walk|walks|stride|advances?)\b"
    r")",
    flags=re.IGNORECASE,
)
# 主题开篇是否在建立「走进场景」（用于强制第 1 镜为行进）
_OPENING_LOCO_RE = re.compile(
    r"("
    r"走入|走进|迈进|走去|步行进入|"
    r"\bwalks?\s+(?:into|toward|towards|down|through)\b|"
    r"\benters?\b|\bsteps?\s+into\b"
    r")",
    flags=re.IGNORECASE,
)

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 主题里声明时长的常见写法（解析后可从叙事文本中剥离）
_DURATION_PATTERNS = (
    (re.compile(r"(?:约|大概|左右)?\s*(\d{1,3})\s*秒(?:钟|钟钟)?", re.I), 1.0),
    (re.compile(r"(?:约|大概|左右)?\s*(\d{1,3})\s*s(?:ec(?:onds?)?)?\b", re.I), 1.0),
    (re.compile(r"(?:约|大概|左右)?\s*(\d{1,2})\s*分钟", re.I), 60.0),
    (re.compile(r"(?:约|大概|左右)?\s*(\d{1,2})\s*min(?:ute)?s?\b", re.I), 60.0),
    (re.compile(r"(?:时长|长度|目标|成片)\s*[：:为]?\s*(\d{1,3})\s*秒", re.I), 1.0),
    (re.compile(r"(?:时长|长度|目标|成片)\s*[：:为]?\s*(\d{1,2})\s*分钟", re.I), 60.0),
)
_DURATION_LITERALS = (
    (re.compile(r"半分钟"), 30.0),
    (re.compile(r"一分钟|1\s*分钟"), 60.0),
    (re.compile(r"一分半|一分半钟"), 90.0),
)


def parse_target_seconds_from_text(text: str) -> float | None:
    """从主题文本解析用户想要的成片秒数；找不到返回 None。"""
    raw = text or ""
    for pat, _ in _DURATION_LITERALS:
        if pat.search(raw):
            # 字面量优先匹配更具体的；按出现顺序取第一个命中
            for lit_pat, secs in _DURATION_LITERALS:
                if lit_pat.search(raw):
                    return float(secs)
    for pat, mul in _DURATION_PATTERNS:
        m = pat.search(raw)
        if m:
            return float(m.group(1)) * mul
    return None


def strip_duration_directives(text: str) -> str:
    """去掉时长声明，避免 LLM 把「生成30秒」当成剧情。"""
    cleaned = text or ""
    for pat, _ in _DURATION_LITERALS:
        cleaned = pat.sub(" ", cleaned)
    for pat, _ in _DURATION_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = re.sub(r"(?:请)?(?:帮我)?(?:生成|制作|输出)\s*(?:一段|一个)?\s*(?:视频|短片)?", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ，,。.;；")


def estimate_beats_from_topic(topic: str) -> int:
    """按主题信息密度估算镜数（不锁死固定 10 镜）。短主题少镜，事件多则加镜。"""
    text = strip_duration_directives(topic or "").strip()
    if not text:
        return ABS_MIN_BEATS

    parts = re.split(
        r"[。！？；;\n]+|(?:，(?=突然|忽然|然后|接着|随后|最后|于是|这时|此时))",
        text,
    )
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
    event_hints = len(
        re.findall(
            r"突然|忽然|然后|接着|随后|最后|于是|发现|察觉|回头|追踪|降落|走入|走进|出现|攻击|逃跑|瞄准",
            text,
        )
    )
    chars = len(re.sub(r"\s+", "", text))

    n = max(len(parts), event_hints + 1, 3)
    if chars >= 80:
        n = max(n, 5)
    if chars >= 140:
        n = max(n, 7)
    if chars >= 220:
        n = max(n, 9)
    if chars >= 320:
        n = max(n, 11)
    # 短主题封顶防注水；但事件词多时仍允许略加镜
    if chars < 60:
        n = min(n, max(5, min(7, event_hints + 2)))
    elif chars < 100:
        n = min(n, max(7, min(9, event_hints + 2)))
    elif chars < 160:
        n = min(n, max(9, min(11, event_hints + 2)))

    return int(max(ABS_MIN_BEATS, min(ABS_MAX_BEATS, n)))


def resolve_beat_budget(
    user_topic: str,
    target_seconds: float | None = None,
) -> dict:
    """
    决定本单镜数预算（主题无关）。
    优先级：显式 target_seconds > 主题文本里的时长声明 > 按主题密度估算。
    硬顶：MAX_TARGET_SECONDS（默认 60s ≈ 一分钟成片容量）。
    """
    source = "topic_estimate"
    if target_seconds is not None and float(target_seconds) > 0:
        seconds = float(target_seconds)
        source = "user_override"
    else:
        parsed = parse_target_seconds_from_text(user_topic or "")
        if parsed and parsed > 0:
            seconds = float(parsed)
            source = "topic_duration"
        else:
            preferred = estimate_beats_from_topic(user_topic or "")
            seconds = preferred * CLIP_SECONDS
            source = "topic_estimate"

    # 产品上限：约一分钟；更长声明会被压到上限（一分半 → 60s）
    seconds = max(ABS_MIN_BEATS * CLIP_SECONDS, min(MAX_TARGET_SECONDS, float(seconds)))
    preferred = int(round(seconds / CLIP_SECONDS))
    preferred = max(ABS_MIN_BEATS, min(ABS_MAX_BEATS, preferred))
    # 估算模式 ±2；用户指定时长收紧为 ±1
    slack = 1 if source in ("user_override", "topic_duration") else 2
    min_beats = max(ABS_MIN_BEATS, preferred - slack)
    max_beats = min(ABS_MAX_BEATS, preferred + slack)
    if min_beats > max_beats:
        min_beats, max_beats = max_beats, min_beats

    story_topic = strip_duration_directives(user_topic or "") or (user_topic or "").strip()
    return {
        "min_beats": min_beats,
        "preferred_beats": preferred,
        "max_beats": max_beats,
        "target_seconds": round(preferred * CLIP_SECONDS, 1),
        "clip_seconds": CLIP_SECONDS,
        "max_capacity_seconds": MAX_TARGET_SECONDS,
        "source": source,
        "story_topic": story_topic,
    }


_MOTION_VERB_RE = re.compile(
    r"("
    r"走|迈|行|转|抬|举|放|拉|掏|抽|倾|指|伸|跪|回望|回头|回望|扫描|制止|"
    r"进入|飞入|降入|推镜|推向|注视|望向|停下|停步|迈步|前行|微动|微仰|"
    r"\b(?:step|steps|walk|walks|turn|turns|raise|raises|lower|lowers|"
    r"lift|lifts|pull|pulls|tilt|tilts|bend|bends|lean|leans|point|points|"
    r"extend|extends|advance|advances|snap|snaps|kneel|kneels|"
    r"enter|enters|push|pushes|descend|descends|gaze|gazes)\b"
    r")",
    flags=re.IGNORECASE,
)
_STATIC_ACTION_RE = re.compile(
    r"("
    r"站着不动|一动不动|完全静止|保持静止|"
    r"\b(?:stands?\s+(?:still|motionless)|motionless|fully\s+halted|"
    r"remains?\s+(?:still|motionless|stationary)|static\s+pose|for\s+(?:a\s+)?full\s+(?:count|duration))\b"
    r")",
    flags=re.IGNORECASE,
)
_BOTH_HANDS_DOWN_RE = re.compile(
    r"\b(lowers?\s+both\s+hands|both\s+hands?\s+(?:to|at)\s+(?:his\s+|her\s+)?sides?)\b",
    flags=re.IGNORECASE,
)
_HANDHELD_PROP_RE = re.compile(
    r"("
    r"伞|雨伞|剑|刀|枪|手枪|步枪|武器|盾|灯笼|火把|拐杖|包|公文包|矛|弓|斧|锤|手机|平板|"
    r"鱼竿|钓竿|"
    r"\b(?:umbrella|parasol|sword|katana|blade|gun|pistol|rifle|weapon|shield|"
    r"lantern|torch|staff|cane|bag|briefcase|spear|bow|axe|hammer|phone|tablet|"
    r"fishing\s+rod|fish\s+rod|rod|reel|pole|canopy)\b"
    r")",
    flags=re.IGNORECASE,
)
_BOTH_HANDS_DOWN_ZH_RE = re.compile(r"双手放下|两手放下|双手垂下")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def limit_words(text: str, max_words: int) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    # 中文为主时按字符预算截断（约 2 字 ≈ 1 英文词）
    cjk = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    if cjk >= max(4, len(cleaned) // 4):
        max_chars = max(8, max_words * 2)
        return cleaned[:max_chars].strip(" ,.;，。；")
    words = cleaned.split()
    return " ".join(words[:max_words]).strip(" ,.;")


def limit_words_at_clause(text: str, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" ,.;，。；")
    if not cleaned:
        return cleaned
    cjk = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    if cjk >= max(4, len(cleaned) // 4):
        max_chars = max(8, max_words * 2)
        truncated = cleaned[:max_chars].strip(" ,.;，。；")
        # 尽量在标点处截断
        for sep in ("。", "，", "；", ",", ";"):
            idx = truncated.rfind(sep)
            if idx >= max_chars // 2:
                return truncated[: idx + 1].strip(" ,.;，。；")
        return truncated
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    truncated = " ".join(words[:max_words]).strip(" ,.;")
    while truncated:
        last = truncated.split()[-1].lower().strip(".,;:'\"")
        if last in {"by", "to", "his", "her", "the", "a", "an", "and", "of", "from", "with", "on", "in", "at", "as"}:
            truncated = " ".join(truncated.split()[:-1]).strip(" ,.;")
            continue
        break
    return truncated


def extract_json_block(result_text: str) -> str:
    text = (result_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _format_llm_response_error(response, model: str, raw_body: str = "") -> str:
    """HTTP 200 但 choices 为空时，尽量把网关真实错误打出来。"""
    parts = [f"model={model}"]
    for attr in ("id", "model", "object", "created"):
        value = getattr(response, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    error = getattr(response, "error", None)
    if error is None and hasattr(response, "model_extra"):
        error = (response.model_extra or {}).get("error")
    if error is not None:
        parts.append(f"error={error}")
    if raw_body:
        parts.append(f"http_body={raw_body[:800]}")
    else:
        try:
            dump = response.model_dump() if hasattr(response, "model_dump") else None
            if dump is not None:
                parts.append(f"raw={json.dumps(dump, ensure_ascii=False)[:800]}")
        except Exception:
            pass
    return "; ".join(parts)


def _extract_llm_text(response) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    for alt in ("reasoning_content", "refusal"):
        alt_text = getattr(message, alt, None)
        if isinstance(alt_text, str) and alt_text.strip():
            print(f"⚠️ LLM 使用备用字段 {alt}")
            return alt_text
    return None


def _extract_text_from_raw_body(raw_body: str) -> str | None:
    """SDK 把非标准包解析成全 null 时，直接从 HTTP body 抠正文。"""
    text = (raw_body or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

    output = data.get("output")
    if isinstance(output, dict):
        if isinstance(output.get("text"), str) and output["text"].strip():
            return output["text"]
        out_choices = output.get("choices")
        if isinstance(out_choices, list) and out_choices:
            msg = out_choices[0].get("message") or out_choices[0]
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str) and content.strip():
                return content

    for key in ("text", "content", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def call_llm(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """调用兼容 OpenAI 的 Qwen 接口；固定 QWEN_MODEL，空包重试并打印原始 body。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "4096"))
    max_retries = int(os.getenv("QWEN_HTTP_RETRIES", "3"))
    last_detail = ""

    for attempt in range(1, max_retries + 1):
        print(f"🤖 LLM model={QWEN_MODEL} attempt={attempt}/{max_retries}")
        raw_body = ""
        try:
            create = client.chat.completions.create
            # 优先拿原始 HTTP，避免网关非标准 JSON 被 SDK 解成全 null
            with_raw = getattr(client.chat.completions, "with_raw_response", None)
            if with_raw is not None:
                raw_resp = with_raw.create(
                    model=QWEN_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_body = getattr(raw_resp, "text", "") or ""
                response = raw_resp.parse()
            else:
                response = create(
                    model=QWEN_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            last_detail = f"model={QWEN_MODEL}; exception={exc}"
            print(f"⚠️ LLM 请求失败: {last_detail}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
            continue

        text = _extract_llm_text(response)
        if not text and raw_body:
            text = _extract_text_from_raw_body(raw_body)
            if text:
                print("ℹ️ 已从原始 HTTP body 提取正文（SDK choices 为空）")

        if text:
            return text

        last_detail = _format_llm_response_error(response, QWEN_MODEL, raw_body=raw_body)
        print(f"⚠️ LLM 空响应: {last_detail}")
        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"LLM `{QWEN_MODEL}` 连续 {max_retries} 次返回空内容。"
        f" 详情: {last_detail}"
    )


def keyword_set(text: str) -> set[str]:
    stop = {
        "a", "an", "the", "in", "on", "at", "with", "and", "or", "of", "to", "from",
        "he", "she", "they", "his", "her", "their", "same", "person", "face", "outfit",
    }
    return {
        re.sub(r"[^a-z0-9-]", "", w.lower())
        for w in (text or "").split()
        if re.sub(r"[^a-z0-9-]", "", w.lower()) and re.sub(r"[^a-z0-9-]", "", w.lower()) not in stop
    }


def prompt_similarity(a: str, b: str) -> float:
    sa, sb = keyword_set(a), keyword_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def average_pairwise_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(prompt_similarity(texts[i], texts[j]))
    return sum(scores) / len(scores) if scores else 0.0


def extract_motion_signature(action: str) -> str:
    """主题无关的动作指纹：主要动词 + 身体部位/方向词（中英）。"""
    text = action or ""
    zh_verbs = re.findall(
        r"走入|走进|迈步|迈进|前行|行走|大步|踏步|转头|回头|回望|抬头|仰头|"
        r"举起|抬起|放下|伸出|侧倾|扫描|制止|瞄准|掏出|捡起|指向",
        text,
    )
    zh_parts = re.findall(
        r"头|手|臂|脚|下巴|上身|肩|目光|左|右|前|上|下|口袋|眉",
        text,
    )
    if zh_verbs or zh_parts:
        return " ".join((zh_verbs[:2] + zh_parts[:3]) or re.findall(r"[\u4e00-\u9fff]{2}", text)[:4])

    words = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2]
    verbs = [w for w in words if _MOTION_VERB_RE.search(w)]
    body_dir = {
        "head", "hand", "arm", "foot", "chin", "torso", "knee", "shoulder",
        "gaze", "body", "left", "right", "forward", "upward", "downward",
        "aside", "pocket", "brow",
    }
    parts = [w for w in words if w in body_dir]
    sig = verbs[:2] + parts[:3]
    return " ".join(sig) if sig else " ".join(words[:4])


def normalize_action_key(action: str) -> str:
    """用于判定「完全重复」的归一化文案键（中英）。"""
    text = (action or "").lower()
    text = re.sub(r"[^\u4e00-\u9fff a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def action_is_exact_duplicate(candidate: str, previous: list[str]) -> bool:
    key = normalize_action_key(candidate)
    if not key:
        return False
    return any(normalize_action_key(prev) == key for prev in previous)


def action_signature_too_close(
    candidate: str,
    previous: list[str],
    *,
    min_gap: int = MIN_REPEAT_GAP,
) -> bool:
    """同一动作指纹在 min_gap 镜距内再次出现则过近（gap=2 → 禁止相邻）。"""
    sig = extract_motion_signature(candidate)
    if not sig or min_gap <= 1:
        return False
    start = max(0, len(previous) - (min_gap - 1))
    for prev in previous[start:]:
        if extract_motion_signature(prev) == sig:
            return True
    return False


def action_spacing_conflict(
    candidate: str,
    previous: list[str],
    *,
    min_gap: int = MIN_REPEAT_GAP,
) -> bool:
    """完全重复（全文），或相同指纹/动作族靠太近。"""
    if action_is_exact_duplicate(candidate, previous):
        return True
    if action_signature_too_close(candidate, previous, min_gap=min_gap):
        return True
    return action_family_too_close(candidate, previous, min_gap=min_gap)


def is_lookback_beat_action(action: str) -> bool:
    """是否为回头/过肩察觉节拍（用于预算与去重，主题无关）。"""
    return bool(_LOOKBACK_BEAT_RE.search(action or ""))


def is_large_head_or_body_turn(action: str) -> bool:
    """开源 I2V 会崩眼/崩脸的大角度转身或猛拧头。"""
    return bool(
        re.search(
            r"(?:迅速|突然|猛然)?(?:全身|整身)?(?:转身|急转|拧身)|"
            r"猛然回头|猛然回望|猛然转头|大幅度转|"
            r"\bturns?\s+(?:sharply|quickly|suddenly)\b|"
            r"\bspins?\s+around\b|\bwhips?\s+(?:his|her|their)\s+head\b",
            action or "",
            flags=re.I,
        )
    )


def prefers_face_micro_motion(action: str, user_topic: str = "") -> bool:
    """
    见脸镜走「硬切定妆 + I2V 微动」：禁止让模型生成大角度转头过程。
    行进/入画/纯背影微仰不走此路径。
    """
    text = action or ""
    if not text:
        return False
    if is_locomotion_action(text) or is_climax_enter_action(text, user_topic):
        return False
    if re.search(r"保持背影|背影停|镜头微仰跟随", text) and not is_lookback_beat_action(text):
        return False
    return bool(
        is_lookback_beat_action(text)
        or re.search(r"侧脸微动|过肩侧脸微动|目光警觉", text)
    )


def face_micro_gaze_at_climax(context_text: str) -> str:
    objs = extract_climax_object_phrases(context_text)
    obj = objs[0] if objs else "物件"
    return f"侧脸微动，注视已入画的{obj}"


def action_mentions_climax_object(action: str, user_topic: str = "") -> bool:
    """动作是否提及主题高潮物件（或主题无关的入画泛称）。"""
    phrases = extract_climax_object_phrases(user_topic)
    text = action or ""
    if phrases and any(p in text for p in phrases):
        return True
    # 泛称：不依赖具体题材名词
    return bool(
        re.search(
            r"入画的?(?:追踪物|目标|物件)|可见的?(?:追踪|飞行)|"
            r"(?:高潮)?物件|"
            r"\b(?:climax\s+)?(?:object|target|tracker)\b",
            text,
            flags=re.I,
        )
    )


def is_climax_enter_action(action: str, user_topic: str = "") -> bool:
    """
    本镜是否为「物件进入画面」节拍——主题无关的结构规则。
    只认进入写法（从上方进入/进入画面/飞入等），不枚举无人机等具体物件名。
    user_topic 仅用于可选交叉确认：主题有高潮物清单时，要求提及该物或入画泛称，
    避免把无关「进入」句误判；无清单时进入写法本身即成立。
    """
    text = action or ""
    if not _CLIMAX_ENTER_RE.search(text):
        return False
    phrases = extract_climax_object_phrases(user_topic) if user_topic else []
    if phrases:
        return action_mentions_climax_object(text, user_topic)
    return True


def prefers_tilt_up_camera(action: str, user_topic: str = "") -> bool:
    """仰视类动作（不含「物件进入」——进入镜一律走背后推镜，主题无关）。"""
    if is_climax_enter_action(action, user_topic):
        return False
    return bool(
        re.search(
            r"抬头|仰头|仰视|抬下巴|"
            r"\blooks?\s+up\b|\bgaz(?:e|es|ing)\s+up\b|\boverhead\b",
            action or "",
            flags=re.I,
        )
    )


def prefers_reveal_pushin(action: str, user_topic: str = "") -> bool:
    """
    物件入画镜：统一「镜头推向主体背部 + 物件进入」。
    全局结构规则，不绑定任何题材；Wan 对弱运镜词（微仰）几乎无效。
    """
    return is_climax_enter_action(action, user_topic)


def pick_fallback_motion(
    previous: list[str] | None = None,
    *,
    grip_hand: str | None = None,
    min_gap: int = MIN_REPEAT_GAP,
    index: int = 0,
    context_text: str = "",
) -> str:
    """结构兜底：在全局规则下轮换，避免总是落到同一句。"""
    previous = previous or []
    free_zh = {"left": "左手", "right": "右手"}.get(
        (grip_hand or "right").lower(), "右手"
    )
    free_other = "左手" if free_zh == "右手" else "右手"
    preferred: list[str] = []
    pool: list[str] = []
    already_lookback = any(is_lookback_beat_action(p) for p in previous)
    climax_objs = extract_climax_object_phrases(context_text)
    already_climax = any(
        action_mentions_climax_object(p, context_text) for p in previous
    )
    # 开场镜：主题以走进场景起笔时，兜底必须行进，绝不能塞回头
    if index == 0 and topic_opens_with_locomotion(context_text):
        preferred.append(default_opening_locomotion_action(context_text))
    elif (
        topic_requires_lookback_climax(context_text)
        and not already_lookback
        and not already_climax
        and index > 0
    ):
        preferred.append(FACE_MICRO_LOOKBACK)
    elif climax_objs and already_lookback and not already_climax:
        # 回头之后：背后推镜 + 物件进入（比「微仰+正脸抬头」更稳）
        obj = climax_objs[0]
        preferred.extend(
            [
                f"镜头推向主体背部，{obj}从画面上方进入",
                f"{obj}从画面上方进入，镜头推向主体背部",
            ]
        )
    elif climax_objs and already_climax:
        preferred.extend(
            [
                face_micro_gaze_at_climax(context_text),
                f"背影纵深，目光锁定已入画的{climax_objs[0]}",
            ]
        )
    if grip_hand:
        pool.extend(
            [
                f"举起{free_other}抬到眉边向前扫描",
                f"{free_other}向前伸出做制止手势",
                f"放下{free_other}，目光仍盯前方",
            ]
        )
    pool.extend(_FALLBACK_MOTIONS)
    for fb in preferred:
        if not action_spacing_conflict(fb, previous, min_gap=min_gap):
            return fb
    rotated = pool[index % len(pool) :] + pool[: index % len(pool)] if pool else []
    for fb in rotated:
        text = fb.replace("空手", free_other)
        if is_lookback_beat_action(text) and already_lookback:
            continue
        if not action_spacing_conflict(text, previous, min_gap=min_gap):
            return text
    if preferred:
        return preferred[0]
    alts = (
        "侧脸微倾，目光仍盯前方",
        "上身微微侧倾，目光仍盯前方",
        f"举起{free_other}遮挡头顶强光",
    )
    return alts[index % len(alts)]


def looks_truncated(text: str) -> bool:
    cleaned = (text or "").strip().rstrip(".")
    if not cleaned:
        return False
    last = cleaned.split()[-1].lower().strip(".,;:'\"")
    return last in {
        "by", "to", "his", "her", "the", "a", "an", "and", "of", "from", "with",
        "on", "in", "at", "for", "into", "onto", "across", "as", "while",
    }


# ---------------------------------------------------------------------------
# 道具占手（轻量推断，供规则提示与结构校验；不写死主题）
# ---------------------------------------------------------------------------

def detect_handheld_prop(text: str) -> str | None:
    """检测手持道具名词；排除动词用法（如 shield his face）。"""
    match = _HANDHELD_PROP_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1)
    zh_map = {
        "伞": "umbrella", "雨伞": "umbrella", "剑": "sword", "刀": "blade",
        "枪": "gun", "手枪": "pistol", "步枪": "rifle", "武器": "weapon",
        "盾": "shield", "灯笼": "lantern", "火把": "torch", "拐杖": "cane",
        "包": "bag", "公文包": "briefcase", "矛": "spear", "弓": "bow",
        "斧": "axe", "锤": "hammer", "手机": "phone", "平板": "tablet",
        "鱼竿": "rod", "钓竿": "rod",
    }
    if raw in zh_map:
        return zh_map[raw]
    name = raw.lower().replace(" ", "_")
    # shield/hammer 等作动词时不当道具：shield his face / hammer the nail
    if name in {"shield", "hammer", "bow"}:
        after = (text or "")[match.end() : match.end() + 24].lower()
        if re.match(r"\s+(his|her|their|the|a|an|my|your|its)\b", after):
            return None
    if name in {"fishing_rod", "fish_rod", "pole"}:
        return "rod"
    if name == "canopy":
        return "umbrella"
    return name


def is_locomotion_action(action: str) -> bool:
    return bool(_LOCOMOTION_RE.search(action or ""))


def requires_face_readable_action(action: str) -> bool:
    """抬头/眼神/回头等必须在脸或清晰侧脸/过肩可见，否则背影下动作不可见。"""
    return bool(_FACE_READABLE_ACTION_RE.search(action or ""))


def prefers_rear_pushin(action: str, *, prev_was_locomotion: bool = False) -> bool:
    """
    上一镜纵深背影行进、本镜仍是背影可完成的小动作时：
    保持背影 + 缓慢镜头推进，避免同构图硬切换片。
    """
    if not prev_was_locomotion:
        return False
    text = action or ""
    if requires_face_readable_action(text) or is_locomotion_action(text):
        return False
    return bool(_REAR_PUSHIN_COMPAT_RE.search(text))


def motion_family(action: str) -> str:
    """主题无关的动作族，用于相邻镜去重（比指纹更粗）。"""
    text = action or ""
    if is_locomotion_action(text):
        return "locomotion"
    if _LOOK_UP_FAMILY_RE.search(text):
        return "look_up"
    if re.search(r"\b(stop gesture|extends?\s+\w+\s+hand\s+forward)\b", text, flags=re.I):
        return "stop_gesture"
    if re.search(r"\b(brow|scan ahead|shield eyes)\b", text, flags=re.I):
        return "scan_overhead"
    return extract_motion_signature(text) or "other"


def action_family_too_close(
    candidate: str,
    previous: list[str],
    *,
    min_gap: int = MIN_REPEAT_GAP,
) -> bool:
    fam = motion_family(candidate)
    if not fam or fam == "other" or min_gap <= 1:
        return False
    start = max(0, len(previous) - (min_gap - 1))
    for prev in previous[start:]:
        if motion_family(prev) == fam:
            return True
    return False


def opposite_hand(hand: str) -> str:
    return "left" if (hand or "").lower() == "right" else "right"


def infer_prop_grip(*texts: str, default_hand: str = "right") -> tuple[str | None, str | None]:
    blob = " ".join(str(t) for t in texts if t).strip()
    prop = detect_handheld_prop(blob)
    if not prop:
        return None, None
    lower = blob.lower()
    left = bool(re.search(
        rf"\bleft(?:\s+|-)?hand\b[^.]{{0,40}}\b(?:gripping|holding|holds|grips)\b[^.]{{0,30}}\b{re.escape(prop)}\b|"
        rf"\b{re.escape(prop)}\b[^.]{{0,40}}\bin (?:his |her |the )?left(?:\s+|-)?hand\b",
        lower,
    ))
    right = bool(re.search(
        rf"\bright(?:\s+|-)?hand\b[^.]{{0,40}}\b(?:gripping|holding|holds|grips)\b[^.]{{0,30}}\b{re.escape(prop)}\b|"
        rf"\b{re.escape(prop)}\b[^.]{{0,40}}\bin (?:his |her |the )?right(?:\s+|-)?hand\b",
        lower,
    ))
    if left and not right:
        return prop, "left"
    if right and not left:
        return prop, "right"
    return prop, default_hand


def action_uses_hand(action: str, hand: str) -> bool:
    if not action or not hand:
        return False
    return bool(
        re.search(
            rf"\b{hand}\s+(?:hand|arm|fist|palm|finger|fingers)\b|"
            rf"\b(?:raises?|lifts?|lowers?|extends?|points?)\s+(?:his |her |their )?{hand}\b",
            action,
            flags=re.IGNORECASE,
        )
    )


# ---------------------------------------------------------------------------
# 组装 / 轻量清洗（结构 only）
# ---------------------------------------------------------------------------

# 从用户主题抽取设定短语（中英关键词 → 英文画面词），写入锚点/首帧，避免 T2I 画成普通街景
_TOPIC_SETTING_RULES: list[tuple[tuple[str, ...], str]] = [
    (("赛博朋克", "cyberpunk"), "赛博朋克未来都市"),
    (("全息", "holographic", "hologram", "广告牌"), "全息霓虹广告牌"),
    (("霓虹", "neon"), "密集霓虹灯光"),
    (("雨夜", "阴雨", "绵绵", "rainy", "rain"), "雨夜大雨"),
    (("暗巷", "巷", "alley"), "狭窄湿滑暗巷"),
    (("风衣", "trench"), "黑色风衣"),
    (("雨伞", "伞", "umbrella"), "实体雨伞，边缘柔光"),
    (("赏金猎人", "bounty"), "赏金猎人"),
]
# 高潮物件：规则表仅作别名归一；真正识别以主题结构句为主（主题无关）
_TOPIC_CLIMAX_RULES: list[tuple[tuple[str, ...], str]] = [
    (("无人机", "drone", "uav"), "无人机"),
    (("金鱼", "鱼", "fish"), "鱼"),
]
# 从主题句式抽取高潮物件（不依赖题材白名单）
_CLIMAX_TOPIC_STRUCT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"([\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{1,24})"
        r"(?:从天而降|从空中降下|从空中降|从上方降下|忽然出现|突然出现|飞扑而来|飞来|追来|降临)"
    ),
    re.compile(
        r"(?:从天而降|忽然出现|突然出现|从空中降下|从空中降)(?:的|了|一[架只个头])?"
        r"([\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{1,24})"
    ),
    re.compile(
        r"\b([A-Za-z][A-Za-z0-9_-]{1,24})\s+"
        r"(?:descends?(?:\s+from)?|appears?|flies?\s+in|drops?\s+in|swoops?\s+in)\b",
        flags=re.I,
    ),
)
_CLIMAX_STRUCT_STOPWORDS = frozenset(
    {
        "他", "她", "它", "他们", "主角", "人物", "猎人", "有人", "某人",
        "然后", "忽然", "突然", "开始", "发现", "察觉", "回头", "转身",
        "走进", "走入", "森林", "山谷", "暗巷", "街道",
        "he", "she", "they", "someone", "the", "a", "an", "and", "then",
    }
)
_CLIMAX_STRUCT_PREFIX_RE = re.compile(r"^(?:忽然|突然|然后|发现|察觉)+")


def extract_setting_phrases(user_topic: str, *, include_climax: bool = False) -> list[str]:
    text = (user_topic or "").lower()
    # 中文主题也做原文匹配
    raw = user_topic or ""
    phrases: list[str] = []
    rules = list(_TOPIC_SETTING_RULES)
    if include_climax:
        rules.extend(_TOPIC_CLIMAX_RULES)
    for keys, phrase in rules:
        if any(k.lower() in text or k in raw for k in keys):
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def extract_climax_object_phrases(user_topic: str) -> list[str]:
    """
    主题中的高潮物件名词（全局）。
    优先：主题结构句抽取；其次：别名规则表归一。不绑定单一题材。
    """
    raw = user_topic or ""
    text = raw.lower()
    phrases: list[str] = []

    def _add(p: str) -> None:
        p = _CLIMAX_STRUCT_PREFIX_RE.sub("", (p or "").strip(" ，,。．、的了"))
        if not p or len(p) < 2:
            return
        if p.lower() in _CLIMAX_STRUCT_STOPWORDS or p in _CLIMAX_STRUCT_STOPWORDS:
            return
        if p not in phrases:
            phrases.append(p)

    for rx in _CLIMAX_TOPIC_STRUCT_RES:
        for m in rx.finditer(raw):
            _add(m.group(1))

    for keys, phrase in _TOPIC_CLIMAX_RULES:
        if any(k.lower() in text or k in raw for k in keys):
            # 别名表允许单字（如「鱼」）
            p = (phrase or "").strip()
            if p and p not in phrases:
                phrases.append(p)
    return phrases


def strip_climax_phrases_from_text(text: str, user_topic: str = "") -> str:
    """
    从锚点/身份文案去掉高潮物件名词。
    开场与角色卡不应出现后段才入画的物件，否则 T2I/I2V 会提前画成静止道具。
    """
    cleaned = text or ""
    phrases = extract_climax_object_phrases(user_topic)
    # 即使 user_topic 为空，也清掉锚点里已写入的常见高潮词
    for keys, phrase in _TOPIC_CLIMAX_RULES:
        if phrase not in phrases:
            phrases.append(phrase)
        for k in keys:
            if k not in phrases:
                phrases.append(k)
    for p in phrases:
        if not p:
            continue
        cleaned = re.sub(re.escape(p), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[，,]\s*[，,]+", "，", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")
    return cleaned


def should_include_climax_in_keyframe(
    user_topic: str,
    action_hint: str = "",
    shot_index: int = 0,
) -> bool:
    """
    Scene1 不强迫高潮物件入画；仅当本镜动作明确提到该物件或「进入画面」时才注入。
    """
    phrases = extract_climax_object_phrases(user_topic)
    if not phrases or shot_index <= 0:
        return False
    action = action_hint or ""
    if any(p in action for p in phrases):
        return True
    if is_climax_enter_action(action, user_topic):
        return True
    # 不再用「上方/追踪」模糊匹配，避免抬头察觉镜误注入物件
    return False


def enrich_visual_anchor(anchor: str, user_topic: str) -> str:
    """把主题场景设定补进锚点；高潮物件不进锚点（后段动作/首帧再注入）。"""
    base = strip_climax_phrases_from_text(normalize_visual_anchor(anchor), user_topic)
    phrases = extract_setting_phrases(user_topic, include_climax=False)
    if not phrases:
        return base
    lower = base.lower()
    missing = []
    for p in phrases:
        if re.search(r"[\u4e00-\u9fff]", p):
            if p not in base and p not in lower:
                missing.append(p)
        elif not all(tok in lower for tok in p.lower().split()[:2]):
            missing.append(p)
    if not missing:
        return base
    sep = "，" if re.search(r"[\u4e00-\u9fff]", base + "".join(missing)) else ", "
    merged = f"{sep.join(missing)}{sep}{base}" if base else sep.join(missing)
    merged = limit_words(normalize_visual_anchor(merged), MAX_ANCHOR_WORDS + 8)
    # 词级去重：拼接后可能产生重复词，保留首次出现
    if re.search(r"[\u4e00-\u9fff]", merged):
        chunks = re.split(r"[，,]\s*", merged)
        seen: set[str] = set()
        out: list[str] = []
        for c in chunks:
            key = c.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return strip_climax_phrases_from_text(
            normalize_visual_anchor("，".join(out)), user_topic
        )
    tokens = merged.split()
    seen_tok: set[str] = set()
    deduped: list[str] = []
    for tok in tokens:
        bare = tok.lower().strip(".,;")
        if not bare or bare in seen_tok:
            continue
        seen_tok.add(bare)
        deduped.append(tok)
    return strip_climax_phrases_from_text(
        normalize_visual_anchor(" ".join(deduped)), user_topic
    )


def normalize_visual_anchor(anchor: str) -> str:
    text = re.sub(r"\s+", " ", (anchor or "").strip()).strip(" ,.;")
    text = re.sub(r"\bholding\s+(a\s+)?", "with ", text, flags=re.IGNORECASE)
    # glowing umbrella → 半透明伞+柔边光，降低 Wan/wanx 过曝与光剑误读
    text = re.sub(
        r"\bglowing\s+(?:blue\s+)?(?:transparent\s+)?umbrella\b",
        "physical rain umbrella with soft rim light, no overexposed bloom",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bwith\s+glowing\s+(?:blue\s+)?umbrella\b",
        "with physical rain umbrella soft rim light",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\btranslucent rain umbrella(?:\s+with\s+soft\s+cyan\s+rim\s+light)?\b",
        "physical rain umbrella with soft rim light",
        text,
        flags=re.IGNORECASE,
    )
    return text


def boost_i2v_motion_prompt(
    action: str,
    *,
    prev_was_locomotion: bool = False,
    rear_pushin: bool = False,
    context_text: str = "",
) -> str:
    """按动作类型强化主体运动（主题无关的结构规则）。"""
    text = re.sub(r"\s+", " ", (action or "").strip()).strip(" ,.;，。；")
    if not text:
        return text
    if re.search(
        r"(主体向纵深连续迈步前行|腿部交替迈步|五秒镜头内做完该动作|"
        r"允许推镜视差|仅允许推镜视差|"
        r"subject locomotion with shifting|clear visible subject motion|"
        r"slow camera push-in)",
        text,
        flags=re.I,
    ):
        return text

    # 非行进镜：只约束「做完动作」，不加方式副词（慢慢/清晰等易干扰 I2V）
    duration = "五秒镜头内做完该动作"
    subject_read = "主体受光可读，不是纯黑剪影"
    loco_bg = (
        "允许纵深视差：主体远离镜头时在画面中明显变小，"
        "走到巷道/场景更深处，禁止场景融化变形，"
        "以主体大位移为主"
    )
    still_bg = (
        "背景尽量静止，禁止环境融化变形，"
        "以主体运动为主"
    )
    ctx = context_text or ""

    if is_locomotion_action(text):
        # 观感「慢动作」通常不是 16fps，而是五秒内几乎没走远（原地迈腿）
        return (
            f"{text}，主体向纵深大步连续前行，腿部交替大步迈步，"
            f"五秒内约迈八到十步并明显走到更远处，"
            f"主体在画面中明显变小、身体位置持续改变，"
            f"禁止原地踏步，禁止贴地蹭步，禁止只动腿不前进，禁止滑行，禁止跪倒趴下，"
            f"{subject_read}，主体完整可见，{loco_bg}"
        )

    # 物件进入须先于通用 rear_pushin：结构规则，不绑定题材
    if prefers_reveal_pushin(text, ctx) or is_climax_enter_action(text, ctx):
        stop = "停止前行，" if prev_was_locomotion else ""
        return (
            f"{text}，{stop}保持背影或过肩纵深构图，"
            f"镜头推向主体背部，物件从画面上方进入，"
            f"允许推镜视差，禁止背景融化，主体有可见运动，{duration}，"
            f"{subject_read}，完整可见，不消失，不对镜头注视"
        )

    # 见脸硬切：定妆首帧 + 微动，禁止大角度转头过程（开源 I2V 崩眼）
    if prefers_face_micro_motion(text, ctx):
        stop = "停止前行，" if prev_was_locomotion else ""
        return (
            f"{text}，{stop}首帧已是过肩或侧脸定妆，"
            f"仅做极小幅头动与眼神变化，禁止大幅度转头，禁止拧脸，"
            f"禁止双眼不对称或五官变形，主体有可见微动，{duration}，"
            f"{subject_read}，完整可见，不消失，不对镜头注视，{still_bg}"
        )

    if rear_pushin or prefers_rear_pushin(text, prev_was_locomotion=prev_was_locomotion):
        stop = "停止前行，" if prev_was_locomotion else ""
        return (
            f"{text}，{stop}保持与上一镜连续的背影纵深构图，"
            f"镜头推向主体背部，主体有可见运动，{duration}，"
            f"{subject_read}，完整可见，不消失，不对镜头注视，"
            f"环境底板锁定，仅允许推镜视差，禁止场景融化"
        )

    # 纯仰视（无物件进入）：轻量微仰
    if prefers_tilt_up_camera(text, ctx):
        stop = "停止前行，" if prev_was_locomotion else ""
        return (
            f"{text}，{stop}"
            f"镜头微仰跟随主体视线，主体有可见运动，{duration}，"
            f"{subject_read}，完整可见，不消失，不对镜头注视，{still_bg}"
        )

    stop = "停止前行，" if prev_was_locomotion else ""
    return (
        f"{text}，{stop}主体有可见运动，{duration}，"
        f"{subject_read}，完整可见，不消失，不对镜头注视，{still_bg}"
    )


def topic_authorizes_handheld_gadget(context_text: str) -> bool:
    """主题/锚点是否授权出现枪械或可瞄准器械（主题无关的许可检测）。"""
    return bool(_TOPIC_HANDHELD_AUTH_RE.search(context_text or ""))


def topic_authorizes_seize_object(context_text: str) -> bool:
    """主题是否授权捡起/拿起外部物体。"""
    return bool(_TOPIC_SEIZE_AUTH_RE.search(context_text or ""))


def topic_requires_lookback_climax(context_text: str) -> bool:
    """主题是否要求「察觉后回头/锐利注视」类高潮。"""
    return bool(_TOPIC_LOOKBACK_RE.search(context_text or ""))


def topic_opens_with_locomotion(context_text: str) -> bool:
    """
    主题开篇是否在建立走进/进入场景。
    若是，则第 1 镜必须是行进，不能把回头/察觉提前到开场。
    """
    text = strip_duration_directives(context_text or "").strip()
    if not text:
        return False
    # 取高潮事件之前的开篇段（不只看第一句，避免「环境句。人物走进…」漏检）
    head = re.split(
        r"突然|忽然|察觉|猛然回头|开始追踪|从天而降|"
        r"\bsuddenly\b|\bthen\b|\blooks?\s+back\b|\bnotices?\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    if not head:
        head = text[:120]
    return bool(_OPENING_LOCO_RE.search(head))


def default_opening_locomotion_action(context_text: str) -> str:
    """从主题开篇抽取行进动作；抽不到则用通用纵深走进。"""
    text = strip_duration_directives(context_text or "").strip()
    m = re.search(r"((?:走入|走进|迈进)[^，。；!\n?]{2,28})", text)
    if m:
        return m.group(1).strip(" ，,。.;；")
    m = re.search(
        r"((?:walks?|enters?|steps?)\s+(?:into|toward|towards|down|through)[^.,;!\n?]{2,40})",
        text,
        flags=re.I,
    )
    if m:
        return m.group(1).strip(" .,;")
    return "向纵深走进场景"


def strip_invented_handheld_action(action: str, context_text: str = "") -> str:
    """
    全局：凭空掏枪/装置/捡起外部追踪物（主题未授权）→ 清空走结构兜底。
    """
    cleaned = (action or "").strip()
    if not cleaned:
        return cleaned
    if not topic_authorizes_seize_object(context_text) and _INVENTED_SEIZE_OBJECT_RE.search(cleaned):
        return ""
    if not topic_authorizes_handheld_gadget(context_text):
        cleaned = re.sub(
            r",?\s*(?:emitting|firing|shooting)\s+(?:a\s+)?(?:beam of light|laser beam|energy beam).*$",
            "",
            cleaned,
            flags=re.I,
        ).strip(" ,.;，。；")
        cleaned = re.sub(
            r"[，,]?\s*(?:射出|发射|打出)\s*(?:一道)?\s*(?:光束|激光|能量束).*$",
            "",
            cleaned,
        ).strip(" ,.;，。；")
        if _INVENTED_HANDHELD_ACTION_RE.search(cleaned):
            return ""
    else:
        cleaned = re.sub(
            r",?\s*(?:emitting|firing|shooting)\s+(?:a\s+)?(?:beam of light|laser beam|energy beam)\b.*$",
            "",
            cleaned,
            flags=re.I,
        ).strip(" ,.;，。；")
        cleaned = re.sub(
            r"[，,]?\s*(?:射出|发射|打出)\s*(?:一道)?\s*(?:光束|激光|能量束).*$",
            "",
            cleaned,
        ).strip(" ,.;，。；")
    if _LOOK_AT_CAMERA_RE.search(cleaned):
        cleaned = _LOOK_AT_CAMERA_RE.sub("看向画外目标", cleaned)
    return cleaned.strip(" ,.;，。；")


def sanitize_shot_motion(
    text: str,
    grip_hand: str | None = None,
    prop_name: str | None = None,
    previous: list[str] | None = None,
    index: int = 0,
    context_text: str = "",
) -> str:
    """仅修结构残句；语义改写交给 LLM 规则。"""
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" ,.;，。；")
    if not cleaned:
        return cleaned

    cleaned = strip_invented_handheld_action(cleaned, context_text=context_text)
    if not cleaned:
        return pick_fallback_motion(
            previous, grip_hand=grip_hand, index=index, context_text=context_text
        )

    # 开场镜：主题以走进/进入起笔时，强制第 1 镜为行进（禁止回头/察觉抢开场）
    if index == 0 and topic_opens_with_locomotion(context_text):
        if is_lookback_beat_action(cleaned) or not is_locomotion_action(cleaned):
            return default_opening_locomotion_action(context_text)

    # 行进句去掉方式副词（稳步/慢慢/清晰等），只留动词本身（主题无关）
    if is_locomotion_action(cleaned):
        cleaned = re.sub(
            r"(?:稳步|缓步|慢慢|缓慢地?|小心翼翼地?|清晰地?|快速地?|飞快地?)"
            r"(?:地)?(?=(?:走|迈|行))",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\b(?:slowly|steadily|cautiously|carefully|clearly|quickly|rapidly)\s+"
            r"(?=(?:walks?|steps?|strides?|advances?)\b)",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")

    # 主题高潮物件：禁止只用「画外」暗示，改为入画可读（全局）
    if extract_climax_object_phrases(context_text):
        cleaned = _OFFSCREEN_TARGET_RE.sub("入画", cleaned)
        cleaned = re.sub(r"入画的?追踪物", "入画的追踪物", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")

    # 回头高潮预算：已有回头镜则不再保留回头族动作；开场镜也不允许回头抢戏
    # 例外：高潮后「侧脸微动注视已入画」收束镜不算重复回头
    if is_lookback_beat_action(cleaned) and (
        (previous and any(is_lookback_beat_action(p) for p in previous))
        or (index == 0 and topic_opens_with_locomotion(context_text))
    ):
        already_climax = bool(
            previous
            and any(is_climax_enter_action(p, context_text) for p in previous)
        )
        post_climax_gaze = already_climax and (
            action_mentions_climax_object(cleaned, context_text)
            or bool(re.search(r"注视已入画|目光锁定已入画|侧脸微动", cleaned))
        )
        if post_climax_gaze:
            return face_micro_gaze_at_climax(context_text)
        return pick_fallback_motion(
            previous, grip_hand=grip_hand, index=index, context_text=context_text
        )

    # 高潮物件进入画面全片最多一次：后续镜改为侧脸微动注视（无过肩，不占回头预算）
    if (
        previous
        and is_climax_enter_action(cleaned, context_text)
        and any(is_climax_enter_action(p, context_text) for p in previous)
    ):
        return face_micro_gaze_at_climax(context_text)

    # 物件进入：去掉「抬头注视」正脸倾向，统一背后推镜入画（主题无关）
    if is_climax_enter_action(cleaned, context_text):
        cleaned = re.sub(r"[，,]?\s*主体抬头注视", "", cleaned)
        cleaned = re.sub(r"[，,]?\s*抬头注视", "", cleaned)
        if not re.search(r"推向|推镜|背部", cleaned):
            cleaned = f"镜头推向主体背部，{cleaned}"
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")
        return cleaned.strip(" ,.;，。；")

    # 察觉桥接：抬头见脸会逼 I2V 从背影拧出正脸 → 改为背影微仰（见脸留给唯一过肩微动镜）
    if (
        prefers_tilt_up_camera(cleaned, context_text)
        and not is_lookback_beat_action(cleaned)
        and not is_climax_enter_action(cleaned, context_text)
    ):
        return REAR_NOTICE_TILT

    cleaned = re.sub(r"\b(raises?|lifts?|lowers?)\s+it\b", r"\1 hand", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\braises?\s+free\s+hand\s+to\s+(raises?\s+(?:left|right)\s+hand)\b",
        r"\1",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\braises?\s+(left|right)\s+hand\s+it\b", r"raises \1 hand", cleaned, flags=re.I)
    cleaned = re.sub(r"(?:举起|抬起|放下)\s*它\b", "抬起空手", cleaned)
    cleaned = re.sub(r"\bwith\s*$", "", cleaned, flags=re.I).strip(" ,.;，。；")

    # 大转身 / 猛拧头 / 任意回头族 → 硬切定妆微动（或高潮后侧脸注视）
    already_climax = bool(
        previous and any(is_climax_enter_action(p, context_text) for p in previous)
    )
    if is_large_head_or_body_turn(cleaned) or is_lookback_beat_action(cleaned):
        if already_climax and (
            action_mentions_climax_object(cleaned, context_text)
            or bool(re.search(r"面向|注视|看向", cleaned))
        ):
            return face_micro_gaze_at_climax(context_text)
        if previous and any(is_lookback_beat_action(p) for p in previous):
            return pick_fallback_motion(
                previous, grip_hand=grip_hand, index=index, context_text=context_text
            )
        return FACE_MICRO_LOOKBACK

    # 「迅速转身面向X」等残留 → 侧脸微动注视（无过肩，避免占回头预算）
    if re.search(r"(?:迅速|突然|猛然)?转身[，,]?面向(.+)", cleaned):
        if already_climax:
            return face_micro_gaze_at_climax(context_text)
        return FACE_MICRO_LOOKBACK

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")

    # 通用兜底：禁后退动作。I2V 对后退步态几乎必崩（月步/劈叉/倒走）。
    if (
        re.search(r"\b(backward|backwards|retreat|retreats|reverse|reversing)\b", cleaned, flags=re.I)
        or re.search(
            r"\b(steps?|walks?|moves?|backs?)\s+(back|backward|backwards|away|away from)\b",
            cleaned,
            flags=re.I,
        )
        or re.search(r"后退|倒退|倒走|退后|往后退|退开|退离", cleaned)
    ):
        cleaned = re.sub(
            r"\b(takes?\s+|a\s+)?(quick\s+|slow\s+|small\s+)?(step|steps)\s+(back|backward|backwards)\b",
            "steps forward",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\b(walks?|moves?|backs?)\s+(back|backward|backwards|away(?:\s+from)?)\b",
            "walks forward",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\b(retreat|retreats|retreating|reverse|reversing|backward|backwards)\b",
            "advances forward",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"(?:猛然|突然|迅速)?(?:后退|倒退|倒走|退后几?步|往后退|退开|退离)", "向前迈步", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;，。；")

    if _STATIC_ACTION_RE.search(cleaned) or not _MOTION_VERB_RE.search(cleaned):
        # 运镜/物件入画/定妆微动也是合法可见变化
        if not (
            is_climax_enter_action(cleaned, context_text)
            or prefers_reveal_pushin(cleaned, context_text)
            or prefers_face_micro_motion(cleaned, context_text)
            or re.search(
                r"镜头推向|推镜|从画面上方进入|进入画面|微动|镜头微仰|保持背影",
                cleaned,
            )
        ):
            return pick_fallback_motion(
                previous, grip_hand=grip_hand, index=index, context_text=context_text
            )

    # 占用手做手势 → 空闲手（结构修复；抬/拉道具本身保留）
    if grip_hand and prop_name:
        free = opposite_hand(grip_hand)
        free_zh = {"left": "左", "right": "右"}.get(free, "右")
        grip_zh = {"left": "左", "right": "右"}.get((grip_hand or "").lower(), "右")
        prop_arm = bool(
            re.search(
                rf"\b(raises?|lifts?|lowers?|pulls?)\s+(?:the\s+)?(?:{re.escape(prop_name)}|rod tip)\b",
                cleaned,
                flags=re.I,
            )
            or re.search(rf"(?:举起|抬起|放下|拉动).{{0,6}}{re.escape(prop_name)}", cleaned)
        )
        if not prop_arm:
            cleaned = re.sub(
                rf"\b(raises?|lifts?|lowers?|extends?)\s+(?:his |her |their )?{grip_hand}\s+(hand|arm)\b",
                rf"\1 {free} \2",
                cleaned,
                flags=re.I,
            )
            cleaned = re.sub(
                rf"(?:举起|抬起|放下|伸出){grip_zh}(?:手|臂)",
                f"举起{free_zh}手",
                cleaned,
            )

    # 若清洗后与已有镜完全重复或指纹过近，换结构兜底（主题无关）
    if previous is not None and action_spacing_conflict(cleaned, previous):
        return pick_fallback_motion(
            previous, grip_hand=grip_hand, index=index, context_text=context_text
        )

    return cleaned.strip(" ,.;，。；")


def compose_i2v_prompt(
    visual_anchor: str,
    action: str,
    shot_index: int,
    beat_action: str = "",
    context_text: str = "",
    previous: list[str] | None = None,
) -> str:
    visual_anchor = normalize_visual_anchor(visual_anchor)
    _, grip_hand = infer_prop_grip(visual_anchor)
    ctx = context_text or visual_anchor
    # 必须传入 shot_index 与 previous：否则入画镜会被误判成无动词而改写成回头
    source = sanitize_shot_motion(
        (action or beat_action).strip(),
        grip_hand=grip_hand,
        context_text=ctx,
        index=shot_index,
        previous=previous,
    )
    source = limit_words(source, MAX_I2V_ACTION_WORDS)
    identity = limit_words(visual_anchor, MAX_I2V_ANCHOR_WORDS)

    # 中文动作：若未以主体起句则补「他」（运镜/物件入画句不加「他」）
    camera_led = bool(
        re.match(
            r"^(?:镜头|推镜|运镜|摄像机|相机|画面)",
            source,
        )
        or re.search(r"从画面上方进入|进入画面|飞入画面", source)
    )
    if source and not camera_led and not re.match(
        r"^(他|她|他们|主体|猎人|人物|He|She|They)\b",
        source,
        flags=re.I,
    ):
        source = re.sub(
            r"^(?:the\s+)?[A-Za-z]+(?:\s+[A-Za-z]+)?\s+"
            r"(?=(?:walks?|turns?|raises?|pulls?|leans?|points?|extends?|"
            r"steps?|snaps?|tilts?|lowers?|aims?|advances?|bends?)\b)",
            "",
            source,
            count=1,
            flags=re.I,
        ).strip()
        if source and re.search(r"[\u4e00-\u9fff]", source):
            if not source.startswith(("他", "她", "主体")):
                source = f"他{source}"
        elif source and not re.match(r"^(he|she|they)\b", source, flags=re.I):
            source = f"He {source[0].lower()}{source[1:]}" if source else source
    source = source.rstrip(" .。") + ("。" if re.search(r"[\u4e00-\u9fff]", source) else ".")

    if shot_index == 0:
        text = f"{identity}。{source}" if identity and source and re.search(r"[\u4e00-\u9fff]", identity + source) else (
            f"{identity}. {source}" if identity and source else (identity or source)
        )
    else:
        lock = "同一主体同一外观" if re.search(r"[\u4e00-\u9fff]", source or identity) else "same subject same appearance"
        sep = "。" if re.search(r"[\u4e00-\u9fff]", identity + source) else ". "
        text = f"{identity}{sep}{source} {lock}。" if identity else f"{source} {lock}。"
    return limit_words(text, MAX_I2V_TOTAL_WORDS).rstrip(" .。") + ("。" if re.search(r"[\u4e00-\u9fff]", text) else ".")


def compose_keyframe_prompt(visual_anchor: str, keyframe_prompt: str) -> str:
    anchor = limit_words(normalize_visual_anchor(visual_anchor), MAX_ANCHOR_WORDS)
    frame = limit_words_at_clause((keyframe_prompt or "").strip(), MAX_KEYFRAME_WORDS)
    if not frame or frame.lower() in {"none", "n/a", "无", "无。"}:
        frame = "中景全身，四分之三侧，双脚可见，自然站姿"
    use_zh = bool(re.search(r"[\u4e00-\u9fff]", (anchor or "") + (frame or "")))
    sep = "。" if use_zh else ". "
    end = "。" if use_zh else "."
    if anchor and anchor.lower() not in frame.lower() and (not use_zh or anchor not in frame):
        remaining = max(8, MAX_KEYFRAME_WORDS - (len(anchor) // 2 if use_zh else len(anchor.split())) - 1)
        frame = limit_words_at_clause(frame, remaining)
        return f"{anchor}{sep}{frame}".rstrip(" .。") + end
    return limit_words_at_clause(frame, MAX_KEYFRAME_WORDS).rstrip(" .。") + end


# ---------------------------------------------------------------------------
# 轻量结构校验（不做主题词表）
# ---------------------------------------------------------------------------

def validate_shot_diversity(
    raw_shots: list[str],
    *,
    min_gap: int = MIN_REPEAT_GAP,
) -> tuple[bool, str]:
    """
    全局规则（主题无关）：
    - 允许相似动作
    - 禁止完全相同的动作文案（归一化后）
    - 相同动作指纹不得靠得过近（默认不相邻）
    """
    seen: list[str] = []
    for i, shot in enumerate(raw_shots):
        text = (shot or "").strip()
        if action_is_exact_duplicate(text, seen):
            prev_idx = next(
                j
                for j, prev in enumerate(seen)
                if normalize_action_key(prev) == normalize_action_key(text)
            )
            return False, (
                f"shots {prev_idx + 1} and {i + 1} are exact duplicates: '{text}'"
            )
        if action_signature_too_close(text, seen, min_gap=min_gap):
            sig = extract_motion_signature(text)
            return False, (
                f"shots near {i + 1} reuse motion signature '{sig}' "
                f"within gap<{min_gap} (similar OK, identical fingerprint too close)"
            )
        seen.append(text)
    return True, "ok"


def validate_beat_action_diversity(beats: list[dict]) -> tuple[bool, str]:
    return validate_shot_diversity([str(b.get("action", "")) for b in beats])


def validate_spine_walking_streak(beats: list[dict]) -> tuple[bool, str]:
    streak = 0
    for index, beat in enumerate(beats, start=1):
        action = str(beat.get("action", "")).lower()
        is_walk = bool(
            (
                re.search(r"\b(walk|step|stride)\b", action)
                and re.search(r"\b(forward|ahead|into)\b", action)
            )
            or re.search(r"(?:走进|走入|向前|前行|迈步)", action)
        )
        if is_walk and not re.search(
            r"\b(turn|head|hand|point)\b|转头|回头|举手|抬手", action
        ):
            streak += 1
            if streak > MAX_CONSECUTIVE_WALKING_BEATS:
                return False, f"too many consecutive walking beats near beat {index}"
        else:
            streak = 0
    return True, "ok"


def validate_lookback_budget(beats: list[dict]) -> tuple[bool, str]:
    """回头/过肩察觉：全片最多 N 镜，且不可连续堆叠。"""
    streak = 0
    total = 0
    for index, beat in enumerate(beats, start=1):
        action = str(beat.get("action", ""))
        if is_lookback_beat_action(action):
            streak += 1
            total += 1
            if streak > MAX_CONSECUTIVE_LOOKBACK_BEATS:
                return False, (
                    f"consecutive look-back beats near beat {index} "
                    f"(max {MAX_CONSECUTIVE_LOOKBACK_BEATS})"
                )
        else:
            streak = 0
    if total > MAX_LOOKBACK_CLIMAX_BEATS:
        return False, (
            f"look-back climax at most {MAX_LOOKBACK_CLIMAX_BEATS} beat(s); "
            f"got {total} — progress the story after one look-back"
        )
    return True, "ok"


def validate_opening_locomotion(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """主题开篇是走进场景时，第 1 镜必须是行进，不能被回头/察觉抢走。"""
    if not topic_opens_with_locomotion(user_topic):
        return True, "ok"
    if not beats:
        return False, "missing beats for opening locomotion"
    action = str(beats[0].get("action", "")).strip()
    if is_lookback_beat_action(action):
        return False, (
            "beat 1 must establish opening walk into the scene; "
            "do not put look-back in beat 1"
        )
    if not is_locomotion_action(action):
        return False, (
            "beat 1 must be locomotion when USER_INPUT opens with walking/entering; "
            f"got '{action}'"
        )
    return True, "ok"


def validate_climax_object_on_screen(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """
    主题写了高潮物件时：至少一镜让该物件入画可读，禁止全程「画外」暗示。
    """
    phrases = extract_climax_object_phrases(user_topic)
    if not phrases:
        return True, "ok"
    for beat in beats:
        action = str(beat.get("action", ""))
        end_state = str(beat.get("end_state", ""))
        blob = f"{action} {end_state}"
        if _OFFSCREEN_TARGET_RE.search(blob) and not re.search(r"入画|进入画面", blob):
            continue
        if any(p in blob for p in phrases):
            return True, "ok"
        if re.search(r"入画的?(?:追踪物|目标)|可见的?(?:追踪|飞行)", blob):
            return True, "ok"
    return False, (
        "topic climax object must appear on-screen in at least one beat "
        f"({', '.join(phrases)}); do not keep it off-screen only"
    )


def validate_climax_reveal_staging(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """
    高潮物件出场须铺垫：先察觉/抬头/回头，再「进入画面」；
    禁止首镜出现，禁止第一句就是已占位凝视。
    """
    phrases = extract_climax_object_phrases(user_topic)
    if not phrases:
        return True, "ok"

    first_idx = None
    for i, beat in enumerate(beats):
        action = str(beat.get("action", ""))
        if action_mentions_climax_object(action, user_topic):
            first_idx = i
            break
    if first_idx is None:
        return True, "ok"  # 由 on_screen 校验负责缺席

    if first_idx == 0:
        return False, "climax object must not appear in beat 1; establish the scene first"

    first_action = str(beats[first_idx].get("action", ""))
    if not _CLIMAX_ENTER_RE.search(first_action):
        return False, (
            "climax object first appearance must enter the frame "
            "(e.g. descends into shot), not already parked on-screen"
        )

    prev = str(beats[first_idx - 1].get("action", ""))
    if action_mentions_climax_object(prev, user_topic):
        return False, "need a notice/look beat before the climax object enters"
    bridge_ok = bool(
        is_lookback_beat_action(prev)
        or requires_face_readable_action(prev)
        or re.search(
            r"抬头|仰头|转头|察觉|警觉|回望|注视上方|看向天空|动静|观察",
            prev,
        )
    )
    if not bridge_ok:
        return False, (
            "before climax object enters, need a notice / look-up / look-back bridge beat"
        )
    return True, "ok"


def validate_climax_enter_once(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """高潮物件「进入画面」全片最多一镜，禁止连镜重复飞入。"""
    if not extract_climax_object_phrases(user_topic):
        return True, "ok"
    idxs = [
        i
        for i, beat in enumerate(beats, start=1)
        if is_climax_enter_action(str(beat.get("action", "")), user_topic)
    ]
    if len(idxs) > 1:
        return False, (
            f"climax object may enter the frame in at most one beat; "
            f"got enters at beats {idxs}"
        )
    return True, "ok"


def validate_spine_safety(
    beats: list[dict],
    grip_hand: str | None = None,
    prop_name: str | None = None,
) -> tuple[bool, str]:
    hard_ok, hard_reason = validate_spine_structure(beats)
    if not hard_ok:
        return False, hard_reason
    soft_ok, soft_reason = validate_spine_occupancy(beats, grip_hand=grip_hand, prop_name=prop_name)
    if not soft_ok:
        return False, soft_reason
    return True, "ok"


def validate_spine_structure(beats: list[dict]) -> tuple[bool, str]:
    """硬结构：空动作 / 残句 / 静止。"""
    for index, beat in enumerate(beats, start=1):
        action = str(beat.get("action", "")).strip()
        if not action:
            return False, f"beat {index} missing action"
        if looks_truncated(action):
            return False, f"beat {index} action looks truncated: '{action}'"
        if re.search(r"\b(raises?|lifts?)\s+it\b", action, flags=re.I):
            return False, f"beat {index} has broken pronoun 'it': '{action}'"
        if _STATIC_ACTION_RE.search(action):
            return False, f"beat {index} is static/motionless: '{action}'"
    return True, "ok"


def validate_spine_occupancy(
    beats: list[dict],
    grip_hand: str | None = None,
    prop_name: str | None = None,
) -> tuple[bool, str]:
    """占手启发式（出片档仅警告）。"""
    if grip_hand is None or prop_name is None:
        blob = " ".join(f"{b.get('action', '')} {b.get('end_state', '')}" for b in beats)
        prop_name, grip_hand = infer_prop_grip(blob)

    for index, beat in enumerate(beats, start=1):
        action = str(beat.get("action", "")).strip()
        if prop_name and (_BOTH_HANDS_DOWN_RE.search(action) or _BOTH_HANDS_DOWN_ZH_RE.search(action)):
            return False, f"beat {index} lowers both hands while holding {prop_name}"
        if grip_hand and prop_name:
            prop_arm = bool(
                re.search(
                    rf"\b(raises?|lifts?|lowers?|pulls?)\s+(?:the\s+)?(?:{re.escape(prop_name)}|rod tip)\b",
                    action,
                    flags=re.I,
                )
            )
            if (
                not prop_arm
                and action_uses_hand(action, grip_hand)
                and re.search(r"\b(raises?|lifts?|points?|collar|brow|hat)\b", action, flags=re.I)
            ):
                return False, (
                    f"beat {index} gestures with occupied {grip_hand} hand "
                    f"(holding {prop_name}) — use free {opposite_hand(grip_hand)} hand, "
                    f"or first stow the prop: '{action}'"
                )
    return True, "ok"


def dedupe_adjacent_actions(
    actions: list[str],
    *,
    min_gap: int = MIN_REPEAT_GAP,
    max_consecutive_locomotion: int = MAX_CONSECUTIVE_LOCOMOTION,
) -> list[str]:
    """本地轻改（主题无关）：
    - 允许相似动作
    - 禁止完全相同文案
    - 相同动作指纹必须拉开镜距
    - 连续纯行进镜不超过上限（防注水），不限制全片行进总次数
    """
    out: list[str] = []
    loco_streak = 0
    for index, action in enumerate(actions):
        candidate = (action or "").strip()
        if not candidate:
            candidate = pick_fallback_motion(out, index=index, min_gap=min_gap)

        needs_replace = action_spacing_conflict(candidate, out, min_gap=min_gap)
        if is_locomotion_action(candidate) and loco_streak >= max_consecutive_locomotion:
            needs_replace = True

        if needs_replace:
            replacement = pick_fallback_motion(out, index=index, min_gap=min_gap)
            # 连续行进时优先换非行进兜底
            if is_locomotion_action(candidate) and loco_streak >= max_consecutive_locomotion:
                for fb in _FALLBACK_MOTIONS:
                    text = fb.replace("free hand", "right hand")
                    if is_locomotion_action(text):
                        continue
                    if not action_spacing_conflict(text, out, min_gap=min_gap):
                        replacement = text
                        break
            if replacement != candidate:
                print(
                    f"[info] 间距/去重：镜头 {index + 1} "
                    f"'{action}' → '{replacement}'"
                )
            candidate = replacement

        out.append(candidate)
        if is_locomotion_action(candidate):
            loco_streak += 1
        else:
            loco_streak = 0
    return out


def dedupe_adjacent_beats(beats: list[dict]) -> list[dict]:
    actions = [str(b.get("action", "")).strip() for b in beats]
    fixed = dedupe_adjacent_actions(actions)
    for beat, action in zip(beats, fixed):
        beat["action"] = action
    return beats


def _should_soft_retry(attempt: int) -> bool:
    """出片档：仅前 2 次对 soft 失败重试，之后强制接受。"""
    if not SHIPPABLE_MODE:
        return attempt < MAX_GENERATION_ATTEMPTS
    return attempt <= 2 and attempt < MAX_GENERATION_ATTEMPTS


def validate_motion_clarity(raw_shots: list[str]) -> tuple[bool, str]:
    for index, shot in enumerate(raw_shots, start=1):
        if re.search(r"\b(spin|pivot|360|twirl|pirouette)\b", shot, flags=re.I):
            return False, f"shot {index} has forbidden spin/pivot wording: '{shot}'"
        if looks_truncated(shot):
            return False, f"shot {index} looks truncated: '{shot}'"
    return True, "ok"


def validate_motion_sanity(
    raw_shots: list[str],
    grip_hand: str | None = None,
    prop_name: str | None = None,
) -> tuple[bool, str]:
    for index, shot in enumerate(raw_shots, start=1):
        if _STATIC_ACTION_RE.search(shot):
            return False, f"shot {index} is static: '{shot}'"
        if prop_name and (_BOTH_HANDS_DOWN_RE.search(shot) or _BOTH_HANDS_DOWN_ZH_RE.search(shot)):
            return False, f"shot {index} lowers both hands while holding {prop_name}"
        if grip_hand and prop_name:
            prop_arm = bool(
                re.search(
                    rf"\b(raises?|lifts?|lowers?|pulls?)\s+(?:the\s+)?(?:{re.escape(prop_name)}|rod tip)\b",
                    shot,
                    flags=re.I,
                )
            )
            if (
                not prop_arm
                and action_uses_hand(shot, grip_hand)
                and re.search(r"\b(raises?|lifts?|points?|collar|brow)\b", shot, flags=re.I)
            ):
                return False, (
                    f"shot {index} uses occupied {grip_hand} hand for a gesture: '{shot}'"
                )
    return True, "ok"


def validate_i2v_feasibility(raw_shots: list[str]) -> tuple[bool, str]:
    # 结构层：只拦明显残句；Wan 易误解动词由 LLM 规则规避
    for index, shot in enumerate(raw_shots, start=1):
        if looks_truncated(shot):
            return False, f"shot {index} truncated: '{shot}'"
        if re.search(r"\b(raises?|lifts?)\s+it\b", shot, flags=re.I):
            return False, f"shot {index} has broken pronoun 'it': '{shot}'"
    return True, "ok"


def validate_composed_prompts(prompts: list[str]) -> tuple[bool, str]:
    for index, prompt in enumerate(prompts, start=1):
        if looks_truncated(prompt):
            return False, f"composed prompt {index} truncated: '{prompt}'"
        if _STATIC_ACTION_RE.search(prompt):
            return False, f"composed prompt {index} static: '{prompt}'"
    return True, "ok"


def validate_compound_action(raw_shots: list[str]) -> tuple[bool, str]:
    """拦一镜多动作（and / 逗号串动作 / 多运动动词），主题无关。"""
    for index, shot in enumerate(raw_shots, start=1):
        text = (shot or "").strip()
        if not text:
            continue
        parts = [p.strip() for p in re.split(r"\s+and\s+|,\s*", text, flags=re.I) if p.strip()]
        motion_parts = [p for p in parts if _MOTION_VERB_RE.search(p)]
        if len(motion_parts) >= 2:
            return False, f"shot {index} has multiple body motions: '{text}'"
        verbs = [v.lower() for v in _MOTION_VERB_RE.findall(text)]
        stems = {re.sub(r"s$", "", v) for v in verbs}
        # 两个以上不同运动动词，且句子在拼多件事
        if len(stems) >= 2 and re.search(
            r"\b(and|then|before|after|while|as he|as she)\b|,",
            text,
            flags=re.I,
        ):
            return False, f"shot {index} packs multiple motion verbs: '{text}'"
    return True, "ok"


def validate_plot_coverage(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """
    主题无关：后半段不能全是微动作注水；回头高潮须覆盖且不得连镜注水；
    高潮物件须至少一镜入画；禁止用捡起追踪物替代回头。
    """
    if len(beats) < 4:
        # 短片仍检查回头高潮
        pass
    else:
        later = beats[len(beats) // 2 :]
        micro = 0
        for beat in later:
            action = str(beat.get("action", "")).strip().lower()
            if not action:
                continue
            only_head_or_walk = bool(
                re.search(r"\b(turns?\s+head|walks?\s+forward|steps?\s+forward)\b", action)
                or re.search(r"(?:转头|回头|回望|向前走|走进)", action)
            ) and not bool(
                re.search(
                    r"\b(aim|aims|pull|pulls|point|points|raise|raises|kneel|extend|"
                    r"grab|catch|draw|strike|open|close|throw|reveal)\b|"
                    r"入画|注视|锁定|举起|伸出|扫描",
                    action,
                )
            )
            if only_head_or_walk:
                micro += 1
        if len(later) >= 2 and micro >= len(later):
            return False, (
                "later beats are only micro head-turns/walks — "
                "progress the story instead of padding"
            )

    ok, reason = validate_lookback_climax(user_topic, beats)
    if not ok:
        return False, reason
    ok, reason = validate_lookback_budget(beats)
    if not ok:
        return False, reason
    ok, reason = validate_opening_locomotion(user_topic, beats)
    if not ok:
        return False, reason
    ok, reason = validate_climax_object_on_screen(user_topic, beats)
    if not ok:
        return False, reason
    ok, reason = validate_climax_reveal_staging(user_topic, beats)
    if not ok:
        return False, reason
    ok, reason = validate_climax_enter_once(user_topic, beats)
    if not ok:
        return False, reason
    return True, "ok"


def validate_lookback_climax(user_topic: str, beats: list[dict]) -> tuple[bool, str]:
    """主题写了察觉/回头，则分镜必须有回头/过肩注视，不能用掏装置/捡起物替代高潮。"""
    if not topic_requires_lookback_climax(user_topic):
        return True, "ok"
    actions = " ".join(str(b.get("action", "")) for b in beats)
    if _INVENTED_SEIZE_OBJECT_RE.search(actions) and not topic_authorizes_seize_object(user_topic):
        return False, (
            "topic requires look-back climax; do not invent picking up / holding the tracker"
        )
    if _LOOKBACK_BEAT_RE.search(actions) or _LOOKBACK_ACTION_RE.search(actions):
        return True, "ok"
    return False, (
        "topic requires a look-back / over-shoulder notice climax beat "
        "(not invented gadgets or picking up the tracker)"
    )


# 兼容旧 import（validate_prompts 曾引用）
_EXTREME_HEAD_UP_RE = re.compile(r"\bhead\s+tilted\s+(?:fully\s+)?(?:upward|back)\b", flags=re.I)
_OVERHEAD_PROP_RE = re.compile(
    r"\b(?:umbrella|canopy)\b[^.]{0,30}\b(?:overhead|over\s+head)\b",
    flags=re.I,
)


# ---------------------------------------------------------------------------
# LLM 系统规则
# ---------------------------------------------------------------------------

def build_spine_system_prompt(
    min_beats: int | None = None,
    max_beats: int | None = None,
    preferred_beats: int | None = None,
) -> str:
    clip_seconds = int(round(CLIP_SECONDS))
    lo = int(min_beats if min_beats is not None else MIN_BEATS)
    hi = int(max_beats if max_beats is not None else MAX_BEATS)
    pref = int(preferred_beats if preferred_beats is not None else PREFERRED_BEATS)
    return f"""
你是 Wan2.2 图生视频（I2V）分镜策划。目标是「故事可看懂」，不是画质炫技。
优先顺序：①贴合 USER_INPUT 主题事件 ②画面不崩 ③镜间不重复 ④运镜服务剧情。
清晰度、细节、霓虹锐度一律次要。
容量：最多约 {int(MAX_TARGET_SECONDS)} 秒（硬顶 {ABS_MAX_BEATS} 镜 × 约 {clip_seconds} 秒）。
按故事密度选镜数：事件少就少写；禁止用注水走路凑时长。

只输出 JSON：
{{
  "visual_anchor": "中文身份+设定短句，约12-22字词量",
  "beats": [
    {{
      "beat": 1,
      "start_state": "无或简短起始姿态（中文）",
      "action": "一个可见的主体连续动作（中文）",
      "end_state": "动作结束后的简短姿态（中文）"
    }}
  ]
}}

最低规则：
1. 镜数 {lo}-{hi}（偏好约 {pref}）。每镜约 {clip_seconds} 秒。
   每镜只推进一件剧情事；事件齐了就结束，禁止填充镜。
2. 一镜一个连续动作，禁止用「然后/并且」塞两个动作。
3. 主角身份与 USER_INPUT 一致，除非主题改写身份。
4. visual_anchor 必须保留 USER_INPUT 中的关键设定名词。
5. 允许相似动作；禁止全文完全相同的动作。
   同一动作指纹不得在间隔 < {MIN_REPEAT_GAP} 的相邻镜重复。
   连续纯行进最多 {MAX_CONSECUTIVE_WALKING_BEATS} 镜（偏好 {MAX_CONSECUTIVE_LOCOMOTION}）。
6. 禁止完全静止。禁止原地旋转/360。禁止后退/倒走（I2V 易崩）。
7. 若 USER_INPUT 开篇是走进/走入/进入场景：第 1 镜必须写该行进（向纵深），
   禁止把回头/察觉/抬头放到第 1 镜；回头高潮放在走进之后。
   若第 1 镜是行进：起势双支撑沾地、躯干直立，向纵深行进（偏背影/过肩），
   禁止横穿画面侧身大步定格；动作句直接写动词，少用方式副词；
   除非 USER_INPUT 明确要求，否则勿写跌倒/跪趴。
8. 反应优先转头/上半身，禁止全身急转。
   禁止连续两镜都是抬头族。需要眼神/下巴/脸的节拍必须侧脸或过肩可读。
9. 手持道具须是实体物；握持在身前/身侧；禁止背后拧臂；禁止穿模。
   禁止发明 USER_INPUT 没有的枪/装置/光束。
   禁止发明捡起/手持外部追踪物（除非主题写明捡起）。
10. 覆盖主题关键事件（故事优先硬规则）：
    主题写了的关键情节必须有对应镜；禁止用重复走路顶替高潮。
    若主题有察觉+回头：全片最多 1 镜过肩侧脸微动（硬切定妆，禁止大转头过程），
    其后必须推进情节；察觉桥接用背影+镜头微仰，不要从背影拧出正脸。
    不能用掏装置/捡起外部物件顶替高潮。
    若 USER_INPUT 出现后段高潮物件：至少 1 镜让该物件入画可读，
    禁止全程只用「画外」暗示。
    出场须铺垫：先有察觉（背影微仰）/过肩微动桥接镜，再写物件进入；
    物件进入镜写「镜头推向主体背部 + 物件从上方进入」，
    禁止正脸猛拧头后再微仰（易脸崩且几乎不做运镜）。
    进入画面全片最多 1 镜；禁止第一镜出现物件，禁止首句已是画面中心静止被凝视。
11. 禁止对镜头注视。
12. 连续两镜若都是近似背影纵深：合并为「镜头推向主体背部」，或第二镜改明显不同动作。
    运镜只写结构句（推向背部/纵深走远/主体变小），禁止空形容词。
13. visual_anchor 只写人物与场景设定，不要写入后段才出现的高潮物件名词。
14. 回头/过肩镜：硬切到过肩侧脸定妆首帧，I2V 只写「侧脸微动/目光」；
   禁止迅速转身、猛然回头、大幅度拧脸。
15. 全片最多 1 镜过肩见脸微动；其后必须推进（物件进入或非回头动作），禁止连镜回头。
    以上为全局规则，适用于任意主题。

start_state/end_state 仅作笔记；成片不把上镜末帧像素接成下镜开场（流水线另有交接策略）。

visual_anchor / action / end_state 一律写中文，不要英文化翻译。
""".strip()


def build_expand_system_prompt() -> str:
    return f"""
把已锁定的连贯脊柱扩成 Wan2.2 I2V 提示。故事优先：贴主题、不崩、不重复、运镜服务剧情。

只输出 JSON：
{{
  "shots": [
    {{
      "beat": 1,
      "shot_prompt": "一个主体动作，中文约8-18字",
      "keyframe_prompt": "静帧描述，中文约25-50字"
    }}
  ]
}}

最低规则：
- 一脊柱节拍对应一镜；shot_prompt 只复述该节拍动作。
- 允许相似；禁止完全相同文案；相同指纹间隔至少 {MIN_REPEAT_GAP}。
- 禁止旋转/360；禁止「举起它」残句；禁止完全静止。
- 禁止全身急转，改用转头/上半身。
- 禁止连续两镜抬头族。
- 禁止后退倒走。
- 禁止发明主题/脊柱没有的枪械装置光束。
- 禁止发明捡起/手持外部追踪物（除非主题写明）。
- 若主题有察觉+回头：全片最多一镜过肩侧脸微动（硬切定妆，禁止大转头），其后推进情节，禁止连镜回头注水。
- 察觉桥接优先「背影 + 镜头微仰」；见脸只用定妆微动，禁止迅速转身/猛然回头。
- 若主题有高潮物件：至少一镜该物件入画可读；须先桥接再「进入画面」，禁止硬切突然占位。
- 物件进入：shot_prompt 写背后推镜入画；keyframe 保持背影/过肩纵深并预留上方空间。
- 禁止对镜头注视。
- keyframe 保持与脊柱相同环境风格；构图可微变，禁止跳到另一世界风格。
- 行进开场：双支撑沾地、纵深行进；禁止横穿侧身 mid-stride 定格。
- 需要脸/下巴/抬头：keyframe 必须侧脸或过肩可读，禁止全背影。
- 连续背影纵深：合并为推镜，勿复制多段同构图走路。
- 清晰度次要；宁可略糊也不要崩脸、重复镜、丢主题事件。
- 主体可读受光，禁止纯黑剪影。
- 景别/物种跟脊柱，不要强行人类全身。
- 不要写「无」当 keyframe_prompt。
- shot_prompt / keyframe_prompt 一律中文，不要英文化。
""".strip()


# ---------------------------------------------------------------------------
# 生成编排
# ---------------------------------------------------------------------------

def build_prompts_from_beats(visual_anchor: str, beats: list[dict]) -> tuple[list[str], list[str], list[str]]:
    prop_name, grip_hand = infer_prop_grip(
        visual_anchor,
        *[f"{b.get('start_state', '')} {b.get('end_state', '')}" for b in beats],
    )
    raw_shots, keyframe_prompts, prompts_list = [], [], []
    for index, beat in enumerate(beats):
        motion = sanitize_shot_motion(
            str(beat.get("action", "")).strip(),
            grip_hand=grip_hand,
            prop_name=prop_name,
            previous=raw_shots,
            index=index,
            context_text=visual_anchor,
        )
        if not motion:
            motion = pick_fallback_motion(raw_shots, grip_hand=grip_hand, index=index)
        raw_shots.append(motion)
        keyframe_prompts.append(
            compose_keyframe_prompt(
                visual_anchor,
                str(beat.get("start_state", "")).strip() or motion,
            )
        )
        prompts_list.append(compose_i2v_prompt(visual_anchor, motion, index))
    raw_shots = dedupe_adjacent_actions(raw_shots)
    prompts_list = [
        compose_i2v_prompt(visual_anchor, raw_shots[i], i) for i in range(len(raw_shots))
    ]
    return raw_shots, keyframe_prompts, prompts_list


def pick_motion_for_beat(beat: dict, shot_prompt: str, previous: list[str]) -> str:
    candidates = [
        str(beat.get("action", "")).strip(),
        (shot_prompt or "").strip(),
    ]
    for cand in candidates:
        if not cand:
            continue
        ok, _ = validate_shot_diversity(previous + [cand])
        if ok or not previous:
            return cand
    return pick_fallback_motion(previous, index=len(previous))


def generate_continuity_spine(
    user_topic: str,
    budget: dict | None = None,
) -> dict:
    clip_seconds = int(round(CLIP_SECONDS))
    budget = budget or resolve_beat_budget(user_topic)
    min_beats = int(budget["min_beats"])
    max_beats = int(budget["max_beats"])
    preferred_beats = int(budget["preferred_beats"])
    target_seconds = float(budget["target_seconds"])
    story_topic = str(budget.get("story_topic") or user_topic).strip() or user_topic

    if SHIPPABLE_MODE:
        print("📦 SHIPPABLE_MODE=1：启发式校验降级为警告，优先保证出片")
    print(
        f"🎚️ 镜数预算: {min_beats}-{max_beats}（偏好 ~{preferred_beats}，"
        f"≈{int(target_seconds)}s，来源={budget.get('source')}）"
    )
    user_message = (
        f"请规划 {min_beats}-{max_beats} 个连贯动作节拍"
        f"（偏好约 {preferred_beats} 镜，总时长约 {int(target_seconds)} 秒，"
        f"每镜约 {clip_seconds} 秒）。镜数匹配故事密度——"
        f"禁止用注水走路/转身硬凑上限。全部字段用中文。\n\n"
        f"USER_INPUT:\n{story_topic}"
    )

    last_good: dict | None = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        result_text = call_llm(
            build_spine_system_prompt(min_beats, max_beats, preferred_beats),
            user_message,
            temperature=0.55 if attempt == 1 else 0.7,
        )
        try:
            parsed = json.loads(extract_json_block(result_text))
            visual_anchor = enrich_visual_anchor(
                str(parsed.get("visual_anchor", "")).strip(),
                story_topic,
            )
            beats = parsed.get("beats") or []
            if len(beats) < min_beats:
                print(f"⚠️ 故事骨架第 {attempt} 次仅 {len(beats)} 拍（需要≥{min_beats}），重试...")
                continue
            if len(beats) > max_beats:
                beats = beats[:max_beats]

            prop_name, grip_hand = infer_prop_grip(
                visual_anchor,
                *[f"{b.get('start_state', '')} {b.get('end_state', '')}" for b in beats],
            )
            if prop_name and grip_hand:
                print(f"🖐️ 手持道具占用(提示): {prop_name} → {grip_hand} hand")

            for i, beat in enumerate(beats, start=1):
                beat["beat"] = i
                prior = [str(b.get("action", "")).strip() for b in beats[: i - 1]]
                beat["action"] = sanitize_shot_motion(
                    str(beat.get("action", "")).strip(),
                    grip_hand=grip_hand,
                    prop_name=prop_name,
                    previous=prior,
                    index=i - 1,
                    context_text=f"{story_topic} {visual_anchor}",
                )
                beat["start_state"] = str(beat.get("start_state", "")).strip() or "none"
                beat["end_state"] = str(beat.get("end_state", "")).strip()
                if not beat["action"]:
                    beat["action"] = pick_fallback_motion(
                        prior, grip_hand=grip_hand, index=i - 1
                    )

            hard_ok, hard_reason = validate_spine_structure(beats)
            if not hard_ok:
                print(f"⚠️ 故事骨架结构失败：{hard_reason}，重试...")
                user_message = (
                    f"{user_message}\n\n上次结构失败：{hard_reason}\n"
                    "请修复空动作/残句/静止动作。"
                )
                continue

            soft_checks = [
                (lambda: validate_spine_occupancy(beats, grip_hand, prop_name), "占手"),
                (lambda: validate_spine_walking_streak(beats), "走路重复"),
                (lambda: validate_opening_locomotion(story_topic, beats), "开场行进"),
                (lambda: validate_lookback_budget(beats), "回头预算"),
                (lambda: validate_climax_object_on_screen(story_topic, beats), "高潮物件入画"),
                (lambda: validate_climax_reveal_staging(story_topic, beats), "高潮出场铺垫"),
                (lambda: validate_climax_enter_once(story_topic, beats), "高潮进入次数"),
                (lambda: validate_beat_action_diversity(beats), "动作多样性"),
                (lambda: validate_compound_action([str(b.get("action", "")) for b in beats]), "复合动作"),
                (lambda: validate_plot_coverage(story_topic, beats), "情节覆盖"),
            ]
            soft_fails: list[str] = []
            for checker, label in soft_checks:
                ok, reason = checker()
                if not ok:
                    soft_fails.append(f"{label}: {reason}")

            last_good = {"visual_anchor": visual_anchor, "beats": [dict(b) for b in beats]}

            if soft_fails and _should_soft_retry(attempt):
                joined = "; ".join(soft_fails)
                print(f"⚠️ 故事骨架软校验失败（将重试 {attempt}/2）：{joined}")
                user_message = (
                    f"{user_message}\n\n上次软校验问题：{joined}\n"
                    "相似动作可以，但禁止完全相同的动作文案，"
                    f"同一动作指纹至少间隔 {MIN_REPEAT_GAP} 镜；"
                    f"回头/过肩察觉全片最多 {MAX_LOOKBACK_CLIMAX_BEATS} 镜；"
                    "主题若以走进场景起笔，第1镜必须是行进；"
                    "主题高潮物件须先桥接再进入画面，禁止硬切突然占位。"
                )
                continue

            if soft_fails:
                print(f"⚠️ 校验降级为警告（出片档强制接受）：{'; '.join(soft_fails)}")

            beats = dedupe_adjacent_beats(beats)
            return {"visual_anchor": visual_anchor, "beats": beats}
        except (json.JSONDecodeError, ValueError) as error:
            print(f"故事骨架 JSON 解析失败：{error}")

    if last_good:
        print("⚠️ 出片档：使用最后一份合法骨架（本地去重后出片）")
        beats = dedupe_adjacent_beats(last_good["beats"])
        return {"visual_anchor": last_good["visual_anchor"], "beats": beats}

    return {"visual_anchor": "", "beats": []}


def expand_spine_to_shots(user_topic: str, spine: dict) -> tuple[list[str], list[str], list[str]]:
    visual_anchor = spine.get("visual_anchor", "")
    beats = spine.get("beats", [])
    beat_count = len(beats)
    if beat_count < 1:
        return [], [], []
    if beat_count < ABS_MIN_BEATS and not SHIPPABLE_MODE:
        return [], [], []

    prop_name, grip_hand = infer_prop_grip(
        visual_anchor,
        *[f"{b.get('start_state', '')} {b.get('end_state', '')}" for b in beats],
    )
    spine_json = json.dumps(
        {"visual_anchor": visual_anchor, "beats": beats},
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        f"请展开为恰好 {beat_count} 条 shot_prompt 与 keyframe_prompt（全部中文）。\n"
        f"USER_INPUT:\n{user_topic}\n\nCONTINUITY_SPINE:\n{spine_json}"
    )

    last_good: tuple[list[str], list[str], list[str]] | None = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        result_text = call_llm(
            build_expand_system_prompt(),
            user_message,
            temperature=0.55 if attempt == 1 else 0.7,
        )
        try:
            parsed = json.loads(extract_json_block(result_text))
            shots = (parsed.get("shots") or [])[:beat_count]
            if len(shots) < beat_count:
                print(f"⚠️ 分镜展开第 {attempt} 次仅 {len(shots)} 条，重试...")
                continue

            raw_shots, keyframe_prompts, prompts_list = [], [], []
            for index, shot in enumerate(shots):
                beat = beats[index]
                shot_prompt = sanitize_shot_motion(
                    str(shot.get("shot_prompt", "")).strip(),
                    grip_hand=grip_hand,
                    prop_name=prop_name,
                    previous=raw_shots,
                    index=index,
                    context_text=f"{user_topic} {visual_anchor}",
                )
                beat_action = sanitize_shot_motion(
                    str(beat.get("action", "")).strip(),
                    grip_hand=grip_hand,
                    prop_name=prop_name,
                    previous=raw_shots,
                    index=index,
                    context_text=f"{user_topic} {visual_anchor}",
                )
                motion = pick_motion_for_beat(
                    {**beat, "action": beat_action},
                    shot_prompt,
                    raw_shots,
                )
                motion = sanitize_shot_motion(
                    motion,
                    grip_hand=grip_hand,
                    prop_name=prop_name,
                    previous=raw_shots,
                    index=index,
                    context_text=f"{user_topic} {visual_anchor}",
                )
                raw_shots.append(motion)
                keyframe_prompts.append(
                    compose_keyframe_prompt(
                        visual_anchor,
                        str(shot.get("keyframe_prompt", beat.get("start_state", ""))).strip(),
                    )
                )
                prompts_list.append(compose_i2v_prompt(visual_anchor, motion, index))

            # 硬：残句 / raises it / 静止 / 组装截断
            hard_ok, hard_reason = validate_i2v_feasibility(raw_shots)
            if hard_ok:
                hard_ok, hard_reason = validate_composed_prompts(prompts_list)
            if hard_ok:
                for shot in raw_shots:
                    if _STATIC_ACTION_RE.search(shot) or looks_truncated(shot):
                        hard_ok, hard_reason = False, f"static/truncated shot: {shot}"
                        break
            if not hard_ok:
                print(f"⚠️ 分镜展开结构失败：{hard_reason}，重试...")
                user_message = (
                    f"{user_message}\n\n上次展开结构失败：{hard_reason}\n"
                    "请修复残句/静止动作。"
                )
                continue

            soft_checks = [
                (lambda: validate_compound_action(raw_shots), "复合动作"),
                (lambda: validate_motion_sanity(raw_shots, grip_hand, prop_name), "占手/静止"),
                (lambda: validate_motion_clarity(raw_shots), "旋转歧义"),
                (lambda: validate_shot_diversity(raw_shots), "镜头多样性"),
            ]
            soft_fails: list[str] = []
            for checker, label in soft_checks:
                ok, reason = checker()
                if not ok:
                    soft_fails.append(f"{label}: {reason}")

            last_good = (list(raw_shots), list(keyframe_prompts), list(prompts_list))

            if soft_fails and _should_soft_retry(attempt):
                print(f"⚠️ 分镜展开软校验失败（将重试）：{'; '.join(soft_fails)}")
                user_message = (
                    f"{user_message}\n\n上次软校验问题：{'; '.join(soft_fails)}\n"
                    "相似动作可以，但禁止完全相同的镜头文案，"
                    f"同一动作指纹至少间隔 {MIN_REPEAT_GAP} 镜。"
                )
                continue

            if soft_fails:
                print(f"⚠️ 校验降级为警告（出片档强制接受）：{'; '.join(soft_fails)}")

            raw_shots = dedupe_adjacent_actions(raw_shots)
            prompts_list = [
                compose_i2v_prompt(visual_anchor, raw_shots[i], i) for i in range(len(raw_shots))
            ]
            return raw_shots, keyframe_prompts, prompts_list
        except (json.JSONDecodeError, ValueError) as error:
            print(f"分镜展开 JSON 解析失败：{error}")

    if last_good:
        print("⚠️ 出片档：使用最后一份展开结果（本地去重）")
        raw_shots, keyframe_prompts, _ = last_good
        raw_shots = dedupe_adjacent_actions(raw_shots)
        prompts_list = [
            compose_i2v_prompt(visual_anchor, raw_shots[i], i) for i in range(len(raw_shots))
        ]
        return raw_shots, keyframe_prompts, prompts_list

    print("⚠️ 展开失败，回退到骨架 beat.action 直出")
    return build_prompts_from_beats(visual_anchor, beats)


def generate_video_prompts(
    user_topic: str,
    target_seconds: float | None = None,
):
    budget = resolve_beat_budget(user_topic, target_seconds=target_seconds)
    story_topic = str(budget.get("story_topic") or user_topic).strip() or user_topic

    print("=" * 40)
    print(f"📝 主题：{user_topic}")
    if story_topic != user_topic.strip():
        print(f"📝 剧情文本（已去掉时长指令）：{story_topic}")
    print(f"🤖 使用模型：{QWEN_MODEL}")
    print(
        f"📦 SHIPPABLE_MODE={'1' if SHIPPABLE_MODE else '0'} | "
        f"目标≈{int(budget['target_seconds'])}s / "
        f"~{budget['preferred_beats']}镜 "
        f"（允许 {budget['min_beats']}-{budget['max_beats']}，来源={budget['source']}）"
    )
    print("=" * 40)

    spine = generate_continuity_spine(story_topic, budget=budget)
    beats = spine.get("beats") or []
    visual_anchor = spine.get("visual_anchor", "")
    if not beats:
        return {
            "user_topic": user_topic,
            "story_topic": story_topic,
            "beat_budget": budget,
            "visual_anchor": "",
            "continuity_spine": [],
            "raw_shots": [],
            "keyframe_prompts": [],
            "prompts": [],
        }

    print(f"✅ 故事骨架 {len(beats)} 拍 | 锚点: {visual_anchor}")
    raw_shots, keyframe_prompts, prompts_list = expand_spine_to_shots(story_topic, spine)
    for i, (video_prompt, keyframe_prompt) in enumerate(
        zip(prompts_list, keyframe_prompts), start=1
    ):
        print(f"  镜头 {i:02d} [视频]: {video_prompt}")
        print(f"  镜头 {i:02d} [首帧]: {keyframe_prompt}")

    return {
        "user_topic": user_topic,
        "story_topic": story_topic,
        "beat_budget": budget,
        "visual_anchor": visual_anchor,
        "continuity_spine": beats,
        "raw_shots": raw_shots,
        "keyframe_prompts": keyframe_prompts,
        "prompts": prompts_list,
    }


if __name__ == "__main__":
    topic = os.getenv("TEST_TOPIC", "晴朗的日子里，带着草帽的老头在湖边钓上来了一条金光闪闪的鱼")
    result = generate_video_prompts(topic)
    print(json.dumps(result, ensure_ascii=False, indent=2))

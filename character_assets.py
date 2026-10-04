"""
免费角色素材库：用本地目录缓存角色卡，锁住跨镜身份。

来源（全部免费）：
1. 用户/作业放入的本地图片（character_assets/ 或 CHARACTER_ASSET_PATH）
2. 本机用已有万相 T2I 按 visual_anchor 生成一次并入库（无额外付费素材站）

不依赖付费素材 API。身份锁定后，流水线不再用 Scene1 首帧做人物回锚。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# 默认：数据盘优先，其次项目内目录
_DEFAULT_DIRS = (
    "/root/autodl-tmp/character_assets",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_assets"),
)

CHARACTER_LOCK_MODE = os.getenv("CHARACTER_LOCK_MODE", "asset").strip().lower()
CHARACTER_ASSET_PATH = (os.getenv("CHARACTER_ASSET_PATH") or "").strip()
CHARACTER_ASSET_DIR = (os.getenv("CHARACTER_ASSET_DIR") or "").strip()


def resolve_asset_root() -> Path:
    if CHARACTER_ASSET_DIR:
        root = Path(CHARACTER_ASSET_DIR)
    else:
        root = Path(_DEFAULT_DIRS[0])
        if not root.exists():
            for candidate in _DEFAULT_DIRS:
                p = Path(candidate)
                if p.exists():
                    root = p
                    break
            else:
                root = Path(_DEFAULT_DIRS[-1])
    root.mkdir(parents=True, exist_ok=True)
    (root / "generated").mkdir(parents=True, exist_ok=True)
    (root / "manual").mkdir(parents=True, exist_ok=True)
    return root


def character_asset_id(visual_anchor: str, user_topic: str = "") -> str:
    """由身份描述生成稳定素材 ID（主题无关的短哈希）。"""
    # 高潮物件不参与身份哈希，避免「有无无人机」生成两套脸
    raw = re.sub(r"\s+", " ", f"{visual_anchor or ''}||{user_topic or ''}".strip().lower())
    for token in (
        "无人机", "飞行器", "追踪器", "drone", "uav", "tracker", "金鱼", "fish",
    ):
        raw = re.sub(re.escape(token), " ", raw, flags=re.I)
    raw = re.sub(r"\d+\s*(秒|分钟|s|min)\b", " ", raw, flags=re.I)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"char_{digest}"


def build_character_sheet_prompt(visual_anchor: str, user_topic: str = "") -> str:
    """生成「角色卡」用的静帧提示：脸清、身份稳，并尽量贴合主题外观。"""
    identity = (visual_anchor or "").strip() or "短片主角"
    # 去掉后段高潮物件词，避免定妆卡里画上静止飞行物/追踪物
    for token in (
        "无人机", "飞行器", "追踪器", "drone", "uav", "tracker", "金鱼",
    ):
        identity = re.sub(re.escape(token), " ", identity, flags=re.I)
    identity = re.sub(r"[，,]\s*[，,]+", "，", identity)
    identity = re.sub(r"\s+", " ", identity).strip(" ,.;，。；") or "短片主角"
    topic = user_topic or ""
    setting_bits = []
    for key, phrase in (
        ("风衣", "黑色风衣"),
        ("伞", "手持实体雨伞"),
        ("雨夜", "雨夜环境光"),
        ("霓虹", "霓虹边缘光"),
        ("赛博", "赛博朋克未来都市气质"),
        ("cyberpunk", "赛博朋克未来都市气质"),
        ("赏金猎人", "赏金猎人装束"),
        ("猎人", "猎人装束"),
        ("兜帽", "连帽兜帽"),
        ("面具", "面部半遮面具"),
    ):
        if key.lower() in topic.lower() or key.lower() in identity.lower():
            setting_bits.append(phrase)
    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for p in setting_bits:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    extra = "，".join(uniq)
    extra = f"：{extra}" if extra else ""
    return (
        f"角色定妆半身像，脸部清晰可读，四分之三侧或正脸，"
        f"与主题描述完全一致的外貌、服装与气质{extra}。"
        f"主体：{identity}。"
        f"中性站姿，双手自然，不要大幅度迈步定格，"
        f"不要发明枪械或手持器械（除非描述写明），"
        f"不要出现后段才出场的高潮物件，"
        f"目光不看镜头，写实电影光影，清晰对焦，干净可复用的身份参考图。"
    )


def build_character_lock_instruction(
    *,
    keyframe_prompt: str = "",
    action_hint: str = "",
    visual_anchor: str = "",
    shot_index: int = 0,
) -> str:
    """以素材库角色图为 Image1，生成本镜场景姿态（中文）。"""
    action = (action_hint or "").strip()
    setting = (visual_anchor or "").strip()
    scene = (keyframe_prompt or "").strip()
    parts = [
        "图像1是免费角色素材库中的定妆参考，必须保持同一人物身份："
        "同一张脸、发型、体型、服装与已描述道具。不要换成另一个人。",
        "按本镜要求生成新的电影静帧：可改姿态、景别与机位，"
        "但人物身份必须与图像1一致。",
        "不要发明描述中没有的枪械或器械。目光在场景内或画外，禁止看向镜头。",
    ]
    if action:
        parts.append(f"本镜主体动作/姿态：{action}。")
        if shot_index == 0:
            parts.append(
                "这是开场镜：若动作为行进，双脚沾地起势、向纵深；"
                "禁止横穿画面 mid-stride 定格。"
            )
        elif re.search(r"回头|回望|过肩|越过肩膀|侧脸微动|look\s*back|over\s+shoulder", action, flags=re.I):
            parts.append(
                "过肩/侧脸定妆镜：首帧已经是过肩或清晰侧脸，"
                "头部仅极小幅预备，双眼对称、五官完整，"
                "禁止正脸大拧头，禁止生成转头过程中间态。"
            )
        elif not re.search(r"走|迈|行进|walk|step", action, flags=re.I):
            parts.append("非行进镜：脸或侧脸/过肩可读，避免又一张全背影走开。")
    if setting:
        parts.append(f"设定与外观关键词：{setting}。")
    if scene:
        parts.append(f"场景与构图要求：{scene[:280]}。")
    parts.append(
        "写实电影静帧，主体受光可读，一个连贯姿态，便于图生视频。"
    )
    return "".join(parts)


def _meta_path(asset_id: str) -> Path:
    return resolve_asset_root() / "generated" / f"{asset_id}.json"


def _image_candidates(asset_id: str) -> list[Path]:
    root = resolve_asset_root()
    names = (
        f"{asset_id}.png",
        f"{asset_id}.jpg",
        f"{asset_id}.jpeg",
        f"{asset_id}.webp",
    )
    out: list[Path] = []
    for folder in (root / "manual", root / "generated", root):
        for name in names:
            out.append(folder / name)
    return out


def find_matched_character_asset(
    visual_anchor: str,
    user_topic: str = "",
) -> str | None:
    """只返回与本主题身份 ID 精确匹配的素材（或显式 CHARACTER_ASSET_PATH）。"""
    if CHARACTER_ASSET_PATH and os.path.isfile(CHARACTER_ASSET_PATH):
        return CHARACTER_ASSET_PATH

    asset_id = character_asset_id(visual_anchor, user_topic)
    for path in _image_candidates(asset_id):
        if path.is_file() and path.stat().st_size > 1024:
            return str(path)
    return None


def find_fallback_default_asset() -> str | None:
    """通用兜底脸：仅当主题定妆生成失败时使用。"""
    root = resolve_asset_root()
    for name in (
        "default.png",
        "default.jpg",
        "default.jpeg",
        "default.webp",
        "char_male_01.jpg",
        "char_male_01.png",
    ):
        p = root / "manual" / name
        if p.is_file() and p.stat().st_size > 1024:
            return str(p)
    return None


def find_existing_character_asset(
    visual_anchor: str,
    user_topic: str = "",
) -> str | None:
    """兼容旧调用：先精确匹配，不再直接吞掉 default（避免挡住主题定妆生成）。"""
    return find_matched_character_asset(visual_anchor, user_topic)


def register_generated_asset(
    asset_id: str,
    image_path: str,
    *,
    visual_anchor: str = "",
    user_topic: str = "",
    prompt: str = "",
) -> str:
    """把生成结果登记进素材库（复制到 generated/）。"""
    from shutil import copy2

    root = resolve_asset_root()
    dest = root / "generated" / f"{asset_id}.png"
    src = Path(image_path)
    if src.resolve() != dest.resolve():
        copy2(src, dest)
    meta = {
        "id": asset_id,
        "visual_anchor": visual_anchor,
        "user_topic": user_topic,
        "prompt": prompt,
        "path": str(dest),
        "source": "wanx_t2i_free",
    }
    _meta_path(asset_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(dest)


def character_lock_enabled() -> bool:
    return CHARACTER_LOCK_MODE in {"asset", "library", "character", "1", "true", "yes"}

import os
import json
import re
import gc
import base64
import mimetypes
from pathlib import Path
import requests

# =====================================================================
# 必须加在 import torch 之前
# =====================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 限制权重反序列化时的并行线程，降低加载 checkpoint shards 时的内存尖峰
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from PIL import Image
import numpy as np


def _patch_torch_from_numpy() -> None:
    """
    AutoDL 上常见：NumPy 装卸多次后出现
      TypeError: expected np.ndarray (got numpy.ndarray)
    实质是 Torch 扩展认的 ndarray 类型 与 当前 import 的 numpy.ndarray 不是同一个 class。
    用 tensor(...) 兜底，避免 UniPC scheduler 在 from_numpy 处直接炸。
    """
    if getattr(torch, "_videogen_from_numpy_patched", False):
        return
    _orig = torch.from_numpy

    def _safe_from_numpy(arr):  # type: ignore[no-untyped-def]
        try:
            return _orig(arr)
        except TypeError:
            data = np.asarray(arr)
            # clone 成连续数组，再交给 tensor（不走 from_numpy 的类型闸门）
            return torch.tensor(data.copy())

    torch.from_numpy = _safe_from_numpy  # type: ignore[assignment]
    torch._videogen_from_numpy_patched = True  # type: ignore[attr-defined]


def _assert_numpy_torch_compatible() -> None:
    """启动时冒烟；失败则打印可执行修复命令。"""
    ver = getattr(np, "__version__", "?")
    print(f"ℹ️ NumPy {ver} @ {getattr(np, '__file__', '?')}")
    print(f"ℹ️ Torch {torch.__version__}")
    _patch_torch_from_numpy()
    try:
        t = torch.from_numpy(np.zeros(4, dtype=np.float64))
        assert t.numel() == 4
        print("ℹ️ NumPy↔Torch from_numpy 冒烟通过")
    except Exception as exc:
        raise RuntimeError(
            "NumPy 与 Torch 仍无法互通。请开一个【新终端】执行：\n"
            "  pip uninstall -y numpy\n"
            "  pip install --force-reinstall --no-cache-dir 'numpy==2.1.3'\n"
            "  python -c \"import numpy,torch; print(numpy.__version__, torch.from_numpy(numpy.zeros(4)))\"\n"
            "若你更想钉 1.26：先完全退出所有 Python/Jupyter 内核再装 numpy==1.26.4。\n"
            f"原始错误: {exc}"
        ) from exc


_assert_numpy_torch_compatible()

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel, UMT5EncoderModel

try:
    import dashscope
    from dashscope import ImageSynthesis
except ImportError:
    print("📦 正在自动为你安装阿里云百炼 SDK...")
    os.system("pip install dashscope")
    import dashscope
    from dashscope import ImageSynthesis

try:
    from dashscope import MultiModalConversation
    _HAS_MULTIMODAL = True
except ImportError:
    MultiModalConversation = None  # type: ignore
    _HAS_MULTIMODAL = False

from local_env import load_local_env
from llm_director import (
    action_mentions_climax_object,
    boost_i2v_motion_prompt,
    compose_i2v_prompt,
    detect_handheld_prop,
    enrich_visual_anchor,
    extract_climax_object_phrases,
    extract_setting_phrases,
    is_climax_enter_action,
    is_locomotion_action,
    is_lookback_beat_action,
    normalize_visual_anchor,
    prefers_face_micro_motion,
    prefers_rear_pushin,
    prefers_reveal_pushin,
    prefers_tilt_up_camera,
    requires_face_readable_action,
    sanitize_shot_motion,
    should_include_climax_in_keyframe,
    strip_climax_phrases_from_text,
    strip_invented_handheld_action,
)
from character_assets import (
    build_character_lock_instruction,
    build_character_sheet_prompt,
    character_asset_id,
    character_lock_enabled,
    find_fallback_default_asset,
    find_matched_character_asset,
    register_generated_asset,
)

# 镜间策略：角色素材库锁人 + 交接帧锁背景；成片硬切。失败回退独立 T2I。

# ==========================================
# Wan2.2-I2V-A14B（开源 Diffusers）配置
# - 同一权重支持 480P/720P（靠 max_area 区分）
# - 24GB 显存必须 WAN_OFFLOAD=group/sequential；满血无 offload 约需 80GB
# - 需较新的 diffusers（建议 git 源码安装，见 requirements-wan-runtime.txt）
# ==========================================
_WAN22_DISK = "/root/autodl-tmp/models/Wan2.2-I2V-A14B-Diffusers"
_WAN22_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_PROFILES = {
    "720p": {
        "model_id": _WAN22_ID,
        "disk_path": _WAN22_DISK,
        "max_area": 720 * 1280,
    },
    "480p": {
        "model_id": _WAN22_ID,
        "disk_path": _WAN22_DISK,
        "max_area": 480 * 832,
    },
}
WAN_PROFILE = os.getenv("WAN_I2V_PROFILE", "720p").lower()
# 24GB 默认低显存；有 ≥40GB 显存再设 WAN_LOW_VRAM=0
WAN_LOW_VRAM = os.getenv("WAN_LOW_VRAM", "1") == "1"
# 0.48 ≈ 832~896 宽，避免 944x528 + model offload 占满 23GB
WAN_AREA_SCALE = float(os.getenv("WAN_AREA_SCALE", "0.48" if WAN_LOW_VRAM else "0.85"))
WANX_SIZE = "1280*720"
# 首帧 T2I 模型：默认用新一代 wan2.2-t2i-plus（结构/遵循度优于 wanx2.1-t2i-plus）。
# 可用环境变量覆盖：WANX_T2I_MODEL=wan2.7-image-pro / qwen-image-2.0-pro / wanx2.1-t2i-plus / wanx-v1
WANX_T2I_MODEL = os.getenv("WANX_T2I_MODEL", "wan2.2-t2i-plus")
# 同步直返模型（sync_call）走这里；旧 wanx-v1 走 .call() 异步任务
_SYNC_T2I_MODELS = {"wan2.2-t2i-plus", "wan2.2-t2i-flash", "wan2.7-image-pro", "qwen-image-2.0-pro"}
# T2I 失败时的回退链（按顺序尝试）
_T2I_FALLBACK_CHAIN = ["wan2.2-t2i-flash", "wanx2.1-t2i-plus", "wanx-v1"]
# 身份保真编辑：参考角色素材库（或旧 Scene1）生成新姿态/场景
IDENTITY_EDIT_MODEL = os.getenv("IDENTITY_EDIT_MODEL", "qwen-image-edit-max")
_IDENTITY_EDIT_FALLBACK = ["qwen-image-edit-plus", "qwen-image-edit"]
# 编辑输出尺寸（与 WANX_SIZE 对齐，宽高各在 512–2048）
IDENTITY_EDIT_SIZE = os.getenv("IDENTITY_EDIT_SIZE", WANX_SIZE)
KEYFRAME_NEGATIVE_PROMPT = (
    "解剖错误, 畸形, 残缺, 模糊, 低质量, 水印, 杂乱背景, 抽象色块, "
    "提示词未描述的多余道具, 凭空发明物体, 重复道具, "
    "光剑, 能量刃, 等离子剑, 发光剑刃, 激光剑, 能量武器, "
    "用无特征发光球体替换已描述道具, 过曝光晕吞没主体, "
    "凭空发明枪械, 提示词没有的手枪, "
    "手臂拧到背后握道具, 背后反关节握持, 肩关节错位持物, "
    "道具刺穿头部, 道具焊进颈部, 物体从颅长出, "
    "伞柄穿头, 伞盖焊在领口, 手持物悬空无握持, 手柄缺失, "
    "道具脱手飞走, 头顶被切平, 天灵盖缺失, 过曝吃头, "
    "需要动作时却完全静止如雕像, "
    "月球步, 倒着走, 脚尖朝向与行进方向相反, "
    "正脸迎面走向镜头的行进开场, 横穿画面的侧身大步定格开场, "
    "四分之三侧身单脚离地迈步定格, 滑冰式定格姿势, "
    "劈叉大跨步, 后仰摔倒, 跌倒, 跪倒, 趴下, 身体塌软失衡, "
    "脚融化进地面, "
    "纯黑剪影主体, 欠曝融进背景的黑块人影, "
    "需要看清脸/下巴/眼神时却用全背影, "
    "对镜头注视, 与观众对视, 自拍看镜头姿势, "
    "场景风格与描述不符, 丢掉提示词要求的设定元素"
)

# Wan 要求 num_frames = 4n+1。81 帧 @16fps ≈ 5 秒，与 Wan2.2 Diffusers 示例一致。
# 60 秒成片 = 12 镜 × 5 秒。
NUM_FRAMES = int(os.getenv("WAN_NUM_FRAMES", "81"))
NUM_INFERENCE_STEPS = int(os.getenv("WAN_INFERENCE_STEPS", "30" if WAN_LOW_VRAM else "40"))
# Wan2.2 官方 I2V-A14B：sample_guide_scale=(3.5, 3.5)；Diffusers 里
# guidance_scale=高噪声专家，guidance_scale_2=低噪声专家
GUIDANCE_SCALE = float(os.getenv("WAN_GUIDANCE_SCALE", "3.5"))
GUIDANCE_SCALE_2 = float(os.getenv("WAN_GUIDANCE_SCALE_2", str(GUIDANCE_SCALE)))
# 行进镜抬高 CFG：更服从「走远/变小」；非行进保持官方 3.5 求稳
LOCO_GUIDANCE_SCALE = float(os.getenv("WAN_LOCO_GUIDANCE_SCALE", "4.5"))
LOCO_GUIDANCE_SCALE_2 = float(
    os.getenv("WAN_LOCO_GUIDANCE_SCALE_2", str(LOCO_GUIDANCE_SCALE))
)
# 官方 Wan2.2 I2V-A14B sample_shift=5.0（= Diffusers flow_shift）。
FLOW_SHIFT = float(os.getenv("WAN_FLOW_SHIFT", "5.0"))
# 行进镜默认更高：拉开纵深位移（过抖再降）
LOCO_FLOW_SHIFT = float(os.getenv("WAN_LOCO_FLOW_SHIFT", "8.5"))
SOURCE_FPS = 16
MAX_I2V_PROMPT_WORDS = int(os.getenv("MAX_I2V_PROMPT_WORDS", "64"))
MAX_KEYFRAME_PROMPT_WORDS = int(os.getenv("MAX_KEYFRAME_PROMPT_WORDS", "128"))
MIN_MOTION_FRAME_DELTA = float(os.getenv("MIN_MOTION_FRAME_DELTA", "0.08"))
# 行进验收：故事可读即可，不靠重试凑动态
LOCO_MIN_FRAME_DIFF = float(os.getenv("LOCO_MIN_FRAME_DIFF", "0.012"))
LOCO_MIN_MOTION_DELTA = float(os.getenv("LOCO_MIN_MOTION_DELTA", "0.18"))
MOTION_RETRY = int(os.getenv("WAN_MOTION_RETRY", "0"))
LOCO_NUM_FRAMES = int(os.getenv("WAN_LOCO_NUM_FRAMES", "81"))
# 故事优先：行进镜默认不角色卡重绘（交接帧直出，敢动）；见脸/停步再锁
LOCO_SKIP_ASSET_LOCK = os.getenv("LOCO_SKIP_ASSET_LOCK", "1") == "1"
CONTINUITY_SEARCH_START_RATIO = 0.75
# 连贯策略：
#   hybrid         = 成片硬切 + 下一镜优先用上镜交接帧（锁背景）
#   identity_lock  = 仅参考角色素材库（或旧 Scene1）做身份编辑
#   handoff_only   = 仅用上镜交接帧
CONTINUITY_MODE = os.getenv("CONTINUITY_MODE", "hybrid").strip().lower()
# 人物锁定：asset=免费角色素材库（推荐）；scene1=旧版 Scene1 首帧回锚
CHARACTER_LOCK_MODE = os.getenv("CHARACTER_LOCK_MODE", "asset").strip().lower()
# hybrid 是否再用 qwen-edit 改姿态：默认 0=交接帧直出（强烈推荐）。
HYBRID_POSE_EDIT = os.getenv("HYBRID_POSE_EDIT", "0") == "1"
# 交接帧取在片段后部的比例（1.0=绝对末帧；0.88 更干净，避开 Wan 末几帧结构塌缩）
HANDOFF_FRAME_RATIO = float(os.getenv("HANDOFF_FRAME_RATIO", "0.88"))
# hybrid 下像素混合默认关闭（叠 Scene1 易叠影）
HANDOFF_IDENTITY_BLEND = float(os.getenv("HANDOFF_IDENTITY_BLEND", "0"))
HANDOFF_BLEND_MAX_DELTA = float(os.getenv("HANDOFF_BLEND_MAX_DELTA", "0.18"))
IDENTITY_BLEND_ALPHA = float(os.getenv("IDENTITY_BLEND_ALPHA", "0"))
# Scene1 周期性回锚：素材库模式下默认关闭（0）；旧模式可用 2
_DEFAULT_REANCHOR = "0" if character_lock_enabled() else "2"
IDENTITY_REANCHOR_EVERY = int(os.getenv("IDENTITY_REANCHOR_EVERY", _DEFAULT_REANCHOR))
IDENTITY_REANCHOR_ALPHA = float(os.getenv("IDENTITY_REANCHOR_ALPHA", "0.28"))
# 行进→非行进 / 脸部动作：用角色素材库重绘姿态（不再贴 Scene1 构图）
LOCO_TO_STILL_REANCHOR = os.getenv("LOCO_TO_STILL_REANCHOR", "1") == "1"
# 默认按镜微调 seed：完全同一 seed + 弱动作会导致相邻镜几乎静止/雷同
BASE_SEED = int(os.getenv("WAN_BASE_SEED", "42"))
WAN_FIXED_SEED = os.getenv("WAN_FIXED_SEED", "0") == "1"
ENABLE_VAE_TILING = os.getenv("VAE_TILING", "1") == "1"
# group=分组预取（24GB 首选，快）；sequential=逐层搬（保底，最慢）；
# model=整模块上 GPU（需约 ≥40GB）；none/gpu/full=全上 GPU（需 ≥40GB）
_WAN_OFFLOAD_RAW = os.getenv("WAN_OFFLOAD", "group" if WAN_LOW_VRAM else "model").lower()
WAN_OFFLOAD_FOLDER = os.getenv("WAN_OFFLOAD_FOLDER", "/root/autodl-tmp/model_offload")
WAN_DROP_PAGE_CACHE = os.getenv("WAN_DROP_PAGE_CACHE", "0") == "1"
# FP8 layerwise casting：权重以 float8 存储、bfloat16 计算，CPU↔GPU 传输量减半，
# group offload 下速度提升明显，画质影响可忽略。仅在 group 模式下启用。
WAN_FP8 = os.getenv("WAN_FP8", "1") == "1"


def resolve_wan_offload() -> str:
    """24GB 上强制 group/sequential，防止误开 model offload 占满显存后 OOM。"""
    mode = _WAN_OFFLOAD_RAW
    if mode in ("none", "gpu", "full", "sequential", "group"):
        requested = mode
    else:
        requested = "model"

    if not torch.cuda.is_available():
        return "sequential"

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if requested == "model" and total_gb < 36:
        print(
            f"⚠️ 检测到 GPU 显存约 {total_gb:.1f}GB < 36GB，"
            "强制 WAN_OFFLOAD=group（model offload 会占满 ~23GB 后推理 OOM）"
        )
        return "group"
    if requested in ("none", "gpu", "full") and total_gb < 40:
        print(
            f"⚠️ 检测到 GPU 显存约 {total_gb:.1f}GB < 40GB，"
            "强制 WAN_OFFLOAD=group（full GPU 无法装下 Wan2.2-A14B）"
        )
        return "group"
    return requested


WAN_OFFLOAD = resolve_wan_offload()
ENABLE_ATTENTION_SLICING = os.getenv(
    "WAN_ATTENTION_SLICING",
    "1" if WAN_OFFLOAD == "sequential" else "0",
) == "1"

# Wan 官方中文负向 + 通用运动/身份稳定性约束（Wan2.2 示例基底）
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，"
    "需要动作时主体却冻结如雕像，只有背景在动，环境运动强过主体，"
    "原地踏步，脚黏地挪不动，几乎静止的行走，粘地挪步，"
    "慢动作行走，极小步幅，蹭地挪移，五秒只挪半步，"
    "只动腿不前进，原地迈腿假走，主体尺度几乎不变，"
    "招牌文字乱变，广告牌融化变形，场景自行改写，"
    "正脸迎面走向镜头，镜中途从侧身拧成背影，朝向突变，"
    "滑行，漂移，滑冰式平移，整姿刚性平移，悬空走，"
    "画面撕裂，闪烁，时间不一致，重影，双曝光，多肢，轮廓涂抹，"
    "劈叉，大跨步，后仰摔倒，跌倒，跪倒，趴下，身体突然塌软，"
    "中途失去平衡倒下，跪地爬行，四肢着地，"
    "髋关节脱节，脚融化进地面，肢体融化，"
    "原地旋转，转圈，360度转身，身份替换，换人，"
    "光剑，能量刃，凭空发明枪械，道具脱手飞走，道具穿模进身体，"
    "对镜头注视，与观众对视，主体突然消失，传送，中途溶解，"
    "纯黑剪影融化，欠曝融进背景，脸部动作却用全背影，"
    "背景扭曲变形，环境传送，不稳定漂移环境"
)
QUALITY_SUFFIX = (
    "同一主体同一外观，"
    "以主体运动为主，主体完整可见，"
    "不对镜头注视，高质量"
)


def resolve_wan_profile() -> dict:
    profile = WAN_PROFILES.get(WAN_PROFILE, WAN_PROFILES["720p"])
    if os.path.exists(profile["disk_path"]):
        print(f"🔀 [智能路径映射] 使用数据盘模型: {profile['disk_path']}")
        return {**profile, "source": profile["disk_path"], "local": True}
    print(f"⚠️ 数据盘未找到模型，将从 HuggingFace 拉取: {profile['model_id']}")
    print(f"   建议预先下载: huggingface-cli download {profile['model_id']} --local-dir {profile['disk_path']}")
    return {**profile, "source": profile["model_id"], "local": False}


def tensor_to_pil(frame_tensor) -> Image.Image:
    if isinstance(frame_tensor, Image.Image):
        return frame_tensor
    if isinstance(frame_tensor, np.ndarray):
        arr = frame_tensor
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr)
    return Image.fromarray((frame_tensor.cpu().numpy() * 255).astype(np.uint8))


def normalize_frame_list(frames) -> list:
    """Wan 可能返回 list 或 numpy 数组 (T,H,W,C)，统一为帧列表。"""
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [frames[i] for i in range(frames.shape[0])]
        if frames.ndim == 3:
            return [frames]
    if isinstance(frames, (list, tuple)):
        return list(frames)
    return [frames]


def frame_pixel_delta(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.asarray(img_a.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR), dtype=np.float32)
    b = np.asarray(img_b.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def clip_mean_consecutive_delta(frames) -> float:
    """
    相邻帧平均像素差（/255）。
    Wan 可读行进常见 0.012–0.02；粘地慢动作常 <0.01。
    """
    frame_list = normalize_frame_list(frames)
    if len(frame_list) < 2:
        return 0.0
    total = 0.0
    prev = tensor_to_pil(frame_list[0])
    for item in frame_list[1:]:
        cur = tensor_to_pil(item)
        total += frame_pixel_delta(prev, cur)
        prev = cur
    return total / (len(frame_list) - 1)


def select_continuity_frame(frames) -> tuple[Image.Image, int, float]:
    """选取交接帧：默认取片段约 88% 处（避开 Wan 末几帧结构塌缩），并回报相对首帧的运动幅度。"""
    frame_list = normalize_frame_list(frames)
    if len(frame_list) == 0:
        raise ValueError("select_continuity_frame requires at least one frame")

    first_frame = tensor_to_pil(frame_list[0])
    last_index = len(frame_list) - 1
    ratio = max(0.5, min(1.0, float(HANDOFF_FRAME_RATIO)))
    target_index = int(round(last_index * ratio))
    target_index = max(1, min(last_index, target_index))

    # 在 target 附近小窗内选与首帧差适中的帧（既有运动又不取最糊的末帧）
    window = max(1, int(len(frame_list) * 0.06))
    lo = max(1, target_index - window)
    hi = min(last_index, target_index + window)
    best_index = target_index
    best_score = None
    for index in range(lo, hi + 1):
        delta = frame_pixel_delta(first_frame, tensor_to_pil(frame_list[index]))
        # 偏好接近 MIN_MOTION 以上、又不过分大的帧（过大常意味着崩坏）
        score = abs(delta - max(MIN_MOTION_FRAME_DELTA, 0.12))
        if best_score is None or score < best_score:
            best_score = score
            best_index = index

    best_delta = frame_pixel_delta(first_frame, tensor_to_pil(frame_list[best_index]))
    return tensor_to_pil(frame_list[best_index]), best_index, best_delta


def blend_identity_reference(
    handoff: Image.Image,
    reference: Image.Image,
    alpha: float,
) -> Image.Image:
    """末帧 × Scene1 身份参考轻混：alpha 越大越靠 Scene1 脸/衣着，越小越靠上镜背景。"""
    if alpha <= 0.01 or reference is None:
        return handoff.convert("RGB")
    alpha = max(0.0, min(1.0, float(alpha)))
    base = handoff.convert("RGB")
    ref = reference.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    return Image.blend(base, ref, alpha)


def build_hybrid_edit_instruction(
    action_hint: str = "",
    visual_anchor: str = "",
    shot_index: int = 1,
    *,
    used_identity_blend: bool = False,
) -> str:
    """硬切叙事 + 交接帧环境连续：编辑上镜交接帧，只改姿态，锁背景与身份。"""
    action = (action_hint or "").strip()
    setting = (visual_anchor or "").strip()
    parts = [
        "图像1是同一短片上一镜的交接帧。",
        "环境锁定：保持图像1的地点风格、建筑、天气、地面与光色，不要传送到另一条街。",
        "身份锁定：保持图像1中同一主体的脸、发型、体型与服装。",
        "姿态只做小幅调整以匹配本镜动作；尽量保持相近机位，不要把背影硬拧成正脸。"
        "静帧要干净：无重影、无双身子、无融化肢体。",
        "不要发明动作里没有的枪械或器械。目光在场景内或画外，禁止看向镜头。",
    ]
    if used_identity_blend:
        parts.append("图像1可能与身份参考轻混；若有叠影，优先保留上一镜背景与单一清晰主体。")
    if action:
        parts.append(f"本镜要求的主体动作/姿态：{action}。")
        if is_locomotion_action(action):
            parts.append("继续行进：保持与图像1相近的行进方向与机位侧，向纵深前行。")
        elif shot_index > 0:
            parts.append("非行进动作：脸或侧脸可读，但保持同一地点，避免大幅跳机位。")
    if setting:
        parts.append(f"设定关键词需兼容：{setting}。")
    parts.append("写实电影静帧，单一清晰轮廓，光影匹配图像1，便于图生视频。")
    return "".join(parts)


def prepare_next_keyframe_base(
    *,
    mode: str,
    handoff_frame: Image.Image | None,
    identity_reference: Image.Image | None,
    shot_index: int,
) -> tuple[Image.Image | None, str, float]:
    """
    返回 (编辑底图, 来源标签, 实际 identity blend)。
    hybrid 默认：纯上镜末帧（不像素混 Scene1，避免叠影崩镜）；
    仅当显式 HANDOFF_IDENTITY_BLEND>0 且两帧足够相似时才轻混。
    """
    mode = (mode or "hybrid").lower()
    requested_blend = max(0.0, min(0.85, float(HANDOFF_IDENTITY_BLEND)))

    # 周期性 Scene1 像素回锚：素材库模式下关闭（IDENTITY_REANCHOR_EVERY 默认 0）
    if (
        not character_lock_enabled()
        and IDENTITY_REANCHOR_EVERY > 0
        and shot_index > 0
        and shot_index % IDENTITY_REANCHOR_EVERY == 0
        and identity_reference is not None
        and handoff_frame is not None
    ):
        delta = frame_pixel_delta(handoff_frame, identity_reference)
        if delta <= HANDOFF_BLEND_MAX_DELTA:
            alpha = max(requested_blend, IDENTITY_REANCHOR_ALPHA)
            return (
                blend_identity_reference(handoff_frame, identity_reference, alpha),
                "hybrid_reanchor",
                alpha,
            )
        print(
            f"   ⚠️ 周期性回锚跳过像素混合：末帧与 Scene1 差太大 "
            f"(delta={delta:.3f}>{HANDOFF_BLEND_MAX_DELTA})"
        )

    if mode == "identity_lock" or handoff_frame is None:
        if identity_reference is None:
            return None, "none", 0.0
        return identity_reference.copy(), "identity_lock", 1.0

    if mode == "handoff_only" or identity_reference is None:
        return handoff_frame.convert("RGB"), "handoff_only", 0.0

    # hybrid：默认不像素混
    if requested_blend <= 0.01:
        return handoff_frame.convert("RGB"), "hybrid_handoff", 0.0

    delta = frame_pixel_delta(handoff_frame, identity_reference)
    if delta > HANDOFF_BLEND_MAX_DELTA:
        print(
            f"   ⚠️ 跳过像素混合防崩镜：末帧与 Scene1 姿态/构图差太大 "
            f"(delta={delta:.3f}>{HANDOFF_BLEND_MAX_DELTA})，改用纯末帧"
        )
        return handoff_frame.convert("RGB"), "hybrid_handoff_safe", 0.0

    return (
        blend_identity_reference(handoff_frame, identity_reference, requested_blend),
        "hybrid_blend",
        requested_blend,
    )


def resolve_scene_seed(visual_anchor: str, scene_index: int) -> int:
    """各镜 seed 按镜递增。每镜独立首帧下 seed 差异进一步拉开镜间视觉差异，
    故用大质数倍数；可用 WAN_FIXED_SEED=1 强制同 seed（仅用于复现调试）。"""
    if BASE_SEED > 0:
        if WAN_FIXED_SEED:
            return BASE_SEED % 1_000_000_007
        return (BASE_SEED + scene_index * 9973) % 1_000_000_007
    base = abs(hash((visual_anchor or "wan").strip().lower())) % 1_000_000
    if WAN_FIXED_SEED:
        return base
    return (base + scene_index * 9973) % 1_000_000


def fit_image_to_video_size(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    target_ratio = target_w / target_h
    src_w, src_h = img.size
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def apply_wan_flow_shift(pipe: WanImageToVideoPipeline, flow_shift: float) -> float:
    """
    设置 UniPC flow_shift（= 官方 sample_shift）。
    这是 Wan 时间动力学的主旋钮：过低→粘地弱动；过高→易抖/崩。
    官方 Wan2.2 I2V-A14B 默认 5.0。
    """
    target = float(flow_shift)
    current = float(getattr(pipe.scheduler.config, "flow_shift", 0.0) or 0.0)
    if abs(current - target) < 1e-6:
        return current
    try:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=target,
        )
    except TypeError:
        # 极旧 diffusers：尽力写 config
        pipe.scheduler.register_to_config(flow_shift=target)
    new_val = float(getattr(pipe.scheduler.config, "flow_shift", target) or target)
    print(f"   🎚️ flow_shift: {current:.2f} → {new_val:.2f}")
    return new_val


def resolve_shot_flow_shift(
    *,
    is_locomotion: bool,
    is_reveal: bool = False,
    rear_pushin: bool = False,
) -> float:
    """行进/入画/背后推镜用更高 shift；其余保持官方默认。"""
    if (is_locomotion or is_reveal or rear_pushin) and LOCO_FLOW_SHIFT > 0:
        return float(LOCO_FLOW_SHIFT)
    return float(FLOW_SHIFT)


def resolve_shot_guidance(
    *,
    is_locomotion: bool,
    is_reveal: bool = False,
    rear_pushin: bool = False,
) -> tuple[float, float]:
    """行进/入画抬高 CFG；其余求稳。"""
    if is_locomotion or is_reveal or rear_pushin:
        return float(LOCO_GUIDANCE_SCALE), float(LOCO_GUIDANCE_SCALE_2)
    return float(GUIDANCE_SCALE), float(GUIDANCE_SCALE_2)


def configure_wan_scheduler(pipe: WanImageToVideoPipeline) -> None:
    """加载后立刻对齐官方采样日程，避免沿用过弱的默认 flow_shift。"""
    applied = apply_wan_flow_shift(pipe, FLOW_SHIFT)
    boundary = getattr(pipe.config, "boundary_ratio", None)
    print(
        f"ℹ️ Wan 采样: flow_shift={applied:.2f} "
        f"(行进镜可用 {LOCO_FLOW_SHIFT:.2f}) | "
        f"guidance={GUIDANCE_SCALE}/{GUIDANCE_SCALE_2} | "
        f"boundary_ratio={boundary}"
    )


def compute_wan_dimensions(image: Image.Image, pipe: WanImageToVideoPipeline, max_area: int) -> tuple[int, int]:
    """按 Wan 官方规则，根据输入图比例计算对齐后的 height/width。"""
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(height), int(width)


def prepare_scene_image(
    image: Image.Image,
    pipe: WanImageToVideoPipeline,
    max_area: int,
) -> tuple[Image.Image, int, int]:
    height, width = compute_wan_dimensions(image, pipe, max_area)
    resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return resized, height, width


def dedupe_prompt_text(prompt: str) -> str:
    prompt = re.sub(r"\bstatic camera\b", " ", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ,.;")

    # 通用重复短语去重：检测任意连续 2~5 词的片段是否出现 ≥2 次，保留首次。
    # 不写死任何主题/道具词表，对所有主题统一生效。
    words = prompt.split()
    seen_spans: dict[str, tuple[int, int]] = {}
    drop: list[tuple[int, int]] = []
    for span_len in (5, 4, 3, 2):
        for i in range(len(words) - span_len + 1):
            span = " ".join(words[i : i + span_len]).lower().strip(".,;")
            if len(span.split()) < 2:
                continue
            if span in seen_spans:
                drop.append((i, i + span_len))
            else:
                seen_spans[span] = (i, i + span_len)
    # 从后往前删，避免下标错位
    for start, end in sorted(drop, key=lambda x: x[0], reverse=True):
        words[start:end] = []
    prompt = " ".join(words)
    # 去重删词后可能留下孤立/重复标点（如 "alley, , ." 或 ",,"），统一清理
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,", ", ", prompt)
    prompt = re.sub(r"\s*\.\s*\.", ". ", prompt)
    prompt = re.sub(r"[,;]\s*([,;.])", r"\1", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ,.;")

    words = prompt.split()
    cleaned = []
    prev = None
    for word in words:
        bare = word.lower().strip(".,;")
        if bare and bare == prev:
            continue
        cleaned.append(word)
        prev = bare
    return " ".join(cleaned).strip(" ,.;")


def _trim_to_budget(words: list[str], budget: int) -> list[str]:
    """按完整短语（逗号边界）裁剪到 budget 词内，避免裁在词中间产生残词。
    通用：不区分主题，只按标点边界保留完整短语。"""
    if len(words) <= budget:
        return words
    kept: list[str] = []
    phrase: list[str] = []
    for w in words:
        phrase.append(w)
        if w.endswith(",") or w.endswith(";"):
            # 短语结束：若加入后不超 budget，则保留，否则丢弃本短语
            if len(kept) + len(phrase) <= budget:
                kept.extend(phrase)
            phrase = []
    # 处理末尾无标点的尾巴短语
    if phrase and len(kept) + len(phrase) <= budget:
        kept.extend(phrase)
    # 若一个完整短语都没保住（短语都比 budget 长），退化为硬截断但补句号
    if not kept:
        kept = words[:budget]
    return kept


def build_i2v_model_prompt(
    prompt: str,
    shot_index: int = 0,
    *,
    prev_was_locomotion: bool = False,
    rear_pushin: bool = False,
    context_text: str = "",
) -> str:
    """Wan I2V：必须完整保留动作句；超长时只裁视觉锚点，禁止从动作中间腰斩。

    非首镜（shot_index>0）把动作句置顶，让 I2V 优先服从本镜动作描述，
    强化本镜动作与首帧姿态的一致性——通用机制，对任何主题/动作统一生效。
    """
    prompt = dedupe_prompt_text(prompt.strip().replace("\n", " "))
    prompt = re.sub(
        r"\bSame character, same scene\.?\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\bsame person,?\s*(?:identical appearance|keep pose natural|same face|same outfit)(?:\s+\w+){0,6}\.?\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\bsame (?:person same face same outfit|subject same appearance)\.?\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"同一主体同一外观[。.]?\s*", "", prompt)
    prompt = prompt.strip(" ,.;，。；")

    use_zh = bool(re.search(r"[\u4e00-\u9fff]", prompt))
    # 拆成「锚点。动作」——动作句绝不能被截断
    if use_zh:
        parts = [p.strip(" ,.;，。；") for p in re.split(r"[。．]\s*", prompt) if p.strip(" ,.;，。；")]
    else:
        parts = [p.strip(" ,.;") for p in re.split(r"\.\s+", prompt) if p.strip(" ,.;")]
    if len(parts) >= 2:
        identity = parts[0]
        sep = "。" if use_zh else ". "
        action = sep.join(parts[1:])
    else:
        identity = ""
        action = parts[0] if parts else prompt

    # 去掉动作句里残留的身份锁短语
    action = re.sub(
        r"\bsame person(?: same face)?(?: same outfit)?\b",
        "",
        action,
        flags=re.IGNORECASE,
    )
    action = re.sub(r"同一主体同一外观", "", action).strip(" ,.;，。；")
    action = boost_i2v_motion_prompt(
        action,
        prev_was_locomotion=prev_was_locomotion,
        rear_pushin=rear_pushin,
        context_text=context_text,
    )
    prop_lock = _prop_lock_for_i2v(f"{identity} {action} {prompt}")
    if prop_lock:
        if use_zh:
            if prop_lock not in action:
                action = f"{action}，{prop_lock}"
        elif prop_lock.lower() not in action.lower():
            action = f"{action}, {prop_lock}"

    # 动作/锚点各自去重；质量后缀单独拼接，禁止被 2-gram 去重撕成残句
    action = dedupe_prompt_text(action)
    identity = dedupe_prompt_text(identity) if identity else ""

    suffix = QUALITY_SUFFIX
    if use_zh:
        max_chars = MAX_I2V_PROMPT_WORDS * 2
        body = "。".join(p for p in (action, identity) if p).strip(" ，。；")
        text = f"{body}。{suffix}" if body else suffix
        if len(text) > max_chars:
            keep = max(24, max_chars - len(suffix) - 2)
            body = (action or identity or "")[:keep].rstrip(" ，。；")
            text = f"{body}。{suffix}"
        return text.rstrip(" ，。；") + "。"

    suffix_words = suffix.split()
    action_words = action.split()
    identity_words = identity.split()
    reserved = len(suffix_words) + len(action_words) + 2
    budget = MAX_I2V_PROMPT_WORDS - reserved
    if budget < 4:
        keep_action = max(10, MAX_I2V_PROMPT_WORDS - len(suffix_words) - 6)
        action_words = _trim_to_budget(action_words, keep_action)
        identity_words = identity_words[:4]
    else:
        identity_words = _trim_to_budget(identity_words, budget)

    chunks = []
    if action_words and identity_words:
        chunks.append(" ".join(action_words).strip(" ,.;"))
        chunks.append(" ".join(identity_words).strip(" ,.;"))
    else:
        if identity_words:
            chunks.append(" ".join(identity_words).strip(" ,.;"))
        if action_words:
            chunks.append(" ".join(action_words).strip(" ,.;"))
    body = ". ".join(chunks).strip(" ,.;")
    text = f"{body}. {suffix}" if body else suffix
    words = text.split()
    if len(words) > MAX_I2V_PROMPT_WORDS:
        overflow = len(words) - MAX_I2V_PROMPT_WORDS
        if len(identity_words) >= overflow:
            identity_words = identity_words[: len(identity_words) - overflow]
        else:
            need = overflow - len(identity_words)
            identity_words = []
            action_words = _trim_to_budget(action_words, max(12, len(action_words) - need))
        chunks = []
        if action_words and identity_words:
            chunks.append(" ".join(action_words).strip(" ,.;"))
            chunks.append(" ".join(identity_words).strip(" ,.;"))
        else:
            if identity_words:
                chunks.append(" ".join(identity_words).strip(" ,.;"))
            if action_words:
                chunks.append(" ".join(action_words).strip(" ,.;"))
        body = ". ".join(chunks).strip(" ,.;")
        text = f"{body}. {suffix}" if body else suffix
    return text.rstrip(" ,.;") + "."


def _soften_glowing_prop_words(prompt: str) -> str:
    """仅缓和发光道具被画成光剑/光团/过曝吃头的风险，不改构图与物种。"""
    text = prompt.strip().replace("\n", " ").strip(" ,.;，。；")
    text = re.sub(
        r"\bglowing\s+((?:blue|red|green|cyan|purple|golden|transparent|clear)\s+)?([a-z-]+)\b",
        r"physical \1\2 with soft rim light",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:发光|透亮|霓虹光)\s*(?:的)?\s*(伞|雨伞|剑|刀|装置|设备)",
        r"实体\1，边缘柔光",
        text,
    )
    # 伞类：减弱半透明发光导致伞盖过曝切平头顶
    text = re.sub(
        r"\b(translucent|transparent|clear)\s+(?:rain\s+)?umbrella(?:\s+with\s+soft\s+cyan\s+rim\s+light)?\b",
        "physical rain umbrella with soft rim light, no overexposed bloom",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ,.;")


def _prop_bits_for_keyframe(prompt: str) -> list[str]:
    """仅当文本出现手持道具时追加；不假定人类四肢或脸，不写死主题。"""
    prop = detect_handheld_prop(prompt)
    if not prop and re.search(r"伞|雨伞|\bumbrella\b", prompt, flags=re.I):
        prop = "umbrella"
    if not prop:
        return []
    prop_zh = {
        "umbrella": "雨伞",
        "sword": "剑",
        "gun": "枪",
        "phone": "手机",
    }.get(prop, prop)
    bits = [
        f"清晰展示实体{prop_zh}，不要做成能量刃或光团",
        f"{prop_zh}不要遮挡或取代主体主要辨识特征",
        f"{prop_zh}握持自然：持握手在身前或身侧，手臂不要拧到背后，肘部不要锁在脊柱后",
        f"{prop_zh}不得刺穿、融进或从主体头颈/躯干长出",
        f"握持手到{prop_zh}有可见连续连接，不要悬空，不要焊在身上",
        "头顶完整：任何头顶上方的道具下仍可见圆润头顶，禁止切平天灵盖",
    ]
    if prop == "umbrella":
        bits.append(
            "雨伞：伞面在头顶上方，伞柄与手之间有可见伞杆；"
            "手始终在身侧或身前握柄；伞面中心不得融进或切掉头骨；"
            "仅柔光，禁止过曝光晕吞噬头部"
        )
    return bits


def _prop_lock_for_i2v(prompt: str) -> str:
    """I2V：有手持道具时锁住「不脱手、不飞走、不切头」。通用，不写死主题。"""
    prop = detect_handheld_prop(prompt)
    if not prop and re.search(r"伞|雨伞|\bumbrella\b", prompt, flags=re.I):
        prop = "umbrella"
    if not prop:
        return ""
    prop_zh = {
        "umbrella": "雨伞",
        "sword": "剑",
        "gun": "枪",
        "phone": "手机",
    }.get(prop, prop)
    return (
        f"整段镜头{prop_zh}牢牢握在持握手中，"
        f"不要脱手飞走或换手，"
        f"伞面或道具顶部不要切掉头顶"
    )


def build_keyframe_model_prompt(
    prompt: str,
    action_hint: str = "",
    user_topic: str = "",
    shot_index: int = 0,
    *,
    rear_pushin: bool = False,
) -> str:
    """
    任意镜首帧：设定短语置顶（仅当主题里出现时），开场姿态匹配分镜动作起点，不发明道具。
    不写死人类/全身/必须动起来；非行进主题不会被强迫迈步。
    """
    prompt = _soften_glowing_prop_words(prompt)
    action_hint = _soften_glowing_prop_words(action_hint) if action_hint else ""
    setting = extract_setting_phrases(user_topic, include_climax=False)
    if should_include_climax_in_keyframe(user_topic, action_hint, shot_index):
        for phrase in extract_climax_object_phrases(user_topic):
            if phrase not in setting:
                setting.append(phrase)

    hard = []
    if setting:
        hard.append("画面必须出现设定：" + "，".join(setting))
    hard.extend(
        [
            "主体清晰可辨，与描述一致",
            "开场姿态匹配本镜动作起点，不要发明另一套动作",
        ]
    )
    if should_include_climax_in_keyframe(user_topic, action_hint, shot_index):
        hard.append("主题高潮物件须入画可读，禁止仅用画外暗示")
        if is_climax_enter_action(action_hint, user_topic):
            hard.append("高潮物件正从画面上方或边缘进入，开场不要已在画面中心静止占位")
        elif prefers_tilt_up_camera(action_hint):
            hard.append("构图预留画面上方空间，便于物件进入或被注视")
    elif shot_index == 0 and extract_climax_object_phrases(user_topic):
        hard.append("开场镜禁止出现主题后段高潮物件")

    # 行进类通用硬规则
    if is_locomotion_action(action_hint):
        if shot_index == 0:
            hard.extend(
                [
                    "行进构图：向纵深远离镜头（背影纵深或轻过肩），"
                    "禁止横穿画面行走，避免正对镜头迎面走来",
                    "行进开场姿态：身体前倾，一脚在前一脚在后，已进入迈步，"
                    "禁止完全直立站桩，禁止双脚并拢定格",
                    "姿态是正在向纵深迈步，不是站着不动",
                ]
            )
        elif CONTINUITY_MODE in {"hybrid", "handoff_only"}:
            hard.extend(
                [
                    "在同一地点、相近机位侧继续行进",
                    "继续向纵深行进，不要翻成横穿画面走",
                    "主体位置随迈步改变，不要大幅换视角",
                    "姿态匹配本镜行进，地面接触可读",
                ]
            )
        else:
            hard.extend(
                [
                    "行进构图：向纵深（背影纵深、轻过肩或适度四分之三均可），"
                    "避免正对迎面，避免横穿 mid-stride",
                    "姿态匹配本镜行进，双支撑接触可读",
                ]
            )
        hard.extend(
            [
                "躯干直立、重心稳定，禁止开场呈跪趴或失衡倾倒姿态",
                "禁止夸张弓步、劈叉、后仰",
                "主体与地面（或支撑面）接触可读",
                "主体受光可读（轮廓光或环境补光均可）——"
                "禁止融进背景的纯黑剪影",
            ]
        )
    elif action_hint:
        hard.append(f"开场姿态准备做：{action_hint}")
        hard.append("主体受光可读（轮廓光或环境补光均可）——禁止纯黑剪影")
        # 高潮入画 / 背后推镜：背影纵深起点（结构规则，主题无关）
        if rear_pushin or prefers_reveal_pushin(action_hint, user_topic) or is_climax_enter_action(
            action_hint, user_topic
        ):
            hard.extend(
                [
                    "保持背影或过肩纵深构图，主体背部朝向镜头",
                    "机位侧与主体尺度起点一致；本静帧是推镜的起点",
                    "构图预留画面上方空间，便于物件进入",
                    "不要翻成正脸或横穿画面构图",
                ]
            )
        elif requires_face_readable_action(action_hint):
            hard.extend(
                [
                    "必须脸部可读，或清晰四分之三/侧脸/过肩构图，以便看见下巴/眼神/头动",
                    "禁止：本镜使用全背影/背对镜头剪影",
                    "不要发明手持器械或枪械",
                    "若为过肩/侧脸微动：首帧已是过肩或侧脸定妆，"
                    "禁止生成大幅度转头过程",
                ]
            )
        if shot_index > 0:
            hard.append(
                    "仅做姿态变化以匹配本镜动作；保持同一环境风格、天气与光色——"
                    "不要换地点；目光在场景内或画外，禁止看向镜头"
            )

    hard.extend(_prop_bits_for_keyframe(f"{prompt} {action_hint}"))

    soft_bits = [
        "写实摄影感",
        "电影光影",
        "清晰对焦",
        "不要发明描述中没有的道具或特征",
    ]

    lead = f"{'，'.join(setting)}。" if setting else ""
    text = f"{'，'.join(hard)}。{lead}{prompt}。{'，'.join(soft_bits)}。"
    max_chars = MAX_KEYFRAME_PROMPT_WORDS * 2
    if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 8 and len(text) > max_chars:
        hard_part = "，".join(hard)
        soft_part = "，".join(soft_bits)
        mid_budget = max(24, max_chars - len(hard_part) - len(soft_part) - 4)
        mid = f"{lead}{prompt}"[:mid_budget]
        text = f"{hard_part}。{mid}。{soft_part}。"
    else:
        words = text.split()
        if len(words) > MAX_KEYFRAME_PROMPT_WORDS and not re.search(r"[\u4e00-\u9fff]", text):
            hard_words = ", ".join(hard).split()
            soft_words = ", ".join(soft_bits).split()
            mid_budget = max(12, MAX_KEYFRAME_PROMPT_WORDS - len(hard_words) - len(soft_words) - 2)
            mid_src = f"{lead}{prompt}".split()
            mid = mid_src[:mid_budget]
            text = f"{', '.join(hard)}. {' '.join(mid)}. {', '.join(soft_bits)}."
            words = text.split()
            if len(words) > MAX_KEYFRAME_PROMPT_WORDS:
                text = " ".join(words[:MAX_KEYFRAME_PROMPT_WORDS])
    return text.rstrip(" ,.;，。；") + "。"


def build_video_prompt(visual_anchor: str, shot_text: str, shot_index: int = 0) -> str:
    base = compose_i2v_prompt(visual_anchor, shot_text, shot_index)
    return build_i2v_model_prompt(base)


def build_scene_keyframe_prompt(
    visual_anchor: str,
    keyframe_prompts: list[str] | None,
    raw_shots: list[str] | None,
    user_topic: str,
    shot_index: int = 0,
    *,
    rear_pushin: bool = False,
) -> str:
    """任意镜首帧：主题设定置顶 + 本镜动作可见 + 身份锚点锁主角形象。
    每镜独立 T2I 生成不同首帧，身份靠 visual_anchor 文字锁大致一致（硬切不插帧）。"""
    action_hint = str((raw_shots or [""])[shot_index] or "").strip() if raw_shots and shot_index < len(raw_shots) else ""
    identity = enrich_visual_anchor(visual_anchor or "", user_topic)

    if keyframe_prompts and shot_index < len(keyframe_prompts) and str(keyframe_prompts[shot_index] or "").strip():
        base = str(keyframe_prompts[shot_index]).strip()
    elif action_hint:
        base = compose_i2v_prompt(identity, action_hint, shot_index)
    elif user_topic.strip():
        base = f"Opening scene: {user_topic.strip()}" if shot_index == 0 else f"Scene {shot_index + 1}: {user_topic.strip()}"
    else:
        base = identity or "subject in scene"

    # 非首镜叠身份锁文案
    parts = [p for p in (identity, base) if p]
    if shot_index > 0:
        if character_lock_enabled():
            parts.append("与角色素材库同一主体同一外观，服装与身份一致")
        else:
            parts.append("same subject same appearance as the opening scene, identical outfit and identity")
    combined = "。".join(dict.fromkeys(parts)) if re.search(r"[\u4e00-\u9fff]", "".join(parts)) else ". ".join(dict.fromkeys(parts))
    if shot_index == 0:
        print(f"🧷 Scene1 设定短语: {extract_setting_phrases(user_topic) or '(none)'}")
    return build_keyframe_model_prompt(
        combined,
        action_hint=action_hint,
        user_topic=user_topic,
        shot_index=shot_index,
        rear_pushin=rear_pushin,
    )


# 兼容旧调用名
build_scene1_keyframe_prompt = build_scene_keyframe_prompt


def _t2i_extract_url(rsp) -> str | None:
    """兼容不同 T2I 响应形状，取出首图 URL。"""
    # sync_call / call 通用：rsp.output.results[0].url
    output = getattr(rsp, "output", None)
    if output is not None:
        results = getattr(output, "results", None)
        if isinstance(results, list) and results:
            url = getattr(results[0], "url", None) or (results[0].get("url") if isinstance(results[0], dict) else None)
            if url:
                return url
        # 某些模型把 url 直接放在 output 上
        url = getattr(output, "url", None)
        if isinstance(url, str) and url:
            return url
    # 兼容 dict 风格
    if isinstance(rsp, dict):
        out = rsp.get("output") or {}
        results = out.get("results") or []
        if results and results[0].get("url"):
            return results[0]["url"]
        if out.get("url"):
            return out["url"]
    return None


def _t2i_call_once(model: str, prompt: str, negative: str, size: str):
    """按模型选择 sync_call 或 call（异步任务用 wait）。"""
    kwargs = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative,
        "n": 1,
        "size": size,
    }
    if model in _SYNC_T2I_MODELS and hasattr(ImageSynthesis, "sync_call"):
        return ImageSynthesis.sync_call(**kwargs)
    # 旧异步模型：call + wait
    task = ImageSynthesis.call(**kwargs)
    if getattr(task, "status_code", 0) == 200 and hasattr(ImageSynthesis, "wait"):
        return ImageSynthesis.wait(task)
    return task


def call_wanx_generate_api(prompt: str, output_path: str) -> str:
    print("\n🎨 [AI 首帧生成] 正在请求通义万相云端生成基准大图...")
    print(f"💬 画面描述词: {prompt}")
    print(f"🧩 T2I 模型: {WANX_T2I_MODEL}")

    dashscope.api_key = _dashscope_api_key()

    candidates = [WANX_T2I_MODEL] + [m for m in _T2I_FALLBACK_CHAIN if m != WANX_T2I_MODEL]
    rsp = None
    used_model = None
    for model in candidates:
        try:
            rsp = _t2i_call_once(model, prompt, KEYFRAME_NEGATIVE_PROMPT, WANX_SIZE)
        except Exception as exc:
            print(f"⚠️ {model} 调用异常（{exc}），尝试下一个...")
            rsp = None
            continue
        status = getattr(rsp, "status_code", None) or (rsp.get("status_code") if isinstance(rsp, dict) else None)
        url = _t2i_extract_url(rsp)
        if status == 200 and url:
            used_model = model
            break
        msg = getattr(rsp, "message", None) or (rsp.get("message") if isinstance(rsp, dict) else None)
        print(f"⚠️ {model} 未成功（status={status}, msg={msg}），尝试下一个...")
        rsp = None

    if rsp is None:
        raise RuntimeError("所有 T2I 模型均失败，请检查 DASHSCOPE_API_KEY 与模型可用性")

    img_url = _t2i_extract_url(rsp)
    if not img_url:
        raise RuntimeError(f"T2I 响应无图片 URL: {rsp}")

    if used_model != WANX_T2I_MODEL:
        print(f"ℹ️ 实际使用 T2I 模型: {used_model}（{WANX_T2I_MODEL} 不可用已回退）")

    img_data = requests.get(img_url, timeout=120).content
    with open(output_path, "wb") as f:
        f.write(img_data)

    img = Image.open(output_path).convert("RGB")
    img_resized = fit_image_to_video_size(img, 1280, 720)
    img_resized.save(output_path, quality=95)
    print(f"🖼️ 首帧大图已成功下载并保存至: {output_path}（模型 {used_model}）")
    return output_path


def ensure_character_asset(
    visual_anchor: str,
    user_topic: str = "",
    *,
    job_copy_path: str | None = None,
) -> str:
    """
    解析或生成角色素材，返回本地图片路径。

    优先级：
    1. CHARACTER_ASSET_PATH / 本主题已入库的定妆卡（精确 ID）
    2. 按主题用万相免费生成贴合身份的定妆卡并入库
    3. 生成失败时才用 manual/default 通用脸兜底
    """
    existing = find_matched_character_asset(visual_anchor, user_topic)
    if existing:
        print(f"🧬 角色素材库命中（本主题已有定妆）: {existing}")
        path = existing
    else:
        asset_id = character_asset_id(visual_anchor, user_topic)
        sheet_prompt = build_character_sheet_prompt(
            enrich_visual_anchor(visual_anchor or "", user_topic),
            user_topic,
        )
        tmp_dir = os.path.dirname(job_copy_path) if job_copy_path else os.getcwd()
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"_character_sheet_{asset_id}.png")
        print(f"🧬 本主题尚无定妆卡，按主题生成（id={asset_id}）...")
        print(f"   定妆 prompt: {sheet_prompt[:180]}{'...' if len(sheet_prompt) > 180 else ''}")
        try:
            call_wanx_generate_api(prompt=sheet_prompt, output_path=tmp_path)
            path = register_generated_asset(
                asset_id,
                tmp_path,
                visual_anchor=visual_anchor,
                user_topic=user_topic,
                prompt=sheet_prompt,
            )
            print(f"🧬 已入库主题定妆卡: {path}")
        except Exception as exc:
            fallback = find_fallback_default_asset()
            if not fallback:
                raise
            print(f"⚠️ 主题定妆生成失败（{exc}），回退通用素材脸: {fallback}")
            path = fallback

    if job_copy_path:
        from shutil import copy2

        os.makedirs(os.path.dirname(job_copy_path) or ".", exist_ok=True)
        if os.path.abspath(path) != os.path.abspath(job_copy_path):
            copy2(path, job_copy_path)
        return job_copy_path
    return path


def pick_identity_edit_instruction(
    *,
    keyframe_prompt: str,
    action_hint: str,
    visual_anchor: str,
    shot_index: int,
) -> str:
    """素材库模式用角色锁指令；旧 scene1 模式用原 identity-edit 指令。"""
    if character_lock_enabled():
        return build_character_lock_instruction(
            keyframe_prompt=keyframe_prompt,
            action_hint=action_hint,
            visual_anchor=visual_anchor,
            shot_index=shot_index,
        )
    return build_identity_edit_instruction(
        keyframe_prompt=keyframe_prompt,
        action_hint=action_hint,
        visual_anchor=visual_anchor,
        shot_index=shot_index,
    )


def _encode_image_as_data_url(file_path: str) -> str:
    """本地图片转 data URL，供 qwen-image-edit 多模态接口使用。"""
    mime, _ = mimetypes.guess_type(file_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _dashscope_api_key() -> str:
    load_local_env()
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY。请在环境变量或项目目录 .env 中配置，不要把密钥写进代码。"
        )
    return key


def build_identity_edit_instruction(
    keyframe_prompt: str,
    action_hint: str = "",
    visual_anchor: str = "",
    shot_index: int = 1,
) -> str:
    """通用身份保真编辑指令：锁 Image1 主角形象与环境风格，只改姿态/轻微机位。
    明确禁止非开场镜复刻 Image1 的背影行进构图。不写死任何主题词。"""
    action = (action_hint or "").strip()
    setting = (visual_anchor or "").strip()
    scene = (keyframe_prompt or "").strip()

    parts = [
        "保持图像1中主体身份完全一致：同一张脸、发型、体型、服装与已描述的手持道具。"
        "不要换成另一个角色。",
        "关键环境锁定：保持与图像1相同的地点风格、建筑、天气与光色。"
        "可适度调整机位与主体姿态，但不要发明另一条街、另一种巷子风格或另一时段。"
        "背景连续性优先于新奇。",
        "生成本镜新的静帧姿态。不要原样复制图像1构图。"
        "若图像1是背影纵深行走，除非本镜动作就是同一入场行走，否则换明显不同的角度/朝向。",
        "不要发明动作/描述中没有的枪械或器械。",
        "主体目光在场景内或画外——禁止看向镜头。",
    ]
    if action:
        parts.append(f"本镜要求的主体动作/姿态：{action}。")
        if shot_index > 0 and not is_locomotion_action(action):
            parts.append(
                "这不是背影走开镜：动作需要脸或侧脸可读；"
                "不要再生成另一个背对镜头行走。"
            )
            if requires_face_readable_action(action):
                parts.append(
                    "必须：脸部可读或清晰四分之三/侧脸，以便看见下巴/眼神/头动。"
                    "禁止：全背影兜帽剪影/背对镜头。不要发明枪或手持器械。"
                )
        elif shot_index > 0 and is_locomotion_action(action):
            parts.append(
                "行进可以：相对图像1做适度构图变化，继续向纵深"
                "（轻过肩或轻微角度变化）。不要发明横穿画面 mid-stride 行走。"
            )
    if setting:
        parts.append(f"设定身份与以下保持一致：{setting}。")
    if scene:
        parts.append(f"场景描述：{scene[:280]}。")
    parts.append(
        "写实电影静帧，主体受光可读（轮廓光可），"
        "自然光影匹配图像1，不是纯黑剪影。"
        "一个连贯可读姿态，便于图生视频。"
    )
    return "".join(parts)


def _qwen_edit_extract_url(rsp) -> str | None:
    """从 MultiModalConversation 响应取出首张输出图 URL。"""
    try:
        choices = getattr(getattr(rsp, "output", None), "choices", None)
        if not choices:
            if isinstance(rsp, dict):
                choices = (rsp.get("output") or {}).get("choices") or []
            else:
                return None
        message = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message")
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if not content:
            return None
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                return item["image"]
            url = getattr(item, "image", None)
            if url:
                return url
    except Exception:
        return None
    return None


def call_qwen_identity_edit_api(
    reference_image_path: str,
    edit_prompt: str,
    output_path: str,
) -> str | None:
    """用 qwen-image-edit-max 以参考图保身份，生成本镜新姿态/场景首帧。
    失败返回 None，调用方回退独立 T2I。"""
    if not _HAS_MULTIMODAL or MultiModalConversation is None:
        print("⚠️ dashscope.MultiModalConversation 不可用，跳过身份编辑")
        return None
    if not os.path.exists(reference_image_path):
        print(f"⚠️ 身份参考图不存在: {reference_image_path}")
        return None

    print("\n🪪 [身份保真编辑] 正在请求 qwen-image-edit（参考角色素材/身份图）...")
    print(f"💬 编辑指令: {edit_prompt[:220]}{'...' if len(edit_prompt) > 220 else ''}")

    api_key = _dashscope_api_key()
    dashscope.api_key = api_key
    try:
        image_data_url = _encode_image_as_data_url(reference_image_path)
    except Exception as exc:
        print(f"⚠️ 参考图编码失败（{exc}）")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_data_url},
                {"text": edit_prompt},
            ],
        }
    ]
    candidates = [IDENTITY_EDIT_MODEL] + [m for m in _IDENTITY_EDIT_FALLBACK if m != IDENTITY_EDIT_MODEL]
    negative = (
        "different person, identity change, face swap, wrong outfit, "
        "extra limbs, deformed, blurry, watermark, text overlay, "
        "different environment style, different weather, different lighting palette, "
        "looking into the camera, eye contact with viewer, "
        "invented firearm, invented gun, pistol not described"
    )

    for model in candidates:
        try:
            rsp = MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                negative_prompt=negative,
                prompt_extend=True,
                size=IDENTITY_EDIT_SIZE,
            )
        except Exception as exc:
            print(f"⚠️ {model} 调用异常（{exc}），尝试下一个...")
            continue

        status = getattr(rsp, "status_code", None) or (rsp.get("status_code") if isinstance(rsp, dict) else None)
        img_url = _qwen_edit_extract_url(rsp)
        if status == 200 and img_url:
            try:
                img_data = requests.get(img_url, timeout=120).content
                with open(output_path, "wb") as f:
                    f.write(img_data)
                img = Image.open(output_path).convert("RGB")
                img_resized = fit_image_to_video_size(img, 1280, 720)
                img_resized.save(output_path, quality=95)
                print(f"🪪 身份保真首帧已保存: {output_path}（模型 {model}）")
                return output_path
            except Exception as exc:
                print(f"⚠️ {model} 下载/保存失败（{exc}），尝试下一个...")
                continue

        msg = getattr(rsp, "message", None) or (rsp.get("message") if isinstance(rsp, dict) else None)
        print(f"⚠️ {model} 未成功（status={status}, msg={msg}），尝试下一个...")

    print("⚠️ 所有身份编辑模型均失败，将回退独立 T2I 文字锁")
    return None


def _postprocess_clip(video_path: str) -> None:
    temp_path = video_path + ".tmp.mp4"
    cmd = (
        f'ffmpeg -y -i "{video_path}" -c:v libx264 -crf 18 -preset medium '
        f'-pix_fmt yuv420p -movflags +faststart "{temp_path}"'
    )
    code = os.system(cmd)
    if code != 0:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # ffmpeg 缺失时原片仍可用（export_to_video 已写出），只跳过再编码
        print(
            f"⚠️ ffmpeg 后处理失败（exit={code}）。原片已保留: {video_path}。"
            f" AutoDL 可装: apt-get update && apt-get install -y ffmpeg"
        )
        return
    os.replace(temp_path, video_path)


def _read_cgroup_memory_value(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw in ("max", "9223372036854771712"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_memory_budget() -> dict:
    """返回容器/系统内存预算。AutoDL 等环境以 cgroup 限额为准，而非宿主机总内存。"""
    cgroup_limit = None
    cgroup_usage = None
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit = _read_cgroup_memory_value(limit_path)
        if limit is None:
            continue
        cgroup_limit = limit
        cgroup_usage = _read_cgroup_memory_value(usage_path) or 0
        break

    host_total = None
    host_available = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        host_total = memory.total
        host_available = memory.available
    except ImportError:
        pass

    effective_limit = cgroup_limit or host_total
    if cgroup_limit is not None:
        effective_used = cgroup_usage or 0
        effective_available = max(0, cgroup_limit - effective_used)
    elif host_total is not None and host_available is not None:
        effective_used = host_total - host_available
        effective_available = host_available
    else:
        effective_used = None
        effective_available = None

    return {
        "cgroup_limit_gb": cgroup_limit / (1024 ** 3) if cgroup_limit else None,
        "cgroup_used_gb": cgroup_usage / (1024 ** 3) if cgroup_usage is not None else None,
        "host_total_gb": host_total / (1024 ** 3) if host_total else None,
        "host_available_gb": host_available / (1024 ** 3) if host_available else None,
        "effective_limit_gb": effective_limit / (1024 ** 3) if effective_limit else None,
        "effective_available_gb": effective_available / (1024 ** 3) if effective_available is not None else None,
        "effective_used_gb": effective_used / (1024 ** 3) if effective_used is not None else None,
    }


def prepare_memory_for_model_load() -> None:
    """加载大模型前尽量释放 RAM / VRAM。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    budget = get_memory_budget()
    if budget["cgroup_limit_gb"] is not None:
        print(
            f"ℹ️ 容器内存: 限额 {budget['cgroup_limit_gb']:.1f} GB, "
            f"已用 {budget['effective_used_gb']:.1f} GB, "
            f"剩余约 {budget['effective_available_gb']:.1f} GB"
        )
        if budget["host_total_gb"] is not None:
            print(
                f"ℹ️ 宿主机内存: 总计 {budget['host_total_gb']:.1f} GB, "
                f"可用 {budget['host_available_gb']:.1f} GB（容器进程无法全部使用）"
            )
    elif budget["host_total_gb"] is not None:
        print(
            f"ℹ️ 系统内存: 总计 {budget['host_total_gb']:.1f} GB, "
            f"可用 {budget['host_available_gb']:.1f} GB"
        )

    # Wan2.2 MoE 双 Transformer + T5，分步加载尖峰高于 Wan2.1
    min_required_gb = 40 if WAN_LOW_VRAM else 90
    available_gb = budget["effective_available_gb"]
    used_gb = budget["effective_used_gb"]
    limit_gb = budget["effective_limit_gb"]
    if used_gb is not None and limit_gb is not None and used_gb >= 70:
        print(
            f"⚠️ 常态内存已用 {used_gb:.1f} GB / {limit_gb:.1f} GB。"
            "加载 Wan2.2-A14B 时极易被 Killed。请先运行: bash diagnose_memory.sh"
        )
        print(
            "   常见原因: 其他终端残留的 python/celery、多次加载产生的页缓存。"
            "可尝试: export WAN_DROP_PAGE_CACHE=1 后重试，或重启实例。"
        )
    if available_gb is not None and available_gb < min_required_gb:
        print(
            f"⚠️ 容器剩余内存约 {available_gb:.1f} GB，低于 Wan2.2 分步加载建议值 {min_required_gb} GB。"
            "加载时若出现 Killed，请关闭其他终端任务/Jupyter 进程，"
            "sync && echo 3 > /proc/sys/vm/drop_caches，或重启实例后再 start_worker。"
        )


def _try_drop_page_cache() -> None:
    drop_path = "/proc/sys/vm/drop_caches"
    if not os.path.exists(drop_path):
        return
    os.system("sync")
    try:
        with open(drop_path, "w", encoding="utf-8") as f:
            f.write("3\n")
        print("ℹ️ 已释放页缓存（page cache），常态内存占用应会下降")
        gc.collect()
    except OSError:
        print("⚠️ 无法写入 drop_caches；可在终端执行: sync && echo 3 > /proc/sys/vm/drop_caches")


def _load_pretrained_submodule(
    model_cls,
    model_source: str,
    subfolder: str,
    use_local: bool,
    label: str,
    dtype: torch.dtype | None = None,
    heavy: bool = False,
):
    """分步加载子模块到 CPU。heavy 参数保留兼容，不再使用 disk device_map（会触发 RuntimeError）。"""
    del heavy  # 120GB 内存足够整模进 RAM；disk max_memory 在当前 accelerate 下会报 device string: disk
    print(f"   📦 正在加载 {label}...")
    prepare_memory_for_model_load()
    module = model_cls.from_pretrained(
        model_source,
        subfolder=subfolder,
        torch_dtype=dtype or torch.float32,
        local_files_only=use_local,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"   ✅ {label} 加载完成")
    return module


def _subfolder_has_weights(model_source: str, subfolder: str) -> bool:
    """本地目录是否含该子模块权重（Wan2.2 的 image_encoder 为 null，不可强行加载）。"""
    base = Path(model_source)
    if not base.is_dir():
        return True  # 远端拉取交给 from_pretrained
    sub = base / subfolder
    if not sub.is_dir():
        return False
    for name in (
        "model.safetensors",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        if (sub / name).exists():
            return True
    # 分片权重
    return any(sub.glob("*.safetensors")) or any(sub.glob("pytorch_model*.bin"))


def _enable_fp8_layerwise_casting(pipe: WanImageToVideoPipeline) -> bool:
    """transformer 权重以 FP8 存储、BF16 计算：offload 传输量减半，速度显著提升。"""
    try:
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        print("ℹ️ transformer 已启用 FP8 layerwise casting（传输量减半，提速）")
        if getattr(pipe, "transformer_2", None) is not None:
            pipe.transformer_2.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            print("ℹ️ transformer_2 已启用 FP8 layerwise casting")
        return True
    except Exception as error:
        print(f"⚠️ FP8 layerwise casting 不可用（{error}），保持 BF16")
        return False


def group_offload_supported() -> bool:
    try:
        from diffusers.hooks import apply_group_offloading  # noqa: F401
        return True
    except ImportError:
        print("⚠️ 当前 diffusers 版本不支持 group offload（需 >=0.33），回退 sequential。")
        print("   升级: pip install -U 'diffusers>=0.33.0'")
        return False


def _enable_group_offload(pipe: WanImageToVideoPipeline) -> bool:
    """
    分组 offload：transformer 逐层预取 + CUDA stream 重叠「权重搬运」与「计算」，
    显存占用与 sequential 相近（约 6-12GB），但速度快数倍。需 diffusers>=0.33。
    """
    from diffusers.hooks import apply_group_offloading

    try:
        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")
        for tf_name in ("transformer", "transformer_2"):
            tf = getattr(pipe, tf_name, None)
            if tf is None:
                continue
            tf.enable_group_offload(
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="leaf_level",
                use_stream=True,
            )
        for name in ("text_encoder", "image_encoder", "vae"):
            module = getattr(pipe, name, None)
            if module is None:
                continue
            apply_group_offloading(
                module,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
            )
        return True
    except Exception as error:
        print(f"⚠️ group offload 启用失败（{error}），回退 sequential。")
        return False


def unload_i2v_pipeline(pipe: WanImageToVideoPipeline | None) -> None:
    if pipe is None:
        return
    try:
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
    except Exception:
        pass
    del pipe
    prepare_memory_for_model_load()


def load_i2v_pipeline() -> tuple[WanImageToVideoPipeline, dict]:
    global WAN_OFFLOAD, ENABLE_ATTENTION_SLICING
    profile = resolve_wan_profile()
    model_source = profile["source"]
    use_local = profile["local"]

    # 运行时再判定一次，防止环境变量在 import 之后才设置
    WAN_OFFLOAD = resolve_wan_offload()
    ENABLE_ATTENTION_SLICING = os.getenv(
        "WAN_ATTENTION_SLICING",
        "1" if WAN_OFFLOAD == "sequential" else "0",
    ) == "1"

    prepare_memory_for_model_load()
    if WAN_DROP_PAGE_CACHE:
        _try_drop_page_cache()
        prepare_memory_for_model_load()

    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"ℹ️ GPU 显存: 空闲 {free_gb:.1f} / 总计 {total_gb:.1f} GB")

    print(f"🚀 正在分步加载 Wan2.2-I2V-A14B | offload={WAN_OFFLOAD}（权重先入 CPU RAM）")
    print("   ℹ️ Wan2.2 Diffusers 无独立 image_encoder（model_index 为 null），按 VAE/T5/双 Transformer 加载")

    # Wan2.2：image_encoder / image_processor 为 null，不可按 Wan2.1 强行加载 CLIP
    image_encoder = None
    if _subfolder_has_weights(model_source, "image_encoder"):
        image_encoder = _load_pretrained_submodule(
            CLIPVisionModel, model_source, "image_encoder", use_local, "image_encoder"
        )
    else:
        print("   ⏭️ 跳过 image_encoder（本权重包不含该组件）")

    vae = _load_pretrained_submodule(
        AutoencoderKLWan, model_source, "vae", use_local, "vae"
    )
    text_encoder = _load_pretrained_submodule(
        UMT5EncoderModel,
        model_source,
        "text_encoder",
        use_local,
        "text_encoder (UMT5-XXL)",
        dtype=torch.bfloat16,
    )
    transformer = _load_pretrained_submodule(
        WanTransformer3DModel,
        model_source,
        "transformer",
        use_local,
        "transformer（高噪声专家）",
        dtype=torch.bfloat16,
    )
    transformer_2 = None
    if _subfolder_has_weights(model_source, "transformer_2"):
        transformer_2 = _load_pretrained_submodule(
            WanTransformer3DModel,
            model_source,
            "transformer_2",
            use_local,
            "transformer_2（低噪声专家）",
            dtype=torch.bfloat16,
        )
    else:
        print("   ⚠️ 未找到 transformer_2；若为 Wan2.2 MoE 权重，请检查下载是否完整")

    print("   📦 组装 Pipeline（仅加载 tokenizer / scheduler 等轻量组件）...")
    prepare_memory_for_model_load()
    pipe_kwargs = dict(
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=use_local,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if image_encoder is not None:
        pipe_kwargs["image_encoder"] = image_encoder
    if transformer_2 is not None:
        pipe_kwargs["transformer_2"] = transformer_2

    pipe = WanImageToVideoPipeline.from_pretrained(model_source, **pipe_kwargs)

    if WAN_OFFLOAD == "group" and not group_offload_supported():
        WAN_OFFLOAD = "sequential"
        ENABLE_ATTENTION_SLICING = os.getenv("WAN_ATTENTION_SLICING", "1") == "1"

    if WAN_OFFLOAD == "group":
        # FP8 casting 必须在挂 offload hook 之前应用
        if WAN_FP8:
            _enable_fp8_layerwise_casting(pipe)
        if _enable_group_offload(pipe):
            print("ℹ️ 显存模式: group offload（分组预取+stream 重叠，24GB 推荐，比 sequential 快数倍）")
        else:
            WAN_OFFLOAD = "sequential"
            ENABLE_ATTENTION_SLICING = os.getenv("WAN_ATTENTION_SLICING", "1") == "1"

    if WAN_OFFLOAD == "sequential":
        pipe.enable_sequential_cpu_offload()
        print("ℹ️ 显存模式: sequential（保底方案；显存约 4–8GB，最慢）")
    elif WAN_OFFLOAD in ("none", "gpu", "full"):
        pipe.to("cuda")
        print("ℹ️ 显存模式: full GPU（最快，需约 ≥40GB 显存）")
    elif WAN_OFFLOAD != "group":
        pipe.enable_model_cpu_offload()
        print("ℹ️ 显存模式: model offload（整 transformer 上 GPU；24GB 上 Wan2.2-A14B 极易 OOM，需 ≥40GB）")

    if ENABLE_ATTENTION_SLICING and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
        print("ℹ️ attention slicing: max")
    else:
        print("ℹ️ attention slicing: 关闭（提速）")

    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if ENABLE_VAE_TILING and hasattr(pipe.vae, "enable_tiling"):
        # 解码 81 帧是全流程显存峰值；低显存下用更小的瓦片压低解码峰值
        try:
            if WAN_LOW_VRAM:
                pipe.vae.enable_tiling(
                    tile_sample_min_height=192,
                    tile_sample_min_width=192,
                    tile_sample_stride_height=128,
                    tile_sample_stride_width=128,
                )
                print("ℹ️ VAE tiling: 已开启（小瓦片模式，降低解码显存峰值）")
            else:
                pipe.vae.enable_tiling()
                print("ℹ️ VAE tiling: 已开启")
        except TypeError:
            pipe.vae.enable_tiling()
            print("ℹ️ VAE tiling: 已开启（当前 diffusers 不支持自定义瓦片尺寸）")
    elif not ENABLE_VAE_TILING:
        print("ℹ️ VAE tiling: 已关闭")

    configure_wan_scheduler(pipe)
    return pipe, profile


def _free_cache_before_decode(pipe, step_index, timestep, callback_kwargs):
    """降噪最后两步释放缓存块，压低随后 VAE 解码的显存峰值（第1镜实测冲到 ~24GB）。"""
    if torch.cuda.is_available() and step_index >= NUM_INFERENCE_STEPS - 2:
        torch.cuda.empty_cache()
    return callback_kwargs


def effective_max_area(profile: dict) -> int:
    base = profile["max_area"]
    if WAN_AREA_SCALE >= 0.99:
        return base
    scaled = int(base * WAN_AREA_SCALE)
    print(f"ℹ️ 分辨率面积缩放: {base} → {scaled} (scale={WAN_AREA_SCALE})")
    return scaled


def render_video_clips(
    prompts: list,
    output_dir="output_clips",
    visual_anchor: str = "",
    keyframe_prompts: list | None = None,
    raw_shots: list | None = None,
    continuity_spine: list | None = None,
    user_topic: str = "",
):
    """
    Scene1: 角色素材库锁人 → 再落到本镜开场姿态；不再把 Scene1 当人物回锚源。
    Scene2+: hybrid 默认交接帧直出锁背景；需要换姿态/脸部可读时用素材库重绘。
    成片镜间硬切。视频模型: Wan2.2-I2V-A14B
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for old_clip in os.listdir(output_dir):
            if old_clip.startswith("scene_") and old_clip.endswith(".mp4"):
                os.remove(os.path.join(output_dir, old_clip))

    clip_seconds = NUM_FRAMES / SOURCE_FPS
    total_seconds = clip_seconds * len(prompts)
    print(
        f"\n🎬 Wan2.2-I2V-A14B 流水线启动，一镜一连贯动作，共 {len(prompts)} 段 × "
        f"{clip_seconds:.1f}s ≈ {total_seconds:.0f}s 成片..."
    )
    print(
        f"📐 档位: {WAN_PROFILE} | low_vram={WAN_LOW_VRAM} | offload={WAN_OFFLOAD} | "
        f"帧数: {NUM_FRAMES} | fps: {SOURCE_FPS} | steps: {NUM_INFERENCE_STEPS} | "
        f"area_scale={WAN_AREA_SCALE} | "
        f"flow_shift={FLOW_SHIFT}/{LOCO_FLOW_SHIFT}(行进/入画) | "
        f"guidance={GUIDANCE_SCALE}/{GUIDANCE_SCALE_2} "
        f"loco_cfg={LOCO_GUIDANCE_SCALE}/{LOCO_GUIDANCE_SCALE_2} | "
        f"motion_retry={MOTION_RETRY}"
    )
    print(
        f"🔗 连贯策略: CONTINUITY_MODE={CONTINUITY_MODE} | "
        f"CHARACTER_LOCK={CHARACTER_LOCK_MODE} | "
        f"HYBRID_POSE_EDIT={int(HYBRID_POSE_EDIT)} | "
        f"SCENE1_REANCHOR_EVERY={IDENTITY_REANCHOR_EVERY} | "
        f"HANDOFF_FRAME_RATIO={HANDOFF_FRAME_RATIO} | 成片硬切"
    )
    if character_lock_enabled():
        print("   → 人物：免费角色素材库锁定（不用 Scene1 首帧回锚定人）")
    else:
        print("   → 人物：旧版 Scene1 首帧回锚（CHARACTER_LOCK_MODE=scene1）")
    if CONTINUITY_MODE == "hybrid":
        if HYBRID_POSE_EDIT:
            print("   → 下一镜 = 上镜交接帧 + qwen-edit 改姿态（易伤背景，慎用）")
        else:
            print(
                "   → 下一镜 = 交接帧直出锁背景；"
                "行进→停步/脸部动作时用角色素材重绘姿态"
            )
    elif CONTINUITY_MODE == "identity_lock":
        print("   → 下一镜首帧仅参考身份素材编辑")
    else:
        print("   → 下一镜首帧仅由上镜交接帧编辑（不混身份参考）")
    if not _HAS_MULTIMODAL:
        print("⚠️ MultiModalConversation 不可用，Scene2+ 将回退独立 T2I 文字锁（请升级 dashscope）")
    model_prompts = []

    print("🚀 正在初始化 Wan2.2-I2V-A14B...")
    image_pipe, wan_profile = load_i2v_pipeline()
    max_area = effective_max_area(wan_profile)

    identity_reference = None
    identity_ref_path = None
    character_asset_path = None
    prev_handoff_frame: Image.Image | None = None
    prev_handoff_index: int | None = None
    prev_shot_was_locomotion = False
    prev_shot_was_face_readable = False

    # 开拍前准备角色素材（免费库）；锚点去掉高潮物件，避免定妆卡画上静止飞行物
    if character_lock_enabled():
        asset_anchor = strip_climax_phrases_from_text(
            enrich_visual_anchor(visual_anchor or "", user_topic),
            user_topic,
        )
        character_asset_path = ensure_character_asset(
            asset_anchor,
            user_topic,
            job_copy_path=os.path.join(output_dir, "character_reference.png"),
        )
        identity_ref_path = character_asset_path
        identity_reference = Image.open(identity_ref_path).convert("RGB")
        print(f"   🧷 本任务身份参考 = 角色素材库: {identity_ref_path}")

    for i, prompt in enumerate(prompts):
        scene_no = i + 1
        filename = f"{output_dir}/scene_{scene_no:02d}.mp4"
        keyframe_path = os.path.join(output_dir, f"scene_{scene_no:02d}_keyframe.png")
        handoff_path = os.path.join(output_dir, f"scene_{scene_no:02d}_handoff.png")
        edit_base_path = os.path.join(output_dir, f"scene_{scene_no:02d}_edit_base.png")
        model_prompts_path = os.path.join(output_dir, "model_prompts.json")

        shot_text = raw_shots[i] if raw_shots and i < len(raw_shots) else prompt
        # 出片前再剥一次无中生有器械（防止旧分镜/未同步 LLM 规则漏网）
        ctx = f"{user_topic} {visual_anchor}"
        prior = [str(raw_shots[j]) for j in range(i)] if raw_shots else []
        cleaned_shot = sanitize_shot_motion(
            str(shot_text),
            context_text=ctx,
            previous=prior,
            index=i,
        ) or strip_invented_handheld_action(str(shot_text), context_text=ctx)
        if cleaned_shot and cleaned_shot != str(shot_text).strip():
            print(f"   🧹 动作清洗: '{shot_text}' → '{cleaned_shot}'")
            shot_text = cleaned_shot
            if raw_shots is not None and i < len(raw_shots):
                raw_shots[i] = cleaned_shot
        # 一律用清洗后动作重拼 I2V prompt，避免沿用已污染的 prompts[i]
        # 物件入画：强制背后平推（结构规则，主题无关；Wan 对「微仰」几乎不做运镜）
        rear_pushin = prefers_rear_pushin(
            shot_text, prev_was_locomotion=prev_shot_was_locomotion
        ) or prefers_reveal_pushin(shot_text, user_topic)
        if prefers_reveal_pushin(shot_text, user_topic):
            print("   🎥 物件入画：背后平推 + 物件进入（不用微仰）")
        elif rear_pushin:
            print("   🎥 背影延续：本镜用镜头推进，避免同构图硬切")
        composed_core = compose_i2v_prompt(
            visual_anchor,
            shot_text,
            i,
            context_text=ctx,
            previous=prior,
        )
        # compose 内会再 sanitize：若动作被改写，同步 shot_text，避免「分镜有无人机、I2V 没有」
        composed_action = ""
        if composed_core:
            _parts = [
                p.strip(" ,.;，。；")
                for p in re.split(r"[。．]\s*", composed_core)
                if p.strip(" ,.;，。；")
            ]
            if len(_parts) >= 2:
                composed_action = _parts[1]
            elif _parts:
                composed_action = _parts[0]
            composed_action = re.sub(r"同一主体同一外观", "", composed_action).strip(" ,.;，。；")
            if composed_action.startswith("他") and not str(shot_text).startswith("他"):
                composed_action = composed_action[1:]
        if composed_action and composed_action != str(shot_text).strip():
            print(f"   🧹 compose对齐: '{shot_text}' → '{composed_action}'")
            shot_text = composed_action
            if raw_shots is not None and i < len(raw_shots):
                raw_shots[i] = composed_action
            rear_pushin = prefers_rear_pushin(
                shot_text, prev_was_locomotion=prev_shot_was_locomotion
            ) or prefers_reveal_pushin(shot_text, user_topic)
        enhanced_prompt = build_i2v_model_prompt(
            composed_core,
            shot_index=i,
            prev_was_locomotion=prev_shot_was_locomotion,
            rear_pushin=rear_pushin,
            context_text=ctx,
        )

        if i == 0:
            keyframe_prompt = build_scene_keyframe_prompt(
                visual_anchor, keyframe_prompts, raw_shots, user_topic, shot_index=0
            )
            keyframe_source = "wanx"
        else:
            keyframe_prompt = build_scene_keyframe_prompt(
                visual_anchor,
                keyframe_prompts,
                raw_shots,
                user_topic,
                shot_index=i,
                rear_pushin=rear_pushin,
            )
            if CONTINUITY_MODE in {"hybrid", "handoff_only"} and not HYBRID_POSE_EDIT:
                keyframe_source = "direct_handoff"
            else:
                keyframe_source = f"qwen_edit_{CONTINUITY_MODE}"

        model_prompts.append(
            {
                "scene": scene_no,
                "model": wan_profile["model_id"],
                "profile": WAN_PROFILE,
                "prompt": enhanced_prompt,
                "keyframe_prompt": keyframe_prompt,
                "keyframe_source": keyframe_source,
                "continuity_mode": CONTINUITY_MODE,
                "hybrid_pose_edit": HYBRID_POSE_EDIT,
                "identity_blend_alpha": 0.0,
                "shot_text": shot_text,
                "rear_pushin": rear_pushin,
                "prompt_word_count": len(enhanced_prompt.split()),
                "continuity_action": (
                    str(continuity_spine[i].get("action", "")).strip()
                    if continuity_spine and i < len(continuity_spine)
                    else ""
                ),
                "width": None,
                "height": None,
                "motion_delta": None,
                "handoff_frame_index": None,
                "prev_handoff_frame_index": prev_handoff_index,
                "seed": None,
            }
        )

        print(f"\n⚡ 正在渲染 镜头 {scene_no}/{len(prompts)}")
        print(f"   视频 prompt ({len(enhanced_prompt.split())}词): {enhanced_prompt}")
        print(f"   首帧来源: {keyframe_source}")

        with open(model_prompts_path, "w", encoding="utf-8") as f:
            json.dump(model_prompts, f, ensure_ascii=False, indent=2)

        # 首帧策略：
        # - 素材库模式：身份参考=角色卡；Scene1 由角色卡编辑到开场姿态；
        #   Scene2+ 默认交接帧直出；需换姿态/脸部时用角色卡重绘（不 Scene1 回锚）
        # - 旧 scene1 模式：Scene1 T2I 兼身份参考
        if i == 0:
            if character_lock_enabled() and identity_ref_path and os.path.exists(identity_ref_path):
                edit_instruction = pick_identity_edit_instruction(
                    keyframe_prompt=keyframe_prompt,
                    action_hint=shot_text,
                    visual_anchor=enrich_visual_anchor(visual_anchor or "", user_topic),
                    shot_index=0,
                )
                edited = call_qwen_identity_edit_api(
                    reference_image_path=identity_ref_path,
                    edit_prompt=edit_instruction,
                    output_path=keyframe_path,
                )
                if edited:
                    scene_image = Image.open(keyframe_path).convert("RGB")
                    model_prompts[-1]["keyframe_source"] = "character_asset_scene1"
                    print("   🧬 Scene1 首帧 = 角色素材库锁定后落到开场姿态")
                else:
                    call_wanx_generate_api(prompt=keyframe_prompt, output_path=keyframe_path)
                    scene_image = Image.open(keyframe_path).convert("RGB")
                    model_prompts[-1]["keyframe_source"] = "wanx_scene1_fallback"
                    print("   ⚠️ 角色编辑失败，Scene1 回退万相 T2I（身份仍以素材库为准）")
            else:
                call_wanx_generate_api(prompt=keyframe_prompt, output_path=keyframe_path)
                scene_image = Image.open(keyframe_path).convert("RGB")
                identity_reference = scene_image.copy()
                identity_ref_path = os.path.join(output_dir, "identity_reference.png")
                identity_reference.save(identity_ref_path)
                model_prompts[-1]["keyframe_source"] = "wanx"
                print(f"   🧷 已保存身份参考帧（Scene1 首帧）: {identity_ref_path}")
        else:
            edit_base, base_tag, blend_used = prepare_next_keyframe_base(
                mode=CONTINUITY_MODE,
                handoff_frame=prev_handoff_frame,
                identity_reference=identity_reference,
                shot_index=i,
            )
            model_prompts[-1]["identity_blend_alpha"] = round(blend_used, 3)
            model_prompts[-1]["edit_base"] = base_tag

            # 素材库模式：不做 Scene1 周期性回锚。
            # 仅在「行进→需见脸」等构图突变时用角色卡；高潮入画镜优先交接帧，避免硬切突兀。
            should_asset_lock = False
            climax_shot = action_mentions_climax_object(shot_text, user_topic) or is_climax_enter_action(
                shot_text, user_topic
            )
            lookback_shot = is_lookback_beat_action(shot_text)
            if character_lock_enabled():
                force_loco_to_still = (
                    LOCO_TO_STILL_REANCHOR
                    and prev_shot_was_locomotion
                    and not is_locomotion_action(shot_text)
                    and not rear_pushin
                    and not prefers_rear_pushin(shot_text, prev_was_locomotion=True)
                    and not climax_shot
                    and identity_ref_path
                    and os.path.exists(identity_ref_path)
                )
                # 见脸微动：只在「首次见脸定妆」时角色卡一次；之后交接帧，禁止每镜换脸
                force_face_lock = (
                    prefers_face_micro_motion(shot_text, user_topic)
                    and not climax_shot
                    and not prev_shot_was_face_readable
                    and identity_ref_path
                    and os.path.exists(identity_ref_path)
                )
                if is_locomotion_action(shot_text) and LOCO_SKIP_ASSET_LOCK:
                    should_asset_lock = False
                    print("   🎥 行进镜：交接帧直出，不角色卡重绘（故事优先敢动）")
                elif climax_shot and not lookback_shot:
                    should_asset_lock = False
                    print("   🎥 高潮物件镜：优先交接帧衔接，物件靠 I2V 进入画面")
                elif rear_pushin:
                    should_asset_lock = False
                    print("   🎥 背影推进：交接帧直出 + I2V 推镜")
                elif force_loco_to_still or force_face_lock:
                    should_asset_lock = True
                    if lookback_shot or prefers_face_micro_motion(shot_text, user_topic):
                        print("   🧬 见脸硬切：角色卡过肩/侧脸定妆，I2V 只做微动")
                    elif force_face_lock:
                        print("   🧬 首次需见脸：用角色素材库重绘为过肩或侧脸可读")
                    else:
                        print(
                            "   🧬 行进→非行进：用角色素材库重绘姿态，"
                            "避免走路交接帧直接当本镜起点"
                        )
                should_reanchor = False
            else:
                should_reanchor = (
                    IDENTITY_REANCHOR_EVERY > 0
                    and i > 0
                    and i % IDENTITY_REANCHOR_EVERY == 0
                    and identity_ref_path
                    and os.path.exists(identity_ref_path)
                )
                force_loco_to_still = (
                    LOCO_TO_STILL_REANCHOR
                    and prev_shot_was_locomotion
                    and not is_locomotion_action(shot_text)
                    and not rear_pushin
                    and not prefers_rear_pushin(shot_text, prev_was_locomotion=True)
                    and not climax_shot
                    and identity_ref_path
                    and os.path.exists(identity_ref_path)
                )
                force_face_reanchor = (
                    prefers_face_micro_motion(shot_text, user_topic)
                    and not climax_shot
                    and not prev_shot_was_face_readable
                    and identity_ref_path
                    and os.path.exists(identity_ref_path)
                )
                if is_locomotion_action(shot_text) and LOCO_SKIP_ASSET_LOCK:
                    should_reanchor = False
                    print("   🎥 行进镜：交接帧直出，不身份重锚（故事优先敢动）")
                elif climax_shot and not lookback_shot:
                    should_reanchor = False
                    print("   🎥 高潮物件镜：优先交接帧衔接，物件靠 I2V 进入画面")
                elif rear_pushin:
                    should_reanchor = False
                    print("   🎥 背影推进：交接帧直出 + I2V 推镜")
                elif force_loco_to_still or force_face_reanchor:
                    should_reanchor = True
                    if lookback_shot:
                        print("   🔄 回头高潮：身份重锚为过肩/侧脸，稳定面部")
                    elif force_face_reanchor:
                        print("   🔄 首次需见脸：强制身份重锚为过肩或侧脸可读构图")
                    else:
                        print(
                            "   🔄 行进→非行进过渡：强制身份重锚，"
                            "避免把走路交接帧直接当本镜姿态起点"
                        )
                should_asset_lock = False

            use_direct_handoff = (
                CONTINUITY_MODE in {"hybrid", "handoff_only"}
                and not HYBRID_POSE_EDIT
                and not should_reanchor
                and not should_asset_lock
                and edit_base is not None
            )

            if should_asset_lock or should_reanchor:
                edit_instruction = pick_identity_edit_instruction(
                    keyframe_prompt=keyframe_prompt,
                    action_hint=shot_text,
                    visual_anchor=enrich_visual_anchor(visual_anchor or "", user_topic),
                    shot_index=i,
                )
                edited = call_qwen_identity_edit_api(
                    reference_image_path=identity_ref_path,
                    edit_prompt=edit_instruction,
                    output_path=keyframe_path,
                )
                if edited:
                    scene_image = Image.open(keyframe_path).convert("RGB")
                    model_prompts[-1]["keyframe_source"] = (
                        "character_asset_lock" if should_asset_lock else "identity_reanchor"
                    )
                    model_prompts[-1]["edit_base"] = (
                        "character_asset" if should_asset_lock else "scene1_reanchor"
                    )
                    if should_asset_lock:
                        print("   🧬 本镜首帧 = 角色素材库锁定重绘")
                    else:
                        print(
                            f"   🔄 长片身份重锚（每 {IDENTITY_REANCHOR_EVERY} 镜）："
                            f"以 Scene1 锁脸/衣着，按本镜动作重绘姿态"
                        )
                elif edit_base is not None:
                    scene_image = edit_base.convert("RGB")
                    scene_image.save(keyframe_path)
                    model_prompts[-1]["keyframe_source"] = f"direct_{base_tag}_lock_fallback"
                    print("   ⚠️ 身份锁定失败，本镜回退交接帧直出")
                else:
                    call_wanx_generate_api(prompt=keyframe_prompt, output_path=keyframe_path)
                    scene_image = Image.open(keyframe_path).convert("RGB")
                    model_prompts[-1]["keyframe_source"] = "wanx_lock_fallback"
                    print("   ⚠️ 身份锁定失败，回退独立 T2I")
            elif use_direct_handoff:
                scene_image = edit_base.convert("RGB")
                scene_image.save(keyframe_path)
                model_prompts[-1]["keyframe_source"] = f"direct_{base_tag}"
                print(
                    f"   🧷 本镜首帧 = 上镜交接帧直出（{base_tag}）；"
                    f"姿态变化交给 I2V，成片硬切"
                )
            else:
                edited = None
                if edit_base is not None:
                    edit_base.save(edit_base_path)
                    if CONTINUITY_MODE == "identity_lock":
                        edit_instruction = pick_identity_edit_instruction(
                            keyframe_prompt=keyframe_prompt,
                            action_hint=shot_text,
                            visual_anchor=enrich_visual_anchor(visual_anchor or "", user_topic),
                            shot_index=i,
                        )
                    else:
                        edit_instruction = build_hybrid_edit_instruction(
                            action_hint=shot_text,
                            visual_anchor=enrich_visual_anchor(visual_anchor or "", user_topic),
                            shot_index=i,
                            used_identity_blend=blend_used > 0.01,
                        )
                    edited = call_qwen_identity_edit_api(
                        reference_image_path=(
                            identity_ref_path
                            if CONTINUITY_MODE == "identity_lock" and identity_ref_path
                            else edit_base_path
                        ),
                        edit_prompt=edit_instruction,
                        output_path=keyframe_path,
                    )

                if edited:
                    scene_image = Image.open(keyframe_path).convert("RGB")
                    model_prompts[-1]["keyframe_source"] = f"qwen_edit_{base_tag}"
                    print(
                        f"   🪪 本镜首帧已编辑生成（底图={base_tag}, blend={blend_used:.2f}；"
                        f"成片仍硬切）"
                    )
                else:
                    edited = None
                    if (
                        CONTINUITY_MODE != "identity_lock"
                        and identity_ref_path
                        and os.path.exists(identity_ref_path)
                    ):
                        edit_instruction = pick_identity_edit_instruction(
                            keyframe_prompt=keyframe_prompt,
                            action_hint=shot_text,
                            visual_anchor=enrich_visual_anchor(visual_anchor or "", user_topic),
                            shot_index=i,
                        )
                        edited = call_qwen_identity_edit_api(
                            reference_image_path=identity_ref_path,
                            edit_prompt=edit_instruction,
                            output_path=keyframe_path,
                        )
                    if edited:
                        scene_image = Image.open(keyframe_path).convert("RGB")
                        model_prompts[-1]["keyframe_source"] = "qwen_edit_identity_fallback"
                        print("   🪪 交接帧路径失败，已回退角色/身份素材编辑")
                    else:
                        call_wanx_generate_api(prompt=keyframe_prompt, output_path=keyframe_path)
                        scene_image = Image.open(keyframe_path).convert("RGB")
                        model_prompts[-1]["keyframe_source"] = "wanx_per_scene_fallback"
                        print("   🎨 身份/交接帧路径失败，回退本镜独立 T2I 文字锁")

        scene_image, height, width = prepare_scene_image(scene_image, image_pipe, max_area)
        scene_image.save(keyframe_path)
        model_prompts[-1]["width"] = width
        model_prompts[-1]["height"] = height

        print(f"   📐 本镜分辨率: {width}x{height}")

        scene_seed = resolve_scene_seed(visual_anchor, i)
        model_prompts[-1]["seed"] = scene_seed
        print(f"   🎲 seed={scene_seed}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        is_loco_shot = is_locomotion_action(shot_text)
        is_reveal_shot = prefers_reveal_pushin(shot_text, user_topic) or is_climax_enter_action(
            shot_text, user_topic
        )
        shot_flow_shift = resolve_shot_flow_shift(
            is_locomotion=is_loco_shot,
            is_reveal=is_reveal_shot,
            rear_pushin=rear_pushin,
        )
        shot_guidance, shot_guidance_2 = resolve_shot_guidance(
            is_locomotion=is_loco_shot,
            is_reveal=is_reveal_shot,
            rear_pushin=rear_pushin,
        )
        apply_wan_flow_shift(image_pipe, shot_flow_shift)
        if is_loco_shot or is_reveal_shot or rear_pushin:
            print(
                f"   🎚️ 强动采样: flow_shift={shot_flow_shift:.1f} "
                f"guidance={shot_guidance:.1f}/{shot_guidance_2:.1f}"
                f"{' (入画/推镜)' if (is_reveal_shot or rear_pushin) and not is_loco_shot else ''}"
            )
        shot_frames = NUM_FRAMES
        if is_loco_shot and LOCO_NUM_FRAMES > 0:
            shot_frames = max(17, LOCO_NUM_FRAMES)
            shot_frames = shot_frames - ((shot_frames - 1) % 4)
            if shot_frames != NUM_FRAMES:
                print(f"   ⏱️ 行进镜帧数: {shot_frames}（默认镜 {NUM_FRAMES}）")

        # 观感慢动作多半是「只动腿不前进」；强制尺度变小=真走远
        loco_boost_extra = (
            "，五秒内约迈八到十步并明显走远，"
            "主体在画面中明显变小，禁止原地迈腿"
            if is_loco_shot
            else ""
        )
        run_prompt = enhanced_prompt
        if is_loco_shot and loco_boost_extra not in run_prompt:
            if run_prompt.endswith("。"):
                run_prompt = run_prompt[:-1] + loco_boost_extra + "。"
            else:
                run_prompt = run_prompt + loco_boost_extra

        # 根因靠 flow_shift/guidance；重试默认关闭
        best = None
        attempts = 1 + max(0, MOTION_RETRY if is_loco_shot else 0)

        for attempt in range(attempts):
            attempt_seed = scene_seed + attempt * 9973
            generator = torch.Generator(device="cpu").manual_seed(attempt_seed)
            attempt_prompt = run_prompt
            if attempt > 0:
                print(
                    f"   🔁 运动不足重试 {attempt}/{attempts - 1} "
                    f"(seed={attempt_seed})，可选兜底（默认已关）"
                )
                if is_loco_shot:
                    extra = "，每秒约两步，多迈几步，移到画面更远处"
                    if attempt_prompt.endswith("。"):
                        attempt_prompt = attempt_prompt[:-1] + extra + "。"
                    else:
                        attempt_prompt = attempt_prompt + extra

            pipe_kwargs = dict(
                image=scene_image,
                prompt=attempt_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_frames=shot_frames,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=shot_guidance,
                generator=generator,
            )
            # Wan2.2 MoE：低噪声专家独立 guidance
            if getattr(image_pipe, "transformer_2", None) is not None:
                pipe_kwargs["guidance_scale_2"] = shot_guidance_2

            try:
                frames = image_pipe(
                    **pipe_kwargs,
                    callback_on_step_end=_free_cache_before_decode,
                ).frames[0]
            except TypeError:
                # 旧 diffusers 可能不认 guidance_scale_2
                pipe_kwargs.pop("guidance_scale_2", None)
                try:
                    frames = image_pipe(
                        **pipe_kwargs,
                        callback_on_step_end=_free_cache_before_decode,
                    ).frames[0]
                except TypeError:
                    frames = image_pipe(**pipe_kwargs).frames[0]

            frame_list = normalize_frame_list(frames)
            handoff_frame, handoff_idx, motion_delta = select_continuity_frame(frame_list)
            mean_diff = clip_mean_consecutive_delta(frame_list)
            print(
                f"   📊 镜头 {scene_no} 尝试{attempt + 1}: "
                f"handoff_delta={motion_delta:.4f}, "
                f"mean_frame_diff={mean_diff:.4f}"
                f"{' (行进门槛 mean≥%.3f 或 delta≥%.2f)' % (LOCO_MIN_FRAME_DIFF, LOCO_MIN_MOTION_DELTA) if is_loco_shot else ''}"
            )

            if is_loco_shot:
                # 通过条件（全局）：
                # 1) 首尾位移够大（人确实走到更远处），或
                # 2) 相邻帧差达可读行进，且位移也不至于原地踏步
                score = max(mean_diff, motion_delta * 0.05)
                ok_motion = (
                    motion_delta >= LOCO_MIN_MOTION_DELTA
                    or (
                        mean_diff >= LOCO_MIN_FRAME_DIFF
                        and motion_delta >= LOCO_MIN_MOTION_DELTA * 0.75
                    )
                )
            else:
                score = motion_delta
                ok_motion = motion_delta >= MIN_MOTION_FRAME_DELTA

            candidate = [
                score,
                mean_diff,
                motion_delta,
                frame_list,
                handoff_frame,
                handoff_idx,
                attempt_seed,
                attempt_prompt,
            ]
            if best is None or score > best[0]:
                if best is not None:
                    # 释放落败的旧帧列表，避免重试时显存/内存堆叠
                    old_frames = best[3]
                    best[3] = None
                    del old_frames
                best = candidate
            elif frame_list is not None:
                del frame_list

            del frames
            del generator
            if ok_motion:
                break
            if attempt + 1 < attempts:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        assert best is not None
        (
            _score,
            mean_diff,
            motion_delta,
            frame_list,
            handoff_frame,
            handoff_idx,
            used_seed,
            used_prompt,
        ) = best
        best = None
        model_prompts[-1]["seed"] = used_seed
        model_prompts[-1]["prompt"] = used_prompt
        model_prompts[-1]["num_frames"] = shot_frames
        model_prompts[-1]["flow_shift"] = shot_flow_shift
        model_prompts[-1]["guidance_scale"] = shot_guidance
        model_prompts[-1]["guidance_scale_2"] = shot_guidance_2
        model_prompts[-1]["mean_frame_diff"] = round(mean_diff, 5)
        if is_loco_shot and not (
            motion_delta >= LOCO_MIN_MOTION_DELTA
            or (
                mean_diff >= LOCO_MIN_FRAME_DIFF
                and motion_delta >= LOCO_MIN_MOTION_DELTA * 0.75
            )
        ):
            print(
                f"   ⚠️ 镜头 {scene_no} 行进仍偏弱 "
                f"(mean_frame_diff={mean_diff:.4f}, delta={motion_delta:.4f}；"
                f"门槛 mean≥{LOCO_MIN_FRAME_DIFF} 或 delta≥{LOCO_MIN_MOTION_DELTA})，"
                "已取重试中最强的一次"
            )

        # 保存本镜交接帧，供下一镜 hybrid 使用；成片拼接仍硬切
        handoff_frame.save(handoff_path)
        prev_handoff_frame = handoff_frame.copy()
        prev_handoff_index = handoff_idx

        model_prompts[-1]["motion_delta"] = round(motion_delta, 5)
        model_prompts[-1]["handoff_frame_index"] = handoff_idx
        with open(model_prompts_path, "w", encoding="utf-8") as f:
            json.dump(model_prompts, f, ensure_ascii=False, indent=2)

        export_to_video(frame_list, filename, fps=SOURCE_FPS)
        _postprocess_clip(filename)
        print(f"✨ 镜头 {scene_no} 渲染完毕！已保存至: {filename}")
        print(f"   🧷 交接帧已保存: {handoff_path} (frame_index={handoff_idx})")

        prev_shot_was_locomotion = is_locomotion_action(shot_text)
        prev_shot_was_face_readable = requires_face_readable_action(
            shot_text
        ) or prefers_face_micro_motion(shot_text, user_topic)

        del frame_list
        del scene_image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print(
        f"\n🎉 所有分镜头已处理完毕！"
        f"（CHARACTER_LOCK={CHARACTER_LOCK_MODE}, CONTINUITY_MODE={CONTINUITY_MODE}, "
        f"HYBRID_POSE_EDIT={int(HYBRID_POSE_EDIT)}，成片硬切）"
    )
    unload_i2v_pipeline(image_pipe)


if __name__ == "__main__":
    # 通用示例：不写死任何主题，仅用于本地直接跑 pipeline 联调。
    demo_prompts = [
        "subject in scene. He walks forward naturally.",
        "He turns head slowly to the right.",
    ]
    demo_keyframes = [
        "Medium shot, clear subject in described setting, feet visible.",
        "Medium shot, subject turning head right, scene background.",
    ]

    render_video_clips(
        demo_prompts,
        keyframe_prompts=demo_keyframes,
        user_topic="赛博朋克雨夜，赏金猎人撑伞走入暗巷",
    )

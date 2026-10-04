#!/bin/bash
# AutoDL 启动 Celery Worker（Wan2.2-I2V-A14B / 24GB 显存稳定配置）
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 24GB 必须 group/sequential offload；满血无 offload 约需 80GB
# 首次请先: bash download_wan_model.sh
# 并安装: pip install -U "git+https://github.com/huggingface/diffusers"
export WAN_I2V_PROFILE=720p
export WAN_OFFLOAD=group
export WAN_FP8=1
export WAN_LOW_VRAM=1
export WAN_AREA_SCALE=0.48
# 81 帧 @16fps ≈ 5 秒/镜；12 镜 ≈ 60 秒成片
export WAN_NUM_FRAMES=81
# 故事优先采样：行进敢动；非行进求稳
export WAN_FLOW_SHIFT="${WAN_FLOW_SHIFT:-5.0}"
export WAN_LOCO_FLOW_SHIFT="${WAN_LOCO_FLOW_SHIFT:-8.5}"
export WAN_GUIDANCE_SCALE="${WAN_GUIDANCE_SCALE:-3.5}"
export WAN_GUIDANCE_SCALE_2="${WAN_GUIDANCE_SCALE_2:-3.5}"
export WAN_LOCO_GUIDANCE_SCALE="${WAN_LOCO_GUIDANCE_SCALE:-4.5}"
export WAN_LOCO_GUIDANCE_SCALE_2="${WAN_LOCO_GUIDANCE_SCALE_2:-4.5}"
export LOCO_SKIP_ASSET_LOCK="${LOCO_SKIP_ASSET_LOCK:-1}"
# 动态靠 shift/CFG，默认不重试
export WAN_LOCO_NUM_FRAMES="${WAN_LOCO_NUM_FRAMES:-81}"
export WAN_MOTION_RETRY="${WAN_MOTION_RETRY:-0}"
export LOCO_MIN_FRAME_DIFF="${LOCO_MIN_FRAME_DIFF:-0.012}"
export LOCO_MIN_MOTION_DELTA="${LOCO_MIN_MOTION_DELTA:-0.18}"
export WAN_INFERENCE_STEPS=30
export VAE_TILING=1
export CELERY_POOL=solo
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WAN_DROP_PAGE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# 故事优先：宁短勿注水（偏好约 35s / 最多 8 镜）
export SHIPPABLE_MODE="${SHIPPABLE_MODE:-1}"
export MAX_TARGET_SECONDS="${MAX_TARGET_SECONDS:-60}"
export TARGET_VIDEO_SECONDS="${TARGET_VIDEO_SECONDS:-35}"
export CLIP_SECONDS=5
export ABS_MAX_BEATS="${ABS_MAX_BEATS:-8}"
export MAX_CONSECUTIVE_LOCOMOTION="${MAX_CONSECUTIVE_LOCOMOTION:-1}"
export MAX_CONSECUTIVE_WALKING_BEATS="${MAX_CONSECUTIVE_WALKING_BEATS:-1}"
export CROSSFADE_DURATION=0
export TARGET_FPS=16
# LLM：中文分镜直出（不再强制英文化）
export QWEN_MODEL="${QWEN_MODEL:-qwen-max}"
export CONTINUITY_MODE="${CONTINUITY_MODE:-hybrid}"
export CHARACTER_LOCK_MODE="${CHARACTER_LOCK_MODE:-asset}"
export CHARACTER_ASSET_DIR="${CHARACTER_ASSET_DIR:-/root/autodl-tmp/character_assets}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/output_clips}"
export FINAL_VIDEO_DIR="${FINAL_VIDEO_DIR:-/root/autodl-tmp/final_videos}"
mkdir -p "$OUTPUT_DIR" "$FINAL_VIDEO_DIR" "$CHARACTER_ASSET_DIR"
export HYBRID_POSE_EDIT="${HYBRID_POSE_EDIT:-0}"
export HANDOFF_FRAME_RATIO="${HANDOFF_FRAME_RATIO:-0.88}"
export HANDOFF_IDENTITY_BLEND="${HANDOFF_IDENTITY_BLEND:-0}"
export IDENTITY_BLEND_ALPHA=0
export HANDOFF_BLEND_MAX_DELTA="${HANDOFF_BLEND_MAX_DELTA:-0.18}"
# 素材库锁人后关闭 Scene1 周期性回锚
export IDENTITY_REANCHOR_EVERY="${IDENTITY_REANCHOR_EVERY:-0}"
export IDENTITY_REANCHOR_ALPHA="${IDENTITY_REANCHOR_ALPHA:-0.28}"
# 行进→停步 / 脸部动作：用角色素材库重绘（不是 Scene1 构图回锚）
export LOCO_TO_STILL_REANCHOR="${LOCO_TO_STILL_REANCHOR:-1}"
export WAN_BASE_SEED=42
export WAN_FIXED_SEED=0

echo ">>> 角色素材库: CHARACTER_LOCK_MODE=$CHARACTER_LOCK_MODE DIR=$CHARACTER_ASSET_DIR"
mkdir -p "$CHARACTER_ASSET_DIR/manual" "$CHARACTER_ASSET_DIR/generated"

echo ">>> 检查容器/系统内存..."
free -h
if [ -f /sys/fs/cgroup/memory.max ]; then
  echo ">>> 容器内存限额: $(cat /sys/fs/cgroup/memory.max) bytes"
  echo ">>> 容器当前用量: $(cat /sys/fs/cgroup/memory.current) bytes"
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  echo ">>> 容器内存限额: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) bytes"
  echo ">>> 容器当前用量: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) bytes"
fi

echo ">>> 检查 Wan2.2 模型目录..."
WAN22_DIR="/root/autodl-tmp/models/Wan2.2-I2V-A14B-Diffusers"
if [ ! -f "$WAN22_DIR/model_index.json" ] && [ ! -f "$WAN22_DIR/transformer/config.json" ] && [ ! -f "$WAN22_DIR/transformer_2/config.json" ]; then
  echo "⚠️ 未找到 $WAN22_DIR ，请先: bash download_wan_model.sh"
fi

echo ">>> 检查 NumPy / Torch / Diffusers..."
python - <<'PY'
import numpy as np
import torch
print(f"NumPy {np.__version__} @ {np.__file__}")
print(f"Torch {torch.__version__}")
_orig = torch.from_numpy
def _safe(a):
    try:
        return _orig(a)
    except TypeError:
        return torch.tensor(__import__('numpy').asarray(a).copy())
torch.from_numpy = _safe
print(torch.from_numpy(np.zeros(4, dtype=np.float64)))
print("NumPy↔Torch OK")
try:
    from diffusers import WanImageToVideoPipeline
    import diffusers
    print(f"diffusers {getattr(diffusers, '__version__', '?')} WanImageToVideoPipeline OK")
except Exception as exc:
    print("⚠️ Diffusers/Wan 导入失败，请执行:")
    print('  pip install -U "git+https://github.com/huggingface/diffusers"')
    raise SystemExit(f"diffusers import failed: {exc}")
PY

echo ">>> 启动 Celery Worker (Wan2.2-A14B | solo | max~${MAX_TARGET_SECONDS}s | CONTINUITY=$CONTINUITY_MODE | QWEN_MODEL=$QWEN_MODEL)..."
exec celery -A celery_worker worker -P solo --loglevel=info --concurrency=1

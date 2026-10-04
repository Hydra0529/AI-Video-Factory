#!/bin/bash
# 卸载旧版 CogVideoX 模型，释放数据盘 / HuggingFace 缓存空间
# 用法:
#   bash remove_cogvideo_model.sh          # 交互确认
#   bash remove_cogvideo_model.sh -f       # 直接删除，不询问

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "-f" || "${1:-}" == "--force" ]]; then
  FORCE=1
fi

# 数据盘上的本地模型目录（原 video_generator.py 使用）
DATA_DISK_MODELS=(
  "/root/autodl-tmp/models/CogVideoX1.5-5B-I2V"
  "/root/autodl-tmp/models/CogVideoX-5B-I2V"
  "/root/autodl-tmp/models/CogVideoX1.5-5B-I2V-Diffusers"
)

# HuggingFace 缓存目录名（hub 格式）
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_CACHE_DIRS=(
  "models--THUDM--CogVideoX1.5-5B-I2V"
  "models--THUDM--CogVideoX-5B-I2V"
  "models--THUDM--CogVideoX1.5-5b-I2V"
)

human_size() {
  du -sh "$1" 2>/dev/null | awk '{print $1}'
}

confirm_delete() {
  local path="$1"
  if [[ $FORCE -eq 1 ]]; then
    return 0
  fi
  read -r -p "确认删除 $path ? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]]
}

remove_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "⏭️  跳过（不存在）: $path"
    return 0
  fi

  local size
  size="$(human_size "$path")"
  echo "📦 发现: $path （约 $size）"

  if confirm_delete "$path"; then
    rm -rf "$path"
    echo "✅ 已删除: $path"
  else
    echo "❌ 已取消: $path"
  fi
}

echo "=========================================="
echo " 卸载 CogVideoX 旧模型"
echo " 当前视频流水线已切换为 Wan2.1-I2V-14B"
echo "=========================================="
echo ""

echo ">>> [1/2] 检查数据盘模型..."
for path in "${DATA_DISK_MODELS[@]}"; do
  remove_path "$path"
done

echo ""
echo ">>> [2/2] 检查 HuggingFace 缓存..."
for name in "${HF_CACHE_DIRS[@]}"; do
  remove_path "$HF_CACHE_ROOT/$name"
done

echo ""
echo ">>> 数据盘剩余空间:"
df -h /root/autodl-tmp 2>/dev/null || df -h .

echo ""
echo "完成。Wan 模型目录保留在:"
echo "  /root/autodl-tmp/models/Wan2.1-I2V-14B-720P-Diffusers"
echo ""
echo "若尚未下载 Wan，请执行: bash download_wan_model.sh"

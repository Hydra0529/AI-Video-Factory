#!/bin/bash
# AutoDL 数据盘下载 Wan2.2-I2V-A14B-Diffusers（避免占满系统盘）
# 用法:
#   bash download_wan_model.sh           # 默认 720p 档（同一权重）
#   bash download_wan_model.sh 480p      # 仅改运行时 max_area 提示，权重相同
#
# 依赖提醒：Wan2.2 Diffusers 需较新 diffusers，建议：
#   pip install -U "git+https://github.com/huggingface/diffusers"

set -euo pipefail

PROFILE="${1:-720p}"
MODEL="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TARGET="/root/autodl-tmp/models/Wan2.2-I2V-A14B-Diffusers"

case "$PROFILE" in
  720p|720P|480p|480P)
    ;;
  *)
    echo "用法: bash download_wan_model.sh [720p|480p]"
    exit 1
    ;;
esac

# 缓存也放到数据盘，防止系统盘爆满
export HF_HOME="/root/autodl-tmp/huggingface"
export HF_HUB_CACHE="/root/autodl-tmp/huggingface/hub"
export MODELSCOPE_CACHE="/root/autodl-tmp/modelscope"

mkdir -p "$(dirname "$TARGET")" "$HF_HOME" "$MODELSCOPE_CACHE"

echo "=========================================="
echo " 模型: $MODEL"
echo " 目标: $TARGET"
echo " 运行档位提示: $PROFILE（同一权重，靠 WAN_I2V_PROFILE / max_area）"
echo " 数据盘剩余:"
df -h /root/autodl-tmp | tail -1
echo "=========================================="

if [[ -f "$TARGET/model_index.json" || -f "$TARGET/transformer/config.json" || -f "$TARGET/transformer_2/config.json" ]]; then
  echo "⚠️ 目标目录已有模型文件，跳过下载。"
  echo "   若要重新下载请先: rm -rf \"$TARGET\""
  exit 0
fi

# AutoDL 国内优先 ModelScope，其次 HuggingFace
if command -v modelscope &>/dev/null; then
  echo ">>> 使用 ModelScope 下载到数据盘..."
  modelscope download "$MODEL" --local_dir "$TARGET" || \
    modelscope download --model "$MODEL" --local_dir "$TARGET"
elif command -v huggingface-cli &>/dev/null; then
  echo ">>> 使用 HuggingFace CLI 下载到数据盘..."
  huggingface-cli download "$MODEL" --local-dir "$TARGET"
else
  echo ">>> 安装 huggingface_hub / modelscope 后下载..."
  pip install -U "huggingface_hub[cli]" modelscope -q
  if modelscope download "$MODEL" --local_dir "$TARGET" 2>/dev/null || \
     modelscope download --model "$MODEL" --local_dir "$TARGET"; then
    :
  else
    huggingface-cli download "$MODEL" --local-dir "$TARGET"
  fi
fi

echo ""
echo "✅ 下载完成: $TARGET"
du -sh "$TARGET"
echo ""
echo "启动前设置："
echo "  export WAN_I2V_PROFILE=$PROFILE"
echo "  export HF_HOME=/root/autodl-tmp/huggingface"
echo "  pip install -U \"git+https://github.com/huggingface/diffusers\""
echo "  bash start_worker.sh"

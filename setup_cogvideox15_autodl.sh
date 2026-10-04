#!/usr/bin/env bash
# AutoDL 一键：卸载旧 CogVideoX-5b-I2V，安装 CogVideoX1.5-5B-I2V
set -euo pipefail

OLD_PATHS=(
  "/root/.cache/huggingface/hub/ZhipuAI/CogVideoX-5b-I2V"
  "/root/.cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V"
  "/root/.cache/huggingface/hub/models--ZhipuAI--CogVideoX-5b-I2V"
  "/root/autodl-tmp/models/CogVideoX-5b-I2V"
)

NEW_MODEL_DIR="/root/autodl-tmp/models/CogVideoX1.5-5B-I2V"
MODEL_ID="THUDM/CogVideoX1.5-5B-I2V"

echo "=========================================="
echo "1/4 升级 diffusers 等依赖（CogVideoX1.5 需要较新版本）"
echo "=========================================="
pip install -U pip
pip install -U "diffusers>=0.32.0" "transformers>=4.46.0" "accelerate>=1.1.0" huggingface_hub

echo ""
echo "=========================================="
echo "2/4 删除旧模型 CogVideoX-5b-I2V"
echo "=========================================="
for path in "${OLD_PATHS[@]}"; do
  if [ -e "$path" ]; then
    echo "删除: $path"
    rm -rf "$path"
  else
    echo "跳过（不存在）: $path"
  fi
done

echo ""
echo "=========================================="
echo "3/4 下载新模型 CogVideoX1.5-5B-I2V（约 40GB，请耐心等待）"
echo "=========================================="
mkdir -p "$(dirname "$NEW_MODEL_DIR")"

# 国内 AutoDL 可取消下一行注释使用镜像
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download "$MODEL_ID" --local-dir "$NEW_MODEL_DIR"

echo ""
echo "=========================================="
echo "4/4 写入环境变量（供 video_generator.py 读取）"
echo "=========================================="
ENV_FILE="/root/autodl-tmp/cogvideo_env.sh"
cat > "$ENV_FILE" <<EOF
export I2V_MODEL_PATH="$NEW_MODEL_DIR"
export I2V_LOCAL_FILES_ONLY=true
export VIDEO_RESOLUTION=1360x768
EOF

echo "已写入 $ENV_FILE"
echo ""
echo "完成！每次启动 Celery 前先执行："
echo "  source $ENV_FILE"
echo "  celery -A celery_worker worker --loglevel=info --pool=solo"
echo ""
echo "新模型路径: $NEW_MODEL_DIR"

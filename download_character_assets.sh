#!/bin/bash
# 下载免费角色定妆素材到数据盘（无需 API Key）
# 来源：randomuser.me 合成人像（免费、可本地缓存作身份参考）
#
# 用法（AutoDL）:
#   bash download_character_assets.sh
#
# 下载后目录:
#   /root/autodl-tmp/character_assets/manual/
#   default.png 会被设为其中一张，流水线自动优先使用

set -euo pipefail

TARGET="${CHARACTER_ASSET_DIR:-/root/autodl-tmp/character_assets}"
MANUAL="$TARGET/manual"
GEN="$TARGET/generated"

mkdir -p "$MANUAL" "$GEN"
cd "$MANUAL"

echo "=========================================="
echo " 免费角色素材库 → $MANUAL"
echo " 数据盘剩余:"
df -h /root/autodl-tmp 2>/dev/null | tail -1 || df -h . | tail -1
echo "=========================================="

download_one() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "  skip $out"
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 1 -o "$out" "$url" || return 1
  else
    wget -q -O "$out" "$url" || return 1
  fi
  # 校验不是空/HTML 错误页
  local size
  size=$(wc -c < "$out" | tr -d ' ')
  if [[ "$size" -lt 2000 ]]; then
    rm -f "$out"
    echo "  fail $out (too small)"
    return 1
  fi
  echo "  ok   $out (${size} bytes)"
}

# 固定一批人像编号，保证可复现（非真人照片，合成脸）
MEN=(12 25 32 45 56 67 75 82 91)
WOMEN=(8 15 28 36 44 53 61 74 88)

echo ">>> 下载男性参考..."
i=1
for n in "${MEN[@]}"; do
  download_one "https://randomuser.me/api/portraits/men/${n}.jpg" "char_male_$(printf '%02d' "$i").jpg" || true
  i=$((i + 1))
done

echo ">>> 下载女性参考..."
i=1
for n in "${WOMEN[@]}"; do
  download_one "https://randomuser.me/api/portraits/women/${n}.jpg" "char_female_$(printf '%02d' "$i").jpg" || true
  i=$((i + 1))
done

# 默认身份图：优先 male_01，没有则取第一张 jpg
if [[ -f char_male_01.jpg ]]; then
  cp -f char_male_01.jpg default.png 2>/dev/null || cp -f char_male_01.jpg default.jpg
  # 若系统有 convert/python 可转 png；没有就用 jpg 当 default
  if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    PY=python3
    command -v python3 >/dev/null 2>&1 || PY=python
    $PY - <<'PY'
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    raise SystemExit(0)
src = Path("char_male_01.jpg")
if src.exists():
    Image.open(src).convert("RGB").save("default.png", quality=95)
    print("  ok   default.png (from char_male_01.jpg)")
PY
  fi
fi

# 若只有 jpg default
if [[ ! -f default.png && -f char_male_01.jpg ]]; then
  cp -f char_male_01.jpg default.jpg
fi

COUNT=$(ls -1 char_*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "✅ 完成：共 ${COUNT} 张角色参考图"
echo "   目录: $MANUAL"
ls -lah "$MANUAL" | head -30
echo ""
echo "流水线会自动用 manual/default.png（或 default.jpg）。"
echo "指定某一张："
echo "  export CHARACTER_ASSET_PATH=$MANUAL/char_male_03.jpg"
echo "然后: bash start_worker.sh"

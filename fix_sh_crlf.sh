#!/bin/bash
# 在 AutoDL/Linux 上修复 Windows 换行符（CRLF -> LF）
# 用法: bash fix_sh_crlf.sh
set -e

cd "$(dirname "$0")"

for f in *.sh; do
  if [ -f "$f" ]; then
    sed -i 's/\r$//' "$f"
    chmod +x "$f"
    echo "fixed: $f"
  fi
done

echo "完成。现在可执行: bash start_worker.sh"

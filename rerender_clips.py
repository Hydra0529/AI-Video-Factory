"""
从已有 prompts.json 重新渲染视频（跳过 LLM）。
每镜 I2V 起点回锚 Scene1 首帧（硬切），不使用上一镜末帧。

用法:
  python rerender_clips.py
  python rerender_clips.py /root/autodl-tmp/output_clips/prompts.json
"""
import json
import os
import sys

from llm_director import compose_i2v_prompt, normalize_visual_anchor
from video_generator import render_video_clips

DEFAULT_PROMPTS_PATH = "/root/autodl-tmp/output_clips/prompts.json"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/output_clips"


def load_and_recompose(prompts_path: str) -> dict:
    with open(prompts_path, encoding="utf-8") as f:
        data = json.load(f)

    visual_anchor = normalize_visual_anchor(data.get("visual_anchor", ""))
    raw_shots = data.get("raw_shots") or []
    if not raw_shots:
        raise ValueError("prompts.json 中没有 raw_shots，无法重渲染")

    prompts_list = [
        compose_i2v_prompt(visual_anchor, raw_shots[i], i)
        for i in range(len(raw_shots))
    ]

    print(">>> 使用最新后处理规则重组 I2V prompt：")
    for i, prompt in enumerate(prompts_list, start=1):
        print(f"  镜头 {i:02d}: {prompt}")

    data["visual_anchor"] = visual_anchor
    data["prompts"] = prompts_list
    return data


def main():
    prompts_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPTS_PATH
    output_dir = os.path.dirname(os.path.abspath(prompts_path)) or DEFAULT_OUTPUT_DIR

    if not os.path.exists(prompts_path):
        print(f"❌ 找不到 {prompts_path}")
        sys.exit(1)

    data = load_and_recompose(prompts_path)
    render_video_clips(
        data["prompts"],
        output_dir=output_dir,
        visual_anchor=data.get("visual_anchor", ""),
        keyframe_prompts=data.get("keyframe_prompts"),
        raw_shots=data.get("raw_shots"),
        continuity_spine=data.get("continuity_spine"),
        user_topic=data.get("user_topic", ""),
    )
    print("\n✅ 重渲染完成。")


if __name__ == "__main__":
    main()

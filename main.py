import json
import os
from llm_director import generate_video_prompts

OUTPUT_DIR = "/root/autodl-tmp/output_clips"


def run_pipeline(user_topic: str, target_seconds: float | None = None):
    print("=" * 40)
    print("🎬 开始执行 AI 视频生成任务")
    print(f"📝 主题：{user_topic}")
    if target_seconds:
        print(f"⏱️ 用户指定时长：{target_seconds}s")
    print("=" * 40)

    print(f"📝 用户主题长度: {len(user_topic)} 字")
    print("\n>>> [阶段1] 正在呼叫大语言模型，拆解剧本分镜...")
    storyboard = generate_video_prompts(user_topic, target_seconds=target_seconds)
    prompts_list = storyboard.get("prompts", []) if isinstance(storyboard, dict) else storyboard
    visual_anchor = storyboard.get("visual_anchor", "") if isinstance(storyboard, dict) else ""
    keyframe_prompts = storyboard.get("keyframe_prompts", []) if isinstance(storyboard, dict) else []
    raw_shots = storyboard.get("raw_shots", []) if isinstance(storyboard, dict) else []
    continuity_spine = storyboard.get("continuity_spine", []) if isinstance(storyboard, dict) else []
    beat_budget = storyboard.get("beat_budget", {}) if isinstance(storyboard, dict) else {}

    if not prompts_list:
        print("❌ 剧本生成失败，提前结束程序。")
        return False

    print(f"✅ 剧本拆解成功！共生成 {len(prompts_list)} 个分镜。")
    if visual_anchor:
        print(f"🎯 视觉锚点: {visual_anchor}")
    if keyframe_prompts:
        print(f"🖼️ 首帧专用描述已生成 {len(keyframe_prompts)} 条")

    if continuity_spine:
        print(f"📖 故事骨架已生成 {len(continuity_spine)} 个原子动作（一镜一动作）")

    # 出片优先：多样性只警告，不拒渲
    from llm_director import extract_motion_signature, validate_shot_diversity

    ok, reason = validate_shot_diversity(raw_shots or [])
    if not ok:
        print(f"⚠️ 分镜动作重复（出片档继续渲染）：{reason}")
    for i in range(len(prompts_list)):
        action = (raw_shots[i] if raw_shots and i < len(raw_shots) else prompts_list[i])
        sig = extract_motion_signature(action)
        print(f"  ✓ 镜头 {i + 1:02d} 动作指纹: {sig or '(n/a)'} | {action[:80]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompts_path = os.path.join(OUTPUT_DIR, "prompts.json")
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_topic": user_topic,
                "story_topic": storyboard.get("story_topic", user_topic)
                if isinstance(storyboard, dict)
                else user_topic,
                "beat_budget": beat_budget,
                "visual_anchor": visual_anchor,
                "continuity_spine": continuity_spine,
                "raw_shots": raw_shots,
                "prompts": prompts_list,
                "keyframe_prompts": keyframe_prompts,
                "shot_count": len(prompts_list),
                "strategy": "adaptive_beat_budget",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"🧾 本次分镜提示词已保存：{prompts_path}")

    print("\n>>> [阶段2] 唤醒视频生成模型（预计加载需要几分钟）...")
    from video_generator import render_video_clips

    render_video_clips(
        prompts_list,
        output_dir=OUTPUT_DIR,
        visual_anchor=visual_anchor,
        keyframe_prompts=keyframe_prompts,
        raw_shots=raw_shots,
        continuity_spine=continuity_spine,
        user_topic=user_topic,
    )
    print("\n🎉 全部视频素材生成完毕！")
    return True


if __name__ == "__main__":
    topic = "一只穿着宇航服的黄金猎犬在火星表面探索，发现了古代外星遗迹"
    run_pipeline(topic)

#!/usr/bin/env python3
"""
只生成并校验 LLM 分镜 prompt，不加载 Wan 视频模型、不渲染视频。

用法（AutoDL 终端）:
    python validate_prompts.py "赛博朋克雨夜，赏金猎人撑伞走入暗巷"
    python validate_prompts.py "你的主题" -o /root/autodl-tmp/output_clips/prompt_validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from llm_director import (
    MAX_I2V_ACTION_WORDS,
    MAX_I2V_TOTAL_WORDS,
    _BOTH_HANDS_DOWN_RE,
    _STATIC_ACTION_RE,
    action_uses_hand,
    average_pairwise_similarity,
    boost_i2v_motion_prompt,
    generate_video_prompts,
    infer_prop_grip,
    looks_truncated,
    opposite_hand,
    validate_i2v_feasibility,
    validate_motion_clarity,
    validate_motion_sanity,
    validate_shot_diversity,
    validate_spine_safety,
)

DEFAULT_TOPIC = "赛博朋克雨夜，赏金猎人撑伞走入暗巷"
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/root/autodl-tmp/output_clips")

MAX_I2V_PROMPT_WORDS = int(os.getenv("MAX_I2V_PROMPT_WORDS", "64"))
QUALITY_SUFFIX = (
    "同一主体同一外观，背景稳定锁定，"
    "以主体运动为主，主体清晰可见可读，"
    "不对镜头注视，高质量"
)
KEYFRAME_SUFFIX = (
    "主体清晰可辨，与描述一致，"
    "开场姿态准备做首个动作，不是完全静止冻结，"
    "不要发明描述中没有的道具，"
    "写实摄影感，电影光影，清晰对焦。"
)
MAX_KEYFRAME_PROMPT_WORDS = int(os.getenv("MAX_KEYFRAME_PROMPT_WORDS", "112"))


def build_i2v_model_prompt_preview(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt.strip().replace("\n", " ")).strip(" ,.;，。；")
    prompt = re.sub(r"\bsame (?:person same face same outfit|subject same appearance)\.?\s*", "", prompt, flags=re.I)
    prompt = re.sub(r"同一主体同一外观[。.]?\s*", "", prompt).strip(" ,.;，。；")
    use_zh = bool(re.search(r"[\u4e00-\u9fff]", prompt))
    if use_zh:
        parts = [p.strip(" ,.;，。；") for p in re.split(r"[。．]\s*", prompt) if p.strip(" ,.;，。；")]
    else:
        parts = [p.strip(" ,.;") for p in re.split(r"\.\s+", prompt) if p.strip(" ,.;")]
    if len(parts) >= 2:
        identity, action = parts[0], ("。" if use_zh else ". ").join(parts[1:])
    else:
        identity, action = "", parts[0] if parts else prompt
    action = boost_i2v_motion_prompt(action)
    suffix = QUALITY_SUFFIX
    if use_zh:
        body = "。".join(p for p in (action, identity) if p)
        return f"{body}。{suffix}" if body else suffix
    reserved = len(suffix.split()) + len(action.split()) + 2
    budget = max(4, MAX_I2V_PROMPT_WORDS - reserved)
    identity_words = identity.split()[:budget]
    chunks = []
    if identity_words:
        chunks.append(" ".join(identity_words))
    if action:
        chunks.append(action.strip(" ,.;"))
    chunks.append(suffix)
    text = ". ".join(chunks)
    words = text.split()
    if len(words) > MAX_I2V_PROMPT_WORDS:
        text = " ".join(words[:MAX_I2V_PROMPT_WORDS])
    return text.rstrip(" ,.;") + "."


def build_keyframe_model_prompt_preview(prompt: str) -> str:
    prompt = (prompt or "").strip().replace("\n", " ").strip(" ,.;，。；")
    use_zh = bool(re.search(r"[\u4e00-\u9fff]", prompt + KEYFRAME_SUFFIX))
    sep = "。" if use_zh else ". "
    end = "。" if use_zh else "."
    text = f"{prompt}{sep}{KEYFRAME_SUFFIX}" if prompt else KEYFRAME_SUFFIX
    if use_zh:
        max_chars = MAX_KEYFRAME_PROMPT_WORDS * 2
        if len(text) > max_chars:
            return text[:max_chars].rstrip(" ,.;，。；") + end
        return text.rstrip(" ,.;，。；") + end
    words = text.split()
    if len(words) > MAX_KEYFRAME_PROMPT_WORDS:
        return " ".join(words[:MAX_KEYFRAME_PROMPT_WORDS]).strip(" ,.;") + "."
    return text.rstrip(" ,.;") + "."


def run_batch_validators(
    raw_shots: list[str],
    beats: list[dict] | None = None,
    visual_anchor: str = "",
) -> list[dict]:
    prop_name, grip_hand = infer_prop_grip(
        visual_anchor,
        *[str(b.get("end_state", "") or "") for b in (beats or [])],
        *[str(b.get("start_state", "") or "") for b in (beats or [])],
    )

    def _motion_sanity(shots: list[str]) -> tuple[bool, str]:
        return validate_motion_sanity(shots, grip_hand=grip_hand, prop_name=prop_name)

    validators = [
        ("motion_sanity", "肢体/道具语义", _motion_sanity),
        ("motion_clarity", "旋转歧义", validate_motion_clarity),
        ("i2v_feasibility", "I2V 结构", validate_i2v_feasibility),
        ("shot_diversity", "镜头多样性", validate_shot_diversity),
    ]
    results = []
    for key, label, validator in validators:
        passed, reason = validator(raw_shots)
        results.append({"key": key, "label": label, "passed": passed, "reason": reason})

    passed, reason = validate_spine_safety(beats or [], grip_hand=grip_hand, prop_name=prop_name)
    results.append(
        {"key": "spine_safety", "label": "骨架安全(静止/占手)", "passed": passed, "reason": reason}
    )
    return results


def scan_shot_heuristics(
    scene_no: int,
    raw_shot: str,
    composed_prompt: str,
    model_prompt: str,
    beat_action: str,
    beat_end_state: str = "",
    keyframe_prompt: str = "",
    visual_anchor: str = "",
) -> list[dict]:
    warnings: list[dict] = []
    lower_raw = raw_shot.lower()

    def add(code: str, message: str, severity: str = "warning") -> None:
        warnings.append({"code": code, "severity": severity, "message": message})

    prop_name, grip_hand = infer_prop_grip(visual_anchor, beat_end_state, keyframe_prompt)

    if _STATIC_ACTION_RE.search(raw_shot) or _STATIC_ACTION_RE.search(beat_action):
        add("static_action", "动作静止，I2V 需要可见肢体运动。", "error")
    if prop_name and (_BOTH_HANDS_DOWN_RE.search(raw_shot) or _BOTH_HANDS_DOWN_RE.search(beat_action)):
        add("both_hands_down", f"双手放下会松开 {prop_name}。", "error")
    if grip_hand and prop_name:
        prop_arm = bool(
            re.search(
                rf"\b(raises?|lifts?|lowers?|pulls?)\s+(?:the\s+)?(?:{re.escape(prop_name)}|rod tip)\b",
                raw_shot,
                flags=re.I,
            )
        )
        if (
            not prop_arm
            and action_uses_hand(raw_shot, grip_hand)
            and re.search(r"\b(raises?|lifts?|points?|collar|brow)\b", raw_shot, flags=re.I)
        ):
            add(
                "occupied_hand_gesture",
                f"用占用手({grip_hand})做手势，与 {prop_name} 冲突；应改空闲手或先收道具。",
                "error",
            )

    if re.search(r"\b(raises?|lifts?)\s+it\b", lower_raw) or re.search(
        r"\braises?\s+free\s+hand\s+to\s+raises?\b", lower_raw
    ):
        add("broken_pronoun", "raw_shot 含残代词（raises it / raises free hand to raises）。", "error")

    for label, blob in (
        ("keyframe", keyframe_prompt),
        ("beat", beat_action),
        ("raw_shot", raw_shot),
    ):
        if looks_truncated(blob):
            add(f"{label}_truncated", f"{label} 疑似截断半句。", "error")
            break

    raw_words = len(raw_shot.split())
    if raw_words > MAX_I2V_ACTION_WORDS + 6:
        add("raw_shot_long", f"raw_shot 词数偏多 ({raw_words})。", "warning")
    if len(composed_prompt.split()) > MAX_I2V_TOTAL_WORDS + 8:
        add("composed_long", "composed 词数偏多。", "warning")
    if len(model_prompt.split()) >= MAX_I2V_PROMPT_WORDS:
        add("model_prompt_long", f"model_prompt 可能触顶 ({len(model_prompt.split())} 词)。", "warning")

    return warnings


def build_report(topic: str, storyboard: dict) -> dict:
    raw_shots = storyboard.get("raw_shots") or []
    prompts = storyboard.get("prompts") or []
    keyframes = storyboard.get("keyframe_prompts") or []
    beats = storyboard.get("continuity_spine") or []
    visual_anchor = storyboard.get("visual_anchor", "")

    batch_checks = run_batch_validators(raw_shots, beats=beats, visual_anchor=visual_anchor)
    all_batch_passed = all(c["passed"] for c in batch_checks)

    scenes = []
    total_errors = 0
    total_warnings = 0
    for i, raw in enumerate(raw_shots):
        composed = prompts[i] if i < len(prompts) else ""
        keyframe = keyframes[i] if i < len(keyframes) else ""
        beat = beats[i] if i < len(beats) else {}
        model_prompt = build_i2v_model_prompt_preview(composed)
        keyframe_model = build_keyframe_model_prompt_preview(keyframe)
        heuristics = scan_shot_heuristics(
            i + 1,
            raw,
            composed,
            model_prompt,
            str(beat.get("action", "")),
            beat_end_state=str(beat.get("end_state", "")),
            keyframe_prompt=keyframe,
            visual_anchor=visual_anchor,
        )
        error_count = sum(1 for h in heuristics if h["severity"] == "error")
        warn_count = sum(1 for h in heuristics if h["severity"] == "warning")
        total_errors += error_count
        total_warnings += warn_count
        scenes.append(
            {
                "scene": i + 1,
                "beat_action": str(beat.get("action", "")),
                "beat_end_state": str(beat.get("end_state", "")),
                "raw_shot": raw,
                "composed_prompt": composed,
                "model_prompt": model_prompt,
                "keyframe_prompt": keyframe,
                "keyframe_model_prompt": keyframe_model,
                "word_counts": {
                    "raw_shot": len(raw.split()),
                    "composed": len(composed.split()),
                    "model_prompt": len(model_prompt.split()),
                    "keyframe": len(keyframe.split()),
                },
                "heuristics": heuristics,
                "scene_passed": error_count == 0,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_topic": topic,
        "visual_anchor": visual_anchor,
        "overall_passed": all_batch_passed and total_errors == 0,
        "summary": {
            "scene_count": len(scenes),
            "batch_checks_passed": all_batch_passed,
            "heuristic_errors": total_errors,
            "heuristic_warnings": total_warnings,
            "average_prompt_similarity": round(average_pairwise_similarity(prompts), 3),
            "continuity_mode": "qwen_identity_edit_hardcut",
        },
        "batch_checks": batch_checks,
        "continuity_spine": beats,
        "scenes": scenes,
    }


def format_report_text(report: dict) -> str:
    lines = [
        f"主题: {report['user_topic']}",
        f"视觉锚点: {report.get('visual_anchor') or '(空)'}",
        f"总体结论: {'✅ 通过' if report['overall_passed'] else '❌ 存在问题'}",
        (
            f"批量校验: {'✅' if report['summary']['batch_checks_passed'] else '❌'} | "
            f"error: {report['summary']['heuristic_errors']} | "
            f"warning: {report['summary']['heuristic_warnings']} | "
            f"平均相似度: {report['summary']['average_prompt_similarity']}"
        ),
        "",
        "--- 批量校验 ---",
    ]
    for check in report["batch_checks"]:
        mark = "✅" if check["passed"] else "❌"
        lines.append(f"{mark} [{check['label']}] {check['reason']}")
    lines.append("")
    lines.append("--- 分镜 ---")
    for scene in report["scenes"]:
        mark = "✅" if scene["scene_passed"] else "❌"
        lines.append(f"{mark} Scene {scene['scene']:02d}: {scene['raw_shot']}")
        for h in scene["heuristics"]:
            lines.append(f"    [{h['severity']}] {h['code']}: {h['message']}")
    return "\n".join(lines)


def format_report_summary(report: dict) -> str:
    return (
        f"{'通过' if report['overall_passed'] else '未通过'} | "
        f"error={report['summary']['heuristic_errors']} "
        f"warning={report['summary']['heuristic_warnings']}"
    )


def validate_topic_prompts(
    topic: str,
    output_path: str | None = None,
    target_seconds: float | None = None,
) -> tuple[dict | None, str, str | None]:
    topic = topic.strip()
    if not topic:
        return None, "⚠️ 请输入视频主题！", None

    save_path = output_path or os.path.join(DEFAULT_OUTPUT_DIR, "prompt_validation.json")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    storyboard = generate_video_prompts(topic, target_seconds=target_seconds)
    if not storyboard.get("prompts"):
        return None, "❌ 分镜生成失败，请检查 LLM 服务或 API Key。", None

    report = build_report(topic, storyboard)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report, format_report_text(report), save_path


def print_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print(format_report_text(report))
    print("\n" + "=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成并校验 LLM 分镜 prompt（不渲染视频）")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="视频主题")
    parser.add_argument("-o", "--output", default="", help="验证报告 JSON 输出路径")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="期望成片秒数（可选；不传则从主题解析或按密度估算）",
    )
    args = parser.parse_args()

    topic = args.topic.strip()
    if not topic:
        print("❌ 主题不能为空")
        return 1

    output_path = args.output.strip() or os.path.join(DEFAULT_OUTPUT_DIR, "prompt_validation.json")
    print("=" * 72)
    print("Prompt 验证脚本启动")
    print(f"主题: {topic}")
    if args.seconds:
        print(f"指定时长: {args.seconds}s")
    print(f"报告输出: {output_path}")
    print("=" * 72)

    report, detail_text, saved_path = validate_topic_prompts(
        topic, output_path, target_seconds=args.seconds
    )
    if report is None:
        print(detail_text)
        return 1

    print_report(report)
    print(f"\n🧾 完整报告已保存: {saved_path}")
    return 0 if report["overall_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

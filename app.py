import gradio as gr
import requests

from validate_prompts import format_report_summary, validate_topic_prompts

# FastAPI 服务的本地地址
API_URL = "http://127.0.0.1:8000"


def _normalize_target_seconds(target_seconds):
    """Gradio Number 空值 / 0 → None，交给后端按主题自动估镜。"""
    if target_seconds is None:
        return None
    try:
        value = float(target_seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(10.0, min(60.0, value))


def submit_task(topic, target_seconds=None):
    """向 API 提交生成任务"""
    if not topic or not topic.strip():
        return "⚠️ 请输入视频主题！", ""

    seconds = _normalize_target_seconds(target_seconds)
    payload = {"topic": topic.strip()}
    if seconds is not None:
        payload["target_seconds"] = seconds

    try:
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
        ).json()
        task_id = response.get("task_id")
        duration_hint = (
            f"指定时长约 {int(seconds)} 秒"
            if seconds is not None
            else "镜数按主题复杂度自动分配（上限约 60 秒 / 12 镜）"
        )
        msg = (
            f"✅ 任务提交成功！（{duration_hint}）\n\n🔑 您的专享取件码是：\n{task_id}\n\n"
            "后台正在拼命渲染中。您可以现在关闭网页，"
            "随时在「视频生成」页凭取件码查询进度或下载视频！"
        )
        return msg, task_id
    except Exception as e:
        return f"❌ 提交失败：请确保 FastAPI 服务已启动。错误信息：{e}", ""


def check_status(task_id):
    """凭取件码查询进度，如果完成则显示视频"""
    if not task_id:
        return "⚠️ 请输入取件码！", None, gr.update(visible=False)

    try:
        res = requests.get(f"{API_URL}/status/{task_id}").json()
        state = res.get("state")

        if state == "SUCCESS":
            final_video_path = res["result"]["final_video"]
            return (
                "🎉 视频已生成完毕！请在下方播放或下载。",
                final_video_path,
                gr.update(value=final_video_path, visible=True),
            )
        if state == "PROGRESS":
            info = res.get("info", "")
            return f"⏳ 正在处理中，当前进度：\n{info}", None, gr.update(visible=False)
        if state == "PENDING":
            return "🕒 正在排队等待分配显卡...", None, gr.update(visible=False)
        return f"❌ 任务状态异常：{state}", None, gr.update(visible=False)
    except Exception as e:
        return f"❌ 查询失败，错误信息：{e}", None, gr.update(visible=False)


def run_prompt_validation(topic, target_seconds=None):
    """使用页面输入的主题生成分镜并校验 prompt（不渲染视频）。"""
    seconds = _normalize_target_seconds(target_seconds)
    report, detail_text, json_path = validate_topic_prompts(
        topic, target_seconds=seconds
    )
    if report is None:
        return detail_text, detail_text, gr.update(value=None, visible=False)

    summary_md = format_report_summary(report)
    if json_path:
        summary_md = f"{summary_md}\n\n报告已保存：`{json_path}`"
    return summary_md, detail_text, gr.update(value=json_path, visible=True)


# ==========================================
# Gradio UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AI 电影工厂 (工业级异步架构版)")
    gr.Markdown(
        "输入**一句话或一段故事**，可先 **验证 Prompt** 检查分镜是否合理，"
        "确认无误后再 **提交渲染**。\n\n"
        "镜数不再锁死：系统容量最多约 **1 分钟**，具体长短按主题复杂度自动决定；"
        "也可填「期望时长」，或在主题里写「生成30秒」。"
    )

    topic_input = gr.Textbox(
        label="描述您的视频主题",
        lines=3,
        max_lines=10,
        placeholder="例如：赛博朋克雨夜，赏金猎人撑伞走入暗巷……（也可写：生成30秒）",
    )
    duration_input = gr.Number(
        label="期望成片时长（秒，可选）",
        value=None,
        minimum=10,
        maximum=60,
        step=5,
        info="留空=按主题复杂度自动估镜（短故事更短）；填 30 ≈ 6 镜；上限 60 ≈ 12 镜。",
    )

    with gr.Tabs():
        with gr.Tab("🔍 Prompt 验证"):
            gr.Markdown(
                "只调用 LLM 生成分镜并做语义校验，"
                "**不加载 Wan、不渲染视频**，约 1–2 分钟完成。"
                "镜数随主题/时长变化（约 3–12 镜，每镜约 5 秒，最长约一分钟）。"
            )
            validate_btn = gr.Button("🔍 生成并验证 Prompt", variant="secondary")
            validation_summary = gr.Markdown(label="验证摘要")
            validation_detail = gr.Textbox(
                label="完整验证报告",
                interactive=False,
                lines=24,
                max_lines=40,
            )
            validation_download = gr.DownloadButton(
                "💾 下载 prompt_validation.json",
                visible=False,
            )

        with gr.Tab("🚀 视频生成"):
            gr.Markdown("### 提交任务 & 凭码取件")
            with gr.Row():
                with gr.Column(variant="panel"):
                    submit_btn = gr.Button("🚀 提交渲染任务", variant="primary")
                    submit_msg = gr.Textbox(
                        label="系统通知 (请妥善保存取件码)",
                        interactive=False,
                        lines=4,
                    )
                with gr.Column(variant="panel"):
                    task_id_input = gr.Textbox(label="🔑 输入您的取件码 (Task ID)")
                    check_btn = gr.Button("🔍 查询进度 / 提取视频")
                    status_msg = gr.Textbox(label="当前状态", interactive=False)
                    final_video_player = gr.Video(label="成片预览")
                    download_btn = gr.DownloadButton(
                        "💾 点击下载完整成片",
                        visible=False,
                        variant="primary",
                    )

    validate_btn.click(
        fn=run_prompt_validation,
        inputs=[topic_input, duration_input],
        outputs=[validation_summary, validation_detail, validation_download],
    )
    submit_btn.click(
        fn=submit_task,
        inputs=[topic_input, duration_input],
        outputs=[submit_msg, task_id_input],
    )
    check_btn.click(
        fn=check_status,
        inputs=task_id_input,
        outputs=[status_msg, final_video_player, download_btn],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)

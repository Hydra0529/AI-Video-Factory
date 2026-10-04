from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from celery_worker import process_1min_video_task, app as celery_app
from celery.result import AsyncResult

app = FastAPI(title="AI 电影工厂 API")


class VideoRequest(BaseModel):
    topic: str
    # 可选：期望成片秒数。不传则从主题文本解析（如「生成30秒」），再否则按主题密度估镜数。
    target_seconds: Optional[float] = Field(
        default=None,
        ge=10,
        le=60,
        description="期望成片时长（秒），上限 60。例如 30 → 约 6 镜；留空则按主题复杂度自动分配。",
    )


@app.post("/generate")
def generate_video(req: VideoRequest):
    """接单接口：瞬间返回取件码（task_id）"""
    task = process_1min_video_task.delay(req.topic, req.target_seconds)
    return {
        "message": "已成功下单！请保存好您的取件码，您可以随时关闭网页，稍后来查询。",
        "task_id": task.id,
        "target_seconds": req.target_seconds,
    }


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """查询接口：凭取件码查询进度或提取视频"""
    task = AsyncResult(task_id, app=celery_app)

    if task.state == 'PENDING':
        return {"state": "PENDING", "info": "正在排队中..."}
    elif task.state == 'PROGRESS':
        # 返回我们在 worker 里写好的进度描述
        return {"state": "PROGRESS", "info": task.info.get('step', '正在渲染中...')}
    elif task.state == 'SUCCESS':
        return {"state": "SUCCESS", "result": task.result}
    elif task.state == 'FAILURE':
        return {"state": "FAILURE", "info": "生成失败"}
    else:
        return {"state": task.state, "info": "未知状态"}
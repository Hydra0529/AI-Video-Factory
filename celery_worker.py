import os
import sys

# 确保 Celery worker 无论从哪个目录启动，都能找到同目录下的模块
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
os.chdir(_PROJECT_DIR)

# 彻底关闭国内镜像，强制恢复默认的全球抱脸官方通道！
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

# 依然保持禁用这个不稳定的极速插件
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# 降低 HuggingFace / Accelerate 加载权重时的内存峰值
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from celery import Celery
from celery.signals import worker_ready
from main import run_pipeline
from video_merger import merge_video_clips

app = Celery(
    'video_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

app.conf.update(
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=1,
    # solo 池：不 fork 子进程，避免加载 14B 模型时内存翻倍被 OOM Killer 杀掉
    worker_pool=os.getenv("CELERY_POOL", "solo"),
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Worker 启动时提示：Redis 里若有旧任务会自动执行，并非网页误触发。"""
    print("=" * 60)
    print("Celery Worker 已就绪，等待新任务。")
    print(f"Worker 池模式: {app.conf.worker_pool}（加载 Wan14B 请保持 solo，勿用 prefork）")
    print("注意：Redis 队列中若有之前提交但未执行完的任务，会立即被取出执行。")
    print("若不想跑旧任务，启动 worker 前先执行：celery -A celery_worker purge -f")
    print("若加载模型时出现 Killed / signal 9 (SIGKILL)，是容器内存不足被系统杀死。")
    print("请查看监控面板的「容器内存上限」（非宿主机 1TB 内存），并关闭其他占内存进程。")
    print("=" * 60)


@app.task(bind=True)
def process_1min_video_task(self, topic: str, target_seconds: float | None = None):
    """
    后台耗时任务：原子动作分镜 -> 逐段渲染 -> 拼接成片
    仅由网页/API 调用 .delay(topic, target_seconds=...) 触发，worker 本身不会自动生成任务。
    镜数按主题密度或用户指定时长自动分配，不再锁死固定镜头数。
    """
    output_dir = os.getenv("OUTPUT_DIR", "/root/autodl-tmp/output_clips")
    final_dir = os.getenv("FINAL_VIDEO_DIR", "/root/autodl-tmp/final_videos")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    final_video_path = os.path.join(final_dir, f"final_video_{self.request.id}.mp4")

    print(f"\n>>> 收到 Celery 任务 task_id={self.request.id}")
    print(f">>> 任务主题: {topic[:80]}{'...' if len(topic) > 80 else ''}")
    print(f">>> 成片目录: {final_dir}")
    if target_seconds:
        print(f">>> 用户指定时长: {target_seconds}s")

    try:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                print(
                    f">>> GPU: {props.name} | 显存 {props.total_memory / (1024 ** 3):.1f} GB | "
                    f"请确认 WAN_OFFLOAD=sequential（24GB 勿用 model）"
                )
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        self.update_state(state='PROGRESS', meta={'step': '阶段 1/3：正在构思原子动作分镜（一镜一动作）...'})
        if not run_pipeline(topic, target_seconds=target_seconds):
            raise RuntimeError("剧本或视频素材生成失败，请检查 LLM 返回内容和模型日志。")

        self.update_state(state='PROGRESS', meta={'step': '阶段 3/3：镜头渲染完毕，正在拼接成片...'})
        merged_path = merge_video_clips(input_dir=output_dir, output_path=final_video_path)
        if not merged_path:
            raise RuntimeError("视频合并或补帧失败，请检查 ffmpeg / moviepy 日志。")

        return {
            'status': 'Success',
            'message': '大片生成完毕！',
            'final_video': final_video_path
        }
    except Exception as e:
        print(f"❌ 任务失败: {e}")
        raise


if __name__ == "__main__":
    import sys

    argv = ["worker", "-P", "solo", "--loglevel=info", "--concurrency=1", *sys.argv[1:]]
    app.worker_main(argv=argv)

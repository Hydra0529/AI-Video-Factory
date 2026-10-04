import os
import glob
import shutil
import subprocess
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip

SOURCE_FPS = 16
# 16→24fps 的无插值转换会按 2:1:2:1 复制帧，产生明显颤动；默认保持 16fps。
# 如需 24fps 请用外部插帧工具（RIFE/film）后再转。
TARGET_FPS = int(os.getenv("TARGET_FPS", "16"))
ENCODE_CRF = 14
# 镜间为 Scene1 回锚硬切（姿态可跳），交叉淡化会产生双重曝光叠影，默认硬切。
CROSSFADE_DURATION = float(os.getenv("CROSSFADE_DURATION", "0"))
# 1360x768 成片建议更高码率；可通过环境变量覆盖
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "24000k")


def _convert_to_target_fps(input_path: str, output_path: str, target_fps: int = TARGET_FPS) -> bool:
    """Use a lightweight fps conversion instead of heavy motion interpolation."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("⚠️ 未找到 ffmpeg，无法进行帧率转换，将保留 MoviePy 输出。")
        return False

    command = [
        ffmpeg_bin,
        "-y",
        "-i", input_path,
        "-vf", f"fps={target_fps}",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(ENCODE_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️ ffmpeg 帧率转换失败，将保留 MoviePy 输出。错误信息：")
        print(result.stderr[-2000:])
        return False
    return True


def merge_video_clips(
    input_dir="/root/autodl-tmp/output_clips",
    output_path="/root/autodl-tmp/final_videos/final_video.mp4",
    bgm_path=None,
):
    """
    自动读取指定文件夹下的所有短片，按顺序拼接成完整长片
    :param input_dir: 分镜视频所在的文件夹
    :param output_path: 最终成片的保存路径（默认 /root/autodl-tmp/final_videos/）
    :param bgm_path: 可选，背景音乐的路径（.mp3 或 .wav）
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 40)
    print("🎬 正在启动 Python 自动化剪辑引擎...")
    print("=" * 40)

    video_files = sorted(glob.glob(os.path.join(input_dir, "scene_*.mp4")))

    if not video_files:
        print(f"❌ 错误：在 {input_dir} 目录中未找到 scene_*.mp4 视频片段！")
        return None

    print(f"📦 成功识别到 {len(video_files)} 个原子动作镜头，开始加载视频流...")

    clips = []
    final_clip = None
    try:
        for file in video_files:
            print(f" 正在导入: {os.path.basename(file)}")
            clip = VideoFileClip(file)
            clips.append(clip)

        if len(clips) == 1:
            final_clip = clips[0]
        elif CROSSFADE_DURATION <= 0:
            print(f"\n🔄 正在将 {len(clips)} 个镜头硬切拼接（无交叉淡化，避免叠影）...")
            final_clip = concatenate_videoclips(clips, method="compose")
        else:
            print(f"\n🔄 正在将 {len(clips)} 个镜头拼接（交叉淡化 {CROSSFADE_DURATION}s）...")
            faded_clips = [clips[0]]
            for clip in clips[1:]:
                try:
                    faded_clips.append(clip.crossfadein(CROSSFADE_DURATION))
                except AttributeError:
                    faded_clips.append(clip)
            final_clip = concatenate_videoclips(
                faded_clips,
                method="compose",
                padding=-CROSSFADE_DURATION,
            )

        if bgm_path and os.path.exists(bgm_path):
            print("🎵 检测到背景音乐，正在进行音频混音调音...")
            audio = AudioFileClip(bgm_path)
            audio = audio.subclip(0, final_clip.duration)
            final_clip = final_clip.set_audio(audio)
        else:
            print(" silent：未检测到外部背景音乐，将输出静音成片。")

        temp_output_path = output_path
        if TARGET_FPS != SOURCE_FPS:
            base, ext = os.path.splitext(output_path)
            temp_output_path = f"{base}_source{ext}"

        print("\n💾 正在渲染最终成片，请稍候...")
        final_clip.write_videofile(
            temp_output_path,
            fps=SOURCE_FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate=VIDEO_BITRATE,
            preset="slow",
            threads=8,
            ffmpeg_params=["-crf", str(ENCODE_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

        if TARGET_FPS != SOURCE_FPS:
            print(f"\n🎞️ 正在使用 ffmpeg 转换到 {TARGET_FPS}fps（无运动插帧，避免画面发糊）...")
            if not _convert_to_target_fps(temp_output_path, output_path, TARGET_FPS):
                if temp_output_path != output_path:
                    shutil.copy2(temp_output_path, output_path)
            elif os.path.exists(temp_output_path):
                os.remove(temp_output_path)

        print(f"\n🎉 渲染完美结束！成片已保存至: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ 剪辑渲染过程中发生严重错误: {e}")
        return None

    finally:
        print("🧹 正在清理剪辑缓存...")
        for clip in clips:
            clip.close()
        if final_clip:
            final_clip.close()


if __name__ == "__main__":
    merge_video_clips()

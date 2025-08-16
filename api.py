import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File,Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
import argparse
import gc
import torch
import subprocess
import sys
import logging
import threading
import queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置路径
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="LatentSync Video Generation API",
    description="API for generating lip-synced videos using LatentSync",
    version="1.0.0"
)

def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    """创建推理参数对象"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            str(CHECKPOINT_PATH.absolute()),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )

def clear_cuda_memory():
    """清理CUDA显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("显存已清理")

def run_inference_in_subprocess(args):
    """在子进程中运行推理任务并实时输出日志"""
    command = [
        sys.executable,  # 使用当前Python解释器
        "-m",
        "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--video_path", args.video_path,
        "--audio_path", args.audio_path,
        "--video_out_path", args.video_out_path,
        "--inference_steps", str(args.inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--seed", str(args.seed),
        "--temp_dir", "temp",
    ]
    
    if args.enable_deepcache:
        command.append("--enable_deepcache")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 强制无缓冲输出
    
    try:
        logger.info("启动视频生成子进程")
        
        # 启动子进程
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(PROJECT_ROOT),
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 创建队列和线程来捕获输出
        output_queue = queue.Queue()
        
        # 输出流读取线程
        def read_stream(stream, output_queue, stream_name):
            try:
                for line in stream:
                    output_queue.put(f"{stream_name}: {line.strip()}")
            except Exception as e:
                output_queue.put(f"读取错误: {str(e)}")
            finally:
                output_queue.put(None)  # 结束信号
        
        # 启动线程捕获stdout和stderr
        stdout_thread = threading.Thread(
            target=read_stream, 
            args=(process.stdout, output_queue, "STDOUT"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stream, 
            args=(process.stderr, output_queue, "STDERR"),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # 实时处理输出
        active_threads = 2
        all_output = []
        
        while active_threads > 0:
            try:
                line = output_queue.get(timeout=0.1)
                if line is None:
                    active_threads -= 1
                    continue
                
                # 记录输出
                logger.info(line)
                all_output.append(line)
                
            except queue.Empty:
                if process.poll() is not None:
                    break
        
        # 确保获取所有剩余输出
        while not output_queue.empty():
            line = output_queue.get_nowait()
            if line is not None:
                logger.info(line)
                all_output.append(line)
        
        # 等待进程结束
        return_code = process.wait()
        full_output = "\n".join(all_output)
        
        if return_code != 0:
            logger.error(f"视频生成失败，退出码: {return_code}")
            logger.error(full_output)
            return False, None
        
        logger.info("视频生成成功完成")
        return True, args.video_out_path
    except Exception as e:
        logger.error(f"子进程执行失败: {str(e)}")
        return False, None
    finally:
        clear_cuda_memory()

def generate_video_sync(
    video_path: str,
    audio_path: str,
    guidance_scale: float,
    inference_steps: int,
    seed: int
):
    """同步执行视频生成任务"""
    try:
        # 创建输出路径
        video_file_path = Path(video_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"{video_file_path.stem}_{current_time}.mp4")
        
        # 创建参数对象
        args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)
        
        # 运行推理
        success, result_path = run_inference_in_subprocess(args)
        
        return success, result_path
    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
        return False, None

from urllib.parse import unquote
from pathlib import Path
@app.post("/generate")
async def generate_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(),
    inference_steps: int = Form(),
    seed: int = Form()
):
    """同步生成视频接口，阻塞直到完成"""
    try:
        # 创建临时目录
        task_id = str(uuid.uuid4())
        temp_dir = PROJECT_ROOT / "uploads" / task_id
        
        # 确保使用绝对路径
        temp_dir = temp_dir.absolute()
        
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建临时目录 {temp_dir}: {str(e)}")
            raise HTTPException(status_code=500, detail="无法创建临时目录")

        # 打印调试信息
        logger.info(f"guidance_scale:  {guidance_scale}")
        logger.info(f"inference_steps: {inference_steps}")
        logger.info(f"临时目录路径: {temp_dir}")
        logger.info(f"目录是否存在: {temp_dir.exists()}")

        # 处理文件名 - 移除特殊字符
        def sanitize_filename(filename: str) -> str:
            filename = unquote(filename)
            # 移除非法字符
            return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        
        video_filename = sanitize_filename(video.filename)
        audio_filename = sanitize_filename(audio.filename)

        video_path = temp_dir / video_filename
        audio_path = temp_dir / audio_filename

        # 保存文件
        try:
            # 写入视频文件
            with open(video_path, "wb") as buffer:
                content = await video.read()
                buffer.write(content)
                logger.info(f"已保存视频文件 {video_path} ({len(content)} bytes)")
            
            # 写入音频文件
            with open(audio_path, "wb") as buffer:
                content = await audio.read()
                buffer.write(content)
                logger.info(f"已保存音频文件 {audio_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"文件保存失败: {str(e)}")
            raise HTTPException(status_code=500, detail="文件保存失败")

        # 检查文件是否确实存在
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            raise HTTPException(status_code=500, detail="视频文件保存后不存在")
        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            raise HTTPException(status_code=500, detail="音频文件保存后不存在")

        # 打印完整路径
        logger.info(f"视频文件完整路径: {video_path.absolute()}")
        logger.info(f"音频文件完整路径: {audio_path.absolute()}")

        # 同步执行视频生成
        logger.info(f"开始同步生成视频，任务ID: {task_id}")
        success, result_path = generate_video_sync(
            str(video_path.absolute()),  # 使用绝对路径
            str(audio_path.absolute()),
            guidance_scale,
            inference_steps,
            seed
        )
        
        if not success or not result_path:
            # 清理上传的文件
           # try:
                # for file in temp_dir.glob("*"):
               #     file.unlink()
                # temp_dir.rmdir()
          #  except Exception as e:
          #      logger.error(f"清理临时文件失败: {str(e)}")
            
            raise HTTPException(
                status_code=500, 
                detail="视频生成失败，请检查日志获取详细信息"
            )
        
        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename=Path(result_path).name,
            headers={"X-Task-ID": task_id}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频生成请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"视频生成请求处理失败: {str(e)}"
        )

@app.get("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """清理任务生成的文件"""
    try:
        # 清理上传的文件
        upload_dir = OUTPUT_DIR / "uploads" / task_id
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                file.unlink()
            upload_dir.rmdir()
            logger.info(f"清理上传文件: {upload_dir}")
        
        # 清理生成的视频文件（由客户端决定是否保留）
        # 这里不自动清理生成的视频文件，因为可能还需要使用
        
        return {"status": "success", "message": f"清理任务 {task_id} 的上传文件"}
    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"清理任务失败: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,timeout_keep_alive=7200)
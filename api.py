import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
import argparse
import gc
import torch
import subprocess
import sys
import logging
import threading
import queue
import asyncio
from typing import Dict, Optional
from enum import Enum
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置路径
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 任务状态存储
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 全局任务存储
tasks: Dict[str, dict] = {}

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

async def run_video_generation(task_id: str, video_path: str, audio_path: str, 
                              guidance_scale: float, inference_steps: int, seed: int):
    """异步运行视频生成任务"""
    try:
        # 更新任务状态为处理中
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["start_time"] = datetime.now().isoformat()
        
        # 在线程池中运行同步任务
        loop = asyncio.get_event_loop()
        success, result_path = await loop.run_in_executor(
            None, generate_video_sync, video_path, audio_path, guidance_scale, inference_steps, seed
        )
        
        # 更新任务状态
        if success and result_path:
            tasks[task_id]["status"] = TaskStatus.COMPLETED
            tasks[task_id]["result_path"] = result_path
            tasks[task_id]["end_time"] = datetime.now().isoformat()
            logger.info(f"任务 {task_id} 完成，结果路径: {result_path}")
        else:
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["end_time"] = datetime.now().isoformat()
            tasks[task_id]["error"] = "视频生成失败"
            logger.error(f"任务 {task_id} 失败")
            
    except Exception as e:
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["end_time"] = datetime.now().isoformat()
        tasks[task_id]["error"] = str(e)
        logger.error(f"任务 {task_id} 执行异常: {str(e)}")

from urllib.parse import unquote
from pathlib import Path
@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(1.5),
    inference_steps: int = Form(20),
    seed: int = Form(1247)
):
    """异步生成视频接口，立即返回任务ID"""
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "create_time": datetime.now().isoformat(),
        "video_filename": video.filename,
        "audio_filename": audio.filename,
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "seed": seed
    }
    
    try:
        # 创建临时目录
        temp_dir = PROJECT_ROOT / "uploads" / task_id
        
        # 确保使用绝对路径
        temp_dir = temp_dir.absolute()
        
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建临时目录 {temp_dir}: {str(e)}")
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = "无法创建临时目录"
            raise HTTPException(status_code=500, detail="无法创建临时目录")

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
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = "文件保存失败"
            raise HTTPException(status_code=500, detail="文件保存失败")

        # 检查文件是否确实存在
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = "视频文件不存在"
            raise HTTPException(status_code=500, detail="视频文件保存后不存在")
        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = "音频文件不存在"
            raise HTTPException(status_code=500, detail="音频文件保存后不存在")

        # 更新任务信息
        tasks[task_id]["video_path"] = str(video_path.absolute())
        tasks[task_id]["audio_path"] = str(audio_path.absolute())
        
        # 添加后台任务
        background_tasks.add_task(
            run_video_generation, 
            task_id, 
            str(video_path.absolute()),
            str(audio_path.absolute()),
            guidance_scale,
            inference_steps,
            seed
        )
        
        logger.info(f"已启动后台任务，任务ID: {task_id}")
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "视频生成任务已开始处理",
                "task_id": task_id,
                "status_url": f"/task/{task_id}",
                "download_url": f"/download/{task_id}"
            }
        )
        
    except Exception as e:
        logger.error(f"视频生成请求处理失败: {str(e)}", exc_info=True)
        if task_id in tasks:
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = str(e)
        raise HTTPException(
            status_code=500, 
            detail=f"视频生成请求处理失败: {str(e)}"
        )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = tasks[task_id].copy()
    
    # 移除可能过大的字段
    for key in ["video_path", "audio_path", "result_path"]:
        if key in task_info:
            task_info[key] = "已设置" if task_info[key] else "未设置"
    
    return task_info

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """下载生成的视频文件"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成或失败")
    
    if "result_path" not in task or not task["result_path"]:
        raise HTTPException(status_code=500, detail="任务结果路径不存在")
    
    result_path = Path(task["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="生成的文件不存在")
    
    filename = f"generated_{task_id}{result_path.suffix}"
    
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=filename,
        headers={"X-Task-ID": task_id}
    )

@app.get("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """清理任务生成的文件"""
    try:
        # 清理上传的文件
        upload_dir = PROJECT_ROOT / "uploads" / task_id
        if upload_dir.exists() and upload_dir.is_dir():
            for file in upload_dir.glob("*"):
                file.unlink()
            upload_dir.rmdir()
            logger.info(f"清理上传文件: {upload_dir}")
        
        # 清理生成的视频文件
        if task_id in tasks and "result_path" in tasks[task_id]:
            result_path = Path(tasks[task_id]["result_path"])
            if result_path.exists():
                result_path.unlink()
                logger.info(f"清理生成文件: {result_path}")
        
        # 从任务列表中移除
        if task_id in tasks:
            del tasks[task_id]
        
        return {"status": "success", "message": f"清理任务 {task_id} 的文件完成"}
    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"清理任务失败: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=7200)
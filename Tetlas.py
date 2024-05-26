import os
import torch
import cupy as cp
import numpy as np
import time
import psutil
from PIL import Image
from tqdm import tqdm
import logging
import tkinter as tk
from tkinter import messagebox
from torch.multiprocessing import Pool, set_start_method
import subprocess

try:
    set_start_method('spawn')
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

ASPECT_RATIO = 9 / 16


def compute_tetration_step(c: cp.ndarray, z: cp.ndarray, escape_radius: float) -> tuple[cp.ndarray, cp.ndarray]:
    z = c ** z
    mask = cp.abs(z) > escape_radius
    return z, mask


def compute_tetration_divergence_gpu(nx: int, ny: int, max_iter: int, escape_radius: float, px: float, py: float,
                                     scale: float) -> np.ndarray:
    x = cp.linspace(px - scale, px + scale, nx)
    y = cp.linspace(py - scale * ASPECT_RATIO, py + scale * ASPECT_RATIO, ny)
    c = x[:, None] + 1j * y[None, :]

    z = c.copy()
    divergence_map = cp.zeros(c.shape, dtype=cp.bool_)

    for _ in range(max_iter):
        z, mask = compute_tetration_step(c, z, escape_radius)
        divergence_map[mask] = True
        z[mask] = escape_radius

    return cp.asnumpy(divergence_map)


def create_custom_colormap() -> np.ndarray:
    return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)


def save_frame(divergence_map: np.ndarray, nx: int, ny: int, frame: int, output_folder: str, cmap: np.ndarray) -> str:
    try:
        if divergence_map is None or divergence_map.size == 0:
            raise ValueError("divergence_map is empty or None")

        divergence_image = (divergence_map * 255).astype(np.uint8)
        divergence_image = np.stack((divergence_image,) * 3, axis=-1)  # Convert to RGB

        image = Image.fromarray(divergence_image)
        file_path = os.path.join(output_folder, f'tetration_zoom_frame_{frame}.png')
        image.save(file_path)

        logging.info(f"Frame {frame} saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving frame {frame}: {e}", exc_info=True)
        return None


def process_frame_gpu(args) -> tuple[float, float, str]:
    frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius, output_folder, cmap = args

    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    try:
        scale = scale_initial * (zoom_factor ** frame)
        px_frame = px + (px_target - px) * (1 - zoom_factor ** frame)
        py_frame = py + (py_target - py) * (1 - zoom_factor ** frame)

        divergence_map = compute_tetration_divergence_gpu(nx, ny, max_iter, escape_radius, px_frame, py_frame, scale)

        file_path = save_frame(divergence_map, nx, ny, frame, output_folder, cmap)
    except Exception as e:
        logging.error(f"Error processing frame {frame}: {e}", exc_info=True)
        return None, None, None
    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    duration = end_time - start_time
    memory_used = end_mem - start_mem
    logging.info(f"[GPU] Frame {frame} processed in {duration:.2f} seconds, using {memory_used:.2f} MB of memory")
    return duration, memory_used, file_path


def create_folder_if_not_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"폴더를 생성했어요: {folder_path}")
    else:
        logging.info(f"이 폴더에 저장할게요: {folder_path}")


def show_popup(message: str):
    root = tk.Tk()
    root.withdraw()  # 숨기기
    messagebox.showinfo("작업 완료", message)
    root.destroy()


if __name__ == '__main__':
    overall_start_time = time.time()
    current_path = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"현재 디렉토리: {current_path}")

    px, py = 0, 0

    #####target####

    px_target, py_target = 1.0, 0.0
    ##랜덤##
    # px_target, py_target = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)

    logging.info(f"target: ({px_target}) , ({py_target})")
    scale_initial = 5
    zoom_factor = 0.9
    nx, ny = 1920, 1080
    max_iter = 500
    escape_radius = 1e+10

    results_folder = f"results_{nx}x{ny}/Tetlas_{px_target}_{py_target}"
    output_folder = os.path.join(current_path, results_folder)

    create_folder_if_not_exists(output_folder)
    cmap = create_custom_colormap()

    start_frame = 1
    end_frame = 20

    args = [(frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius,
             output_folder, cmap) for frame in range(start_frame, end_frame + 1)]

    with Pool(processes=1) as pool:
        results = list(tqdm(pool.imap(process_frame_gpu, args), total=len(args), desc="Processing frames"))

    total_size = 0
    for result in results:
        if result[2]:
            file_size = os.path.getsize(result[2]) / (1024 * 1024)
            total_size += file_size

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    logging.info(f"전체 실행 시간: {overall_duration:.2f} 초")
    logging.info(f"저장된 디렉토리: {output_folder}")
    logging.info(f"저장된 총 용량: {total_size:.2f} MB")

    # 자동으로 bake_video.py 실행
    fps = 30
    output_video = os.path.join(current_path, f'Tetlas_{px_target}_{py_target}.mp4')
    subprocess.run(['python', 'bake_video.py', output_folder, output_video, str(fps)])

    show_popup(f"전체 실행 시간: {overall_duration:.2f} 초\n저장된 디렉토리: {output_folder}\n저장된 총 용량: {total_size:.2f} MB")

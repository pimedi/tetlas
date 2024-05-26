import os
import torch
import numpy as np
import time
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dask.distributed import Client, LocalCluster, progress
from tqdm import tqdm
import logging
import tkinter as tk
from tkinter import messagebox

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

ASPECT_RATIO = 9 / 16

def compute_tetration_step(c: torch.Tensor, z: torch.Tensor, escape_radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = c ** z
    mask = torch.abs(z) > escape_radius
    return z, mask

def compute_tetration_divergence_gpu(nx: int, ny: int, max_iter: int, escape_radius: float, px: float, py: float, scale: float, device: torch.device) -> np.ndarray:
    x = torch.linspace(px - scale, px + scale, nx, device=device)
    y = torch.linspace(py - scale * ASPECT_RATIO, py + scale * ASPECT_RATIO, ny, device=device)
    c = x[:, None] + 1j * y[None, :]

    z = c.clone()
    divergence_map = torch.zeros(c.shape, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        z, mask = compute_tetration_step(c, z, escape_radius)
        divergence_map[mask] = True
        z[mask] = escape_radius

    divergence_map_np = divergence_map.cpu().numpy().astype(np.float32)
    return divergence_map_np

def create_custom_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"])

def save_frame(divergence_map: np.ndarray, px: float, py: float, scale: float, nx: int, ny: int, frame: int, output_folder: str, cmap: LinearSegmentedColormap) -> str:
    try:
        dpi = 100
        fig_width, fig_height = nx / dpi, ny / dpi

        if divergence_map is None or divergence_map.size == 0:
            raise ValueError("divergence_map is empty or None")

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        im = ax.imshow(divergence_map.T, extent=[px - scale, px + scale, py - scale * ASPECT_RATIO, py + scale * ASPECT_RATIO], origin='lower', cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.axis('on')
        file_path = os.path.join(output_folder, f'tetration_zoom_frame_{frame}.png')
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logging.info(f"Frame {frame} saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving frame {frame}: {e}", exc_info=True)
        return None

def process_frame_gpu(frame: int, px: float, py: float, px_target: float, py_target: float, scale_initial: float, zoom_factor: float, nx: int, ny: int, max_iter: int, escape_radius: float, output_folder: str, cmap: LinearSegmentedColormap, device: torch.device) -> tuple[float, float, str]:
    start_time = time.time()
    torch.cuda.empty_cache()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    try:
        scale = scale_initial * (zoom_factor ** frame)
        px_frame = px + (px_target - px) * (1 - zoom_factor) ** frame
        py_frame = py + (py_target - py) * (1 - zoom_factor) ** frame

        divergence_map = compute_tetration_divergence_gpu(nx, ny, max_iter, escape_radius, px_frame, py_frame, scale, device)

        file_path = save_frame(divergence_map, px_frame, py_frame, scale, nx, ny, frame, output_folder, cmap)
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
        logging.info(f"폴더 생성했어요: {folder_path}")
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
    px_target, py_target = np.random.uniform(-1.0, 1.0), np.random.uniform(-5.0, 5.0)
    logging.info(f"target: ({px_target}) , ({py_target})")
    scale_initial = 5
    zoom_factor = 0.99
    
    ###해상도#####
    nx, ny = 1920, 1080
    #nx, ny = 3940, 2160
    max_iter = 500
    escape_radius = 1e+10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    results_folder = f"results_{nx}x{ny}/Tetlas_{px_target}_{py_target}"
    output_folder = os.path.join(current_path, results_folder)

    create_folder_if_not_exists(output_folder)
    cmap = create_custom_colormap()

    start_frame = 1
    end_frame = 1080

    cluster = LocalCluster(n_workers=4, memory_limit='2GB')
    client = Client(cluster)

    file_info_list = []

    try:
        futures = []
        for frame in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
            futures.append(client.submit(process_frame_gpu, frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius, output_folder, cmap, device))

        progress(futures)
        results = client.gather(futures)

        total_size = 0
        for result in results:
            if result[2]:
                file_size = os.path.getsize(result[2]) / (1024 * 1024)
                total_size += file_size
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
    finally:
        client.shutdown()
        cluster.close()

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    logging.info(f"전체 실행 시간: {overall_duration:.2f} 초")
    logging.info(f"저장된 디렉토리: {output_folder}")
    logging.info(f"저장된 총 용량: {total_size:.2f} MB")

    show_popup(f"전체 실행 시간: {overall_duration:.2f} 초\n저장된 디렉토리: {output_folder}\n저장된 총 용량: {total_size:.2f} MB")

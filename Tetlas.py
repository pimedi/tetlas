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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

try:
    set_start_method('spawn')
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

ASPECT_RATIO = 1


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

    return cp.asnumpy(divergence_map), cp.abs(z).get()


def create_custom_colormap(colors: list, positions: list = None) -> LinearSegmentedColormap:
    """
    Create a custom colormap from a list of colors and their positions.

    Parameters:
    colors (list): A list of tuples representing colors in RGB format.
    positions (list): A list of positions for each color (0 to 1). If None, colors are evenly spaced.

    Returns:
    LinearSegmentedColormap: A custom linear segmented colormap.
    """
    if not colors or not all(isinstance(color, tuple) and len(color) == 3 for color in colors):
        raise ValueError("colors must be a list of RGB tuples")
    if positions is None:
        positions = np.linspace(0, 1, len(colors))
    if len(colors) != len(positions):
        raise ValueError("The number of colors must match the number of positions")

    return LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)), N=256)


def add_colorbar(image: Image.Image, cmap: LinearSegmentedColormap, image_height: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(1, image_height / 100))  # 세로 길이 설정
    fig.subplots_adjust(left=0.5, right=0.7, top=0.95, bottom=0.05)

    norm = plt.Normalize(0, 1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='vertical')
    cb1.set_label('Values')

    fig.canvas.draw()
    colorbar_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    colorbar_image = colorbar_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    colorbar_image = Image.fromarray(colorbar_image)

    combined_image = Image.new('RGB', (image.width + colorbar_image.width, image.height))
    combined_image.paste(image, (0, 0))
    combined_image.paste(colorbar_image, (image.width, 0))

    plt.close(fig)
    return combined_image


def save_frame(divergence_map: np.ndarray, z_values: np.ndarray, nx: int, ny: int, frame: int, output_folder: str,
               cmap: LinearSegmentedColormap) -> str:
    try:
        if divergence_map is None or divergence_map.size == 0:
            raise ValueError("divergence_map is empty or None")

        if z_values is None or z_values.size == 0:
            raise ValueError("z_values is empty or None")

        # Normalize and scale values for color mapping
        z_values = np.abs(z_values)  # Ensure no negative values for log1p
        normalized_values = np.log1p(z_values)
        max_normalized_value = normalized_values.max()
        if max_normalized_value == 0:
            raise ValueError("Max normalized value is zero, which may indicate a problem with z_values")
        normalized_values /= max_normalized_value

        # Apply colormap to normalized values
        color_mapped_image = cmap(normalized_values)

        # Set diverged values to white
        color_mapped_image[divergence_map] = (1, 1, 1, 1)

        image = (color_mapped_image[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(image)

        image_with_colorbar = add_colorbar(image, cmap, ny)

        file_path = os.path.join(output_folder, f'tetration_zoom_frame_{frame}.png')
        image_with_colorbar.save(file_path)
        logging.info(f"Frame {frame} saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving frame {frame}: {e}", exc_info=True)
        return None


def process_frame_gpu(args) -> tuple[float, float, str]:
    frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius, output_folder, colors, positions = args

    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    try:
        scale = scale_initial * (zoom_factor ** frame)
        px_frame = px + (px_target - px) * (1 - zoom_factor ** frame)
        py_frame = py + (py_target - py) * (1 - zoom_factor ** frame)

        divergence_map, z_values = compute_tetration_divergence_gpu(nx, ny, max_iter, escape_radius, px_frame, py_frame,
                                                                    scale)

        cmap = create_custom_colormap(colors, positions)
        file_path = save_frame(divergence_map, z_values, nx, ny, frame, output_folder, cmap)
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
    root.attributes("-topmost", True)  # 최상위 우선순위 설정
    messagebox.showinfo("작업 완료", message)
    root.destroy()


if __name__ == '__main__':
    overall_start_time = time.time()
    current_path = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"현재 디렉토리: {current_path}")

    px, py = 0, 0

    #####target####

    px_target, py_target = 1e-1, 0.0
    logging.info(f"target: ({px_target}) , ({py_target})")
    scale_initial = 5
    zoom_factor = 0.9
    nx, ny = 1200, 1200
    max_iter = 500
    escape_radius = 1e+10

    results_folder = f"results_{nx}x{ny}/Tetlas_{px_target}_{py_target}"
    output_folder = os.path.join(current_path, results_folder)

    create_folder_if_not_exists(output_folder)

    custom_colors = [
        (0.0, 0.0, 0.0),  # Black
        (0.1, 0.1, 0.5),  # Dark Blue
        (0.0, 0.3, 0.7),  # Medium Blue
        (0.0, 0.5, 1.0),  # Blue
        (0.0, 0.75, 1.0),  # Light Blue
        (0.0, 1.0, 1.0),  # Cyan
        (0.25, 0.75, 0.75),  # Light Cyan
        (0.5, 1.0, 0.5),  # Light Green
        (0.75, 1.0, 0.25),  # Yellow-Green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.75, 0.0),  # Dark Yellow
        (1.0, 0.5, 0.0),  # Orange
        (1.0, 0.25, 0.0),  # Dark Orange
        (1.0, 0.0, 0.0),  # Red
        (0.75, 0.0, 0.25),  # Dark Red
        (0.5, 0.0, 0.5),  # Purple
        (0.75, 0.25, 0.75),  # Light Purple
        (1.0, 0.5, 1.0),  # Magenta
        (1.0, 0.75, 1.0),  # Light Pink
        (1.0, 1.0, 1.0)  # White
    ]

    color_positions = [
        0, 5e-3, 1e-2, 15e-3, 2e-2, 25e-3, 3e-2, 35e-3, 4e-2, 45e-3,
        5e-2, 55e-3, 6e-2, 65e-3, 7e-2, 75e-3, 8e-2, 85e-3, 9e-2, 1
    ]

    start_frame = 1
    end_frame = 20

    args = [(frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius,
             output_folder, custom_colors, color_positions) for frame in range(start_frame, end_frame + 1)]

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

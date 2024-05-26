import os
import torch
import numpy as np
import time
import psutil
import matplotlib
matplotlib.use('Agg')  # 비-GUI 백엔드 설정
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dask.distributed import Client, LocalCluster, progress
from tqdm import tqdm
import gc

ASPECT_RATIO = 9 / 16

def compute_tetration_step(c, z, escape_radius):
    z = c ** z
    mask = torch.abs(z) > escape_radius
    return z, mask

def compute_tetration_divergence_gpu(nx, ny, max_iter, escape_radius, px, py, scale, device):
    x = torch.linspace(px - scale, px + scale, nx, device=device)
    y = torch.linspace(py - scale * ASPECT_RATIO, py + scale * ASPECT_RATIO, ny, device=device)
    c = x[:, None] + 1j * y[None, :]

    z = c.clone()
    divergence_map = torch.zeros(c.shape, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        z, mask = compute_tetration_step(c, z, escape_radius)
        divergence_map[mask] = True
        z[mask] = escape_radius

    return divergence_map.cpu().numpy()

def create_custom_colormap():
    return LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"])

def save_frame(divergence_map, px, py, scale, nx, ny, frame, output_folder, cmap):
    dpi = 100
    fig_width, fig_height = nx / dpi, ny / dpi

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    plt.imshow(divergence_map.T, extent=[px - scale, px + scale, py - scale * ASPECT_RATIO, py + scale * ASPECT_RATIO], origin='lower', cmap=cmap)
    plt.axis('on')
    file_path = os.path.join(output_folder, f'tetration_zoom_frame_{frame}.png')
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_frame_gpu(frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius, output_folder, cmap, device):
    start_time = time.time()
    gc.collect()  # 쓰레기 수집
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # 메모리 사용량 (MB)
    try:
        scale = scale_initial * (zoom_factor ** frame)
        px_frame = px + (px_target - px) * (1 - zoom_factor) ** frame
        py_frame = py + (py_target - py) * (1 - zoom_factor) ** frame

        divergence_map = compute_tetration_divergence_gpu(nx, ny, max_iter, escape_radius, px_frame, py_frame, scale, device)
        save_frame(divergence_map, px_frame, py_frame, scale, nx, ny, frame, output_folder, cmap)
    except Exception as e:
        print(f"Error processing frame {frame}: {e}")
        return None, None
    end_time = time.time()
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # 메모리 사용량 (MB)
    duration = end_time - start_time
    memory_used = end_mem - start_mem
    print(f"[GPU] Frame {frame} processed in {duration:.2f} seconds, using {memory_used:.2f} MB of memory")
    return duration, memory_used

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"폴더 생성했어요: {folder_path}")
    else:
        print(f"이 폴더에 저장할게요: {folder_path}")

if __name__ == '__main__':
    #################################
    ######### 초기 설정##############
    #################################

    px, py = 0, 0  # 시작 중심
    px_target, py_target = np.random.uniform(-1.0, 1.0), np.random.uniform(-5.0, 5.0)
    print(f"target: ({px_target}) , ({py_target})")  # 줌 타겟 중심
    scale_initial = 5  # 초기 스케일
    zoom_factor = 0.99  # 프레임 당 스케일 축소 비율
    nx, ny = 1920, 1080
    max_iter = 500
    escape_radius = 1e+10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 현재 작업 디렉토리 저장
    current_path = os.getcwd()
    print(f"현재 작업 디렉토리: {current_path}")

    # 원하는 폴더 경로 설정 (현재 작업 디렉토리를 기준으로 상대 경로를 사용)
    results_folder = f"results_{nx}x{ny}/Tetlas_{px_target}_{py_target}"
    output_folder = os.path.join(current_path, results_folder)

    # 폴더 생성 함수 호출
    create_folder_if_not_exists(output_folder)

    # 커스텀 컬러맵 생성
    cmap = create_custom_colormap()

    # 프레임 범위 설정
    start_frame = 1  # 시작 프레임
    end_frame = 121  # 종료 프레임(설정값 -1에서 끝남)

    # 초기 프레임 설정 (줌 전 프레임들)
    for frame in range(start_frame - 1):
        scale = scale_initial * (zoom_factor ** frame)
        px = px + (px_target - px) * (1 - zoom_factor)
        py = py + (py_target - py) * (1 - zoom_factor)

    # Dask 클라이언트 설정
    cluster = LocalCluster(n_workers=4, memory_limit='2GB')
    client = Client(cluster)

    try:
        # 애니메이션 프레임 생성 및 진행 상황 표시
        futures = []
        for frame in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
            futures.append(client.submit(process_frame_gpu, frame, px, py, px_target, py_target, scale_initial, zoom_factor, nx, ny, max_iter, escape_radius, output_folder, cmap, device))

        # 진행 상황 표시
        progress(futures)

        # 모든 작업 완료 대기
        results = client.gather(futures)

    finally:
        # 모든 작업 취소 및 클라이언트와 클러스터 종료
        client.close()
        cluster.close()
        gc.collect()  # 강제 garbage collection 실행
    print(f"target: ({px_target}) , ({py_target})")

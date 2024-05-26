import cv2
import os
import re

# 현재 작업 디렉토리 저장
current_path = os.path.dirname(os.path.abspath(__file__))
print(f"현재 디렉토리: {current_path}")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def images_to_video(image_folder, output_video, fps):
    try:
        # Get a list of images in the folder
        images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
        images.sort(key=natural_sort_key)  # Ensure the images are sorted in natural order

        if not images:
            raise ValueError("No images found in the folder")

        # Read the first image to get dimensions
        first_image_path = os.path.join(current_path, image_folder, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            raise ValueError(f"Could not read the first image: {first_image_path}")
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Skipping file {img_path}, as it could not be read.")
                continue
            video.write(frame)

        video.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {output_video}")

    except Exception as e:
        print(f"Error: {e}")
######################################################################################################
################### 설정 Parameters###################################################################
######################################################################################################



###이미지 폴더 이름을 붙여넣기 하세요. Copy and paste the image folder name.
image_folder = os.path.join(current_path, 'Tetlas_-0.9371381382814938_-3.5086305438969156') 
output_video = 'Tetlas_-0.9371381382814938_-3.5086305438969156.mp4'  # 출력할 동영상 파일 이름

fps = 30  # 초당 프레임 수 (fps 설정)
    
# Convert images to video
images_to_video(image_folder, output_video, fps)

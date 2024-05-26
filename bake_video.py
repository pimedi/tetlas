import cv2
import os
import re
import sys

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def images_to_video(image_folder, output_video, fps):
    try:
        # Get a list of images in the folder
        images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".png"))]
        images.sort(key=natural_sort_key)  # Ensure the images are sorted in natural order

        if not images:
            raise ValueError("No images found in the folder")

        # Read the first image to get dimensions
        first_image_path = os.path.join(image_folder, images[0])
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

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python bake_video.py <image_folder> <output_video> <fps>")
        sys.exit(1)

    image_folder = sys.argv[1]
    output_video = sys.argv[2]
    fps = int(sys.argv[3])

    images_to_video(image_folder, output_video, fps)

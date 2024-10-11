import os
import cv2
import shutil
from concurrent.futures import ProcessPoolExecutor

"""
python crop_yolo_parallel.py /path/to/images /path/to/labels /path/to/output 0 1 2 --num_workers 60
"""

def crop_single_image(image_name, image_folder, label_folder, output_folder, target_class_ids):
    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found for image {image_name}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image {image_name}")
        return

    h, w = image.shape[:2]

    # Read the label file
    with open(label_path, "r") as f:
        lines = f.readlines()

    object_count = 0
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        # Skip if the class is not in target_class_ids
        if class_id not in target_class_ids:
            continue

        # Extract bounding box information (YOLO format: x_center, y_center, width, height)
        x_center, y_center, box_width, box_height = map(float, parts[1:])
        x_center, y_center = x_center * w, y_center * h
        box_width, box_height = box_width * w, box_height * h

        # Calculate the bounding box coordinates
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Save the cropped image
        output_image_name = f"{os.path.splitext(image_name)[0]}_class_{class_id}_{object_count}.jpg"
        output_image_path = os.path.join(output_folder, output_image_name)
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Saved cropped image: {output_image_path}")

        object_count += 1


def crop_images_parallel(image_folder, label_folder, output_folder, target_class_ids, num_workers):
    # Ensure output folder exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # List all images in the folder
    image_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Use ProcessPoolExecutor to parallelize over multiple cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image_name in image_names:
            futures.append(executor.submit(crop_single_image, image_name, image_folder, label_folder, output_folder, target_class_ids))

        # Wait for all tasks to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Crop YOLO-formatted images based on specific classes in parallel.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("label_folder", type=str, help="Path to the folder containing label files (YOLO format)")
    parser.add_argument("output_folder", type=str, help="Path to the folder where cropped images will be saved")
    parser.add_argument("class_ids", nargs='+', type=int, help="List of class IDs to crop")
    parser.add_argument("--num_workers", type=int, default=60, help="Number of CPU cores to use for parallel processing")

    args = parser.parse_args()

    # Call the parallel crop function
    crop_images_parallel(args.image_folder, args.label_folder, args.output_folder, args.class_ids, args.num_workers)

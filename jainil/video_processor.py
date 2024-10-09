import cv2
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

# Function to load model in each subprocess
def initialize_model():
    return YOLO("/mnt/storage/kilsar_mohammad/train17/weights/best.pt")

def process_frame(frame):
    # Load the model in the subprocess
    model = initialize_model()
    # Run YOLOv8 inference on the frame
    results = model(frame)
    # Visualize the results on the frame
    return results[0].plot()

def process_video(file, count):
    cap = cv2.VideoCapture(file)
    assert cap.isOpened(), "Error reading video file"

    # Get video details (width, height, FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(f'/mnt/storage/kilsar_jainil/inferences/annotated_video_multiprocessed_{count}.mp4', fourcc, fps, (w, h))
    
    frames = []
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            break

    cap.release()

    # Use the 'spawn' start method
    with Pool(processes=cpu_count()) as pool:
        annotated_frames = pool.map(process_frame, frames)

    for frame in annotated_frames:
        output_video.write(frame)

    output_video.release()
    print("Processed", file)

import pickle
import cv2
import numpy as np
import argparse


def rgb_arrays_to_mp4(data, out_path, fps=30, greyscale=False):
    if greyscale:
        data = np.repeat(np.expand_dims(np.array(data), axis=-1), 3, axis=-1)

# Video properties
    frame_height, frame_width, _ = data[0].shape

# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' codec for MP4 videos
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Write frames to the video file
    for frame in data:
        # Ensure frame is in the correct format (uint8)
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 255)).astype(np.uint8)
        frame = frame[:, :, [2, 1, 0]]
        out.write(frame)

# Release the VideoWriter
    out.release()
    print(f"Video saved as {out_path}")

if __name__ == "__main__":
# Argument parser setup
    parser = argparse.ArgumentParser(description="Convert pickle frames to an MP4 video.")
    parser.add_argument("pkl_path", type=str, help="Path to the pickle file containing video frames")
    parser.add_argument("output", type=str, help="Output MP4 file name (e.g., output_video.mp4)")
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    rgb_arrays_to_mp4(data, args.output)

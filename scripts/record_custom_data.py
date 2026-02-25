"""
Record custom webcam skeleton data for training.

Usage:
  python scripts/record_custom_data.py --output data/custom
  python scripts/record_custom_data.py --output data/custom/wall_pushup --exercise "Wall push-up"

Press SPACE to start/stop recording. Each recording is saved as a .npy file.
Create labels.csv in the output dir with: file,score
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.pose_extraction import PoseExtractor

# Novel exercises (not in UI-PRMD, KIMORE, NTU)
CUSTOM_EXERCISES = [
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge",
    "clamshell", "chin_tuck", "marching_in_place", "step_up",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/custom")
    parser.add_argument("--exercise", type=str, default="", help="Exercise name (e.g. wall_pushup)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.output
    if args.exercise:
        out_dir = os.path.join(args.output, args.exercise)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    extractor = PoseExtractor()

    recording = False
    frames = []
    idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    ex_label = f" [{args.exercise}]" if args.exercise else ""

    print("SPACE: Start/Stop recording | Q: Quit" + ex_label)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame, landmarks = extractor.extract(frame)

        if landmarks is not None and recording:
            frames.append(landmarks.copy())

        status = "REC" if recording else "Ready"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {len(frames)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Record", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording and len(frames) > 0:
                path = os.path.join(out_dir, f"seq_{idx:03d}.npy")
                np.save(path, np.array(frames, dtype=np.float32))
                print(f"Saved {path} ({len(frames)} frames)")
                idx += 1
                frames = []
            recording = not recording
        elif key == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8n.pt"

VIDEO_PATH = Path("Cars_driving_at_night.webm")

OUTPUT_VIDEO = Path("traffic_annotated.mp4")
OUTPUT_CSV = Path("traffic_results.csv")
BENCHMARK_IMAGE = Path("traffic_benchmark_frame.jpg")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[INFO] Loading YOLOv8 model...")
    print(f"[INFO] Device: {device}")

    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU.")

    model = YOLO(MODEL_NAME)

    return model, device


# ============================================================
# 3. VEHICLE DETECTION
# ============================================================

def detect_vehicles(model, frame, device, confidence=0.35):

    results = model.predict(
        source=frame,
        conf=confidence,
        classes=list(VEHICLE_CLASSES.keys()),
        device=device,
        verbose=False
    )

    result = results[0]

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            if class_id in VEHICLE_CLASSES:

                vehicle_name = VEHICLE_CLASSES[class_id]
                counts[vehicle_name] += 1

    annotated_frame = result.plot()

    return annotated_frame, counts


# ============================================================
# 4. TRAFFIC DENSITY
# ============================================================

def estimate_traffic_density(total_vehicles):

    # Illustrative thresholds.
    # They must be calibrated for the camera and road.

    if total_vehicles <= 5:
        return "LOW"

    elif total_vehicles <= 15:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# 5. PROCESS VIDEO
# ============================================================

def process_video(video_path, model, device, max_frames=0):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}. "
            "Check the file path and video codec."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video.")

    records = []
    inference_times = []

    frame_number = 0
    benchmark_image_saved = False

    print("\n[INFO] Starting vehicle detection...")
    print("[INFO] Press Q to stop.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Measure prediction time
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            annotated_frame, counts = detect_vehicles(
                model,
                frame,
                device
            )

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            inference_ms = elapsed * 1000
            inference_times.append(inference_ms)

            # Save one original frame for ONNX benchmark
            if not benchmark_image_saved:
                cv2.imwrite(
                    str(BENCHMARK_IMAGE),
                    frame
                )
                benchmark_image_saved = True

            # Calculate traffic density
            total_vehicles = sum(counts.values())

            density = estimate_traffic_density(
                total_vehicles
            )

            # Display information
            information = [
                f"Frame: {frame_number}",
                f"Cars: {counts['car']}",
                f"Motorcycles: {counts['motorcycle']}",
                f"Buses: {counts['bus']}",
                f"Trucks: {counts['truck']}",
                f"Total vehicles: {total_vehicles}",
                f"Traffic density: {density}",
                f"Inference: {inference_ms:.2f} ms"
            ]

            y_position = 30

            for line in information:

                cv2.putText(
                    annotated_frame,
                    line,
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                y_position += 28

            # Save annotated frame
            writer.write(annotated_frame)

            # Store measurements
            records.append({
                "frame": frame_number,
                "cars": counts["car"],
                "motorcycles": counts["motorcycle"],
                "buses": counts["bus"],
                "trucks": counts["truck"],
                "total_vehicles": total_vehicles,
                "traffic_density": density,
                "inference_ms": inference_ms
            })

            # Display frame
            cv2.imshow(
                "Intelligent Traffic Monitoring",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if max_frames > 0 and frame_number >= max_frames:
                break

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Save CSV
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(OUTPUT_CSV, index=False)

    # Display final results
    print("\n========== PROCESSING RESULTS ==========")

    print(f"Processed frames: {frame_number}")

    if inference_times:

        print(
            f"Average inference time: "
            f"{np.mean(inference_times):.2f} ms"
        )

        print(
            f"P95 inference time: "
            f"{np.percentile(inference_times, 95):.2f} ms"
        )

        print(
            f"Maximum inference time: "
            f"{np.max(inference_times):.2f} ms"
        )

    print(f"\nAnnotated video: {OUTPUT_VIDEO}")
    print(f"CSV results: {OUTPUT_CSV}")
    print(f"Benchmark image: {BENCHMARK_IMAGE}")

    return dataframe


# ============================================================
# 6. EXPORT MODEL TO ONNX
# ============================================================

def export_onnx(model):

    print("\n[INFO] Exporting model to ONNX...")

    try:

        exported_path = model.export(
            format="onnx",
            imgsz=640,
            simplify=False
        )

        print("[SUCCESS] ONNX export completed.")
        print(f"Model: {exported_path}")

        return str(exported_path)

    except Exception as error:

        print(f"[ERROR] ONNX export failed: {error}")

        return None


# ============================================================
# 7. BENCHMARK MODEL
# ============================================================

def benchmark_model(model_path, image_path, runs=30):

    print(f"\n[INFO] Benchmarking: {model_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("[ERROR] Benchmark image not found.")
        return

    try:

        benchmark_model = YOLO(model_path)

        # Warm-up
        for _ in range(5):

            benchmark_model.predict(
                source=image,
                imgsz=640,
                device="cpu",
                verbose=False
            )

        times = []

        # Actual measurements
        for _ in range(runs):

            start = time.perf_counter()

            benchmark_model.predict(
                source=image,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)

        print("\n========== BENCHMARK RESULTS ==========")

        print(f"Model: {model_path}")
        print(f"Runs: {runs}")

        print(
            f"Average inference: "
            f"{np.mean(times):.2f} ms"
        )

        print(
            f"P95 inference: "
            f"{np.percentile(times, 95):.2f} ms"
        )

        print(
            f"Maximum inference: "
            f"{np.max(times):.2f} ms"
        )

    except Exception as error:

        print(f"[ERROR] Benchmark failed: {error}")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Intelligent Traffic Monitoring"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(VIDEO_PATH),
        help="Path to the traffic video"
    )

    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export YOLO model to ONNX"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Export and benchmark ONNX model"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of benchmark iterations"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames; 0 = all frames"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("INTELLIGENT TRAFFIC MONITORING")
    print("Vehicle Detection and Model Optimization")
    print("=" * 55)

    video_path = Path(args.video)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            "Place Cars_driving_at_night.webm "
            "in the project folder or use --video."
        )

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    # Load model
    model, device = load_model()

    # Process video
    process_video(
        video_path,
        model,
        device,
        max_frames=args.max_frames
    )

    # Export ONNX if requested
    onnx_path = None

    if args.export_onnx or args.benchmark:

        onnx_path = export_onnx(model)

    # Benchmark ONNX if requested
    if args.benchmark and onnx_path:

        benchmark_model(
            onnx_path,
            BENCHMARK_IMAGE,
            runs=args.runs
        )

    print("\n[FINISHED] TP completed successfully.")


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
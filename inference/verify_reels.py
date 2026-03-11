"""
Inference pipeline: verifies campaign assets in reel videos.

Downloads reel, extracts frames, runs YOLO detection, checks logo presence/size/persistence.
"""

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_inference_frames(video_path: str, num_frames: int = 5) -> list[np.ndarray]:
    """Extract evenly spaced frames from a video for inference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = [int((i + 1) / (num_frames + 1) * total) for i in range(num_frames)]
    indices = [min(idx, total - 1) for idx in indices]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def verify_reel(
    video_path: str,
    model,
    class_names: dict,
    num_frames: int = 5,
    conf_threshold: float = 0.25,
    min_height_ratio: float = 0.05,
    min_frames_detected: int = 2,
) -> dict:
    """Run verification on a single reel video.

    Args:
        video_path: Path to video file.
        model: Loaded YOLO model.
        class_names: {class_id: name} mapping.
        num_frames: Frames to sample.
        conf_threshold: Detection confidence threshold.
        min_height_ratio: Minimum logo height as fraction of frame height.
        min_frames_detected: Minimum frames where logo must appear.

    Returns:
        Verification result dict.
    """
    frames = extract_inference_frames(video_path, num_frames)
    if not frames:
        return {
            "video_processed": False,
            "error": "Could not extract frames",
            "frames_analyzed": 0,
            "assets_detected": [],
        }

    # Track detections per class
    detections_per_class = {}  # class_id -> list of (frame_idx, height_ratio, conf)

    for frame_idx, frame in enumerate(frames):
        results = model(frame, conf=conf_threshold, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Get bbox in pixel coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_h = y2 - y1
                frame_h = frame.shape[0]
                height_ratio = bbox_h / frame_h

                if cls_id not in detections_per_class:
                    detections_per_class[cls_id] = []
                detections_per_class[cls_id].append({
                    "frame": frame_idx,
                    "height_ratio": height_ratio,
                    "confidence": conf,
                })

    # Build results
    assets_detected = []
    for cls_id, dets in detections_per_class.items():
        name = class_names.get(cls_id, f"class_{cls_id}")
        frames_detected = len(set(d["frame"] for d in dets))
        avg_height = np.mean([d["height_ratio"] for d in dets])
        avg_conf = np.mean([d["confidence"] for d in dets])

        persistent = frames_detected >= min_frames_detected
        size_ok = avg_height >= min_height_ratio

        if persistent and size_ok:
            status = "accepted"
        elif not persistent:
            status = "rejected_insufficient_frames"
        else:
            status = "rejected_too_small"

        assets_detected.append({
            "asset_name": name,
            "class_id": cls_id,
            "frames_detected": frames_detected,
            "total_frames": len(frames),
            "persistent": persistent,
            "average_height_ratio": round(avg_height, 4),
            "average_confidence": round(avg_conf, 4),
            "status": status,
        })

    return {
        "video_processed": True,
        "frames_analyzed": len(frames),
        "assets_detected": assets_detected,
    }


def verify_reels_from_file(
    reels_file: str,
    model_path: str,
    output_path: str = "verification_results.json",
    num_frames: int = 5,
    conf_threshold: float = 0.25,
    min_height_ratio: float = 0.05,
    min_frames_detected: int = 2,
) -> list[dict]:
    """Verify all reels listed in a text file.

    Args:
        reels_file: Path to text file with reel URLs/paths.
        model_path: Path to trained YOLO model weights.
        output_path: Path to save JSON results.
        num_frames: Frames to analyze per video.
        conf_threshold: Detection confidence threshold.
        min_height_ratio: Min logo height ratio.
        min_frames_detected: Min frames for persistence.

    Returns:
        List of verification results.
    """
    from ultralytics import YOLO
    from dataset_generator.frame_extractor import download_video

    model = YOLO(model_path)

    # Extract class names from model
    class_names = model.names if hasattr(model, "names") else {}

    # Read URLs
    urls = []
    with open(reels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        print("[WARN] No URLs found in reels file.")
        return []

    tmp_dir = tempfile.mkdtemp(prefix="clipstake_verify_")
    results = []

    for url in urls:
        print(f"[INFO] Processing: {url}")

        # Check if it's a local file path
        if os.path.isfile(url):
            video_path = url
        else:
            video_path = download_video(url, tmp_dir)

        if video_path is None:
            results.append({
                "url": url,
                "video_processed": False,
                "error": "Download failed",
                "frames_analyzed": 0,
                "assets_detected": [],
            })
            continue

        result = verify_reel(
            video_path, model, class_names,
            num_frames=num_frames,
            conf_threshold=conf_threshold,
            min_height_ratio=min_height_ratio,
            min_frames_detected=min_frames_detected,
        )
        result["url"] = url
        results.append(result)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Verification results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify campaign assets in reel videos")
    parser.add_argument("--reels", type=str, required=True, help="Path to reels.txt")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (best.pt)")
    parser.add_argument("--output", type=str, default="verification_results.json", help="Output JSON path")
    parser.add_argument("--frames", type=int, default=5, help="Frames to analyze per video")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--min-height", type=float, default=0.05, help="Min logo height ratio")
    parser.add_argument("--min-frames", type=int, default=2, help="Min frames for persistence check")
    args = parser.parse_args()

    results = verify_reels_from_file(
        reels_file=args.reels,
        model_path=args.model,
        output_path=args.output,
        num_frames=args.frames,
        conf_threshold=args.conf,
        min_height_ratio=args.min_height,
        min_frames_detected=args.min_frames,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for r in results:
        url = r.get("url", "unknown")
        print(f"\nReel: {url}")
        if not r["video_processed"]:
            print(f"  Status: FAILED - {r.get('error', 'unknown error')}")
            continue
        print(f"  Frames analyzed: {r['frames_analyzed']}")
        for asset in r["assets_detected"]:
            status_icon = "PASS" if asset["status"] == "accepted" else "FAIL"
            print(f"  [{status_icon}] {asset['asset_name']}: "
                  f"detected in {asset['frames_detected']}/{r['frames_analyzed']} frames, "
                  f"avg height ratio={asset['average_height_ratio']:.3f}, "
                  f"conf={asset['average_confidence']:.3f}")
        if not r["assets_detected"]:
            print("  [FAIL] No campaign assets detected")


if __name__ == "__main__":
    main()

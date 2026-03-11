"""
Frame extractor: downloads reel videos and extracts evenly-spaced frames.
"""

import os
import subprocess
import cv2
import tempfile
from pathlib import Path
from tqdm import tqdm


def download_video(url: str, output_dir: str) -> str | None:
    """Download a video using yt-dlp and return the local file path."""
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[ext=mp4]/best",
        "-o", output_template,
        "--no-overwrites",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [WARN] yt-dlp failed for {url}: {result.stderr.strip()[:200]}")
            return None
        # Find the downloaded file from yt-dlp output
        for line in result.stdout.splitlines():
            if "Destination:" in line:
                return line.split("Destination:")[-1].strip()
            if "has already been downloaded" in line:
                # Extract path from "already downloaded" message
                path = line.split("[download]")[-1].strip().split(" has already")[0].strip()
                return path
        # Fallback: find newest file in output_dir
        files = sorted(Path(output_dir).glob("*.*"), key=os.path.getmtime, reverse=True)
        if files:
            return str(files[0])
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Download timed out for {url}")
    except Exception as e:
        print(f"  [WARN] Download error for {url}: {e}")
    return None


def extract_frames(video_path: str, num_frames: int = 20) -> list:
    """Extract evenly spaced frames from a video file.

    Returns list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    # Sample at 5%, 10%, ..., 95% of video length
    percentages = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
    target_indices = [int(p * total_frames) for p in percentages]
    target_indices = [min(idx, total_frames - 1) for idx in target_indices]

    frames = []
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def extract_frames_from_reels(
    reels_file: str,
    output_dir: str,
    frames_per_reel: int = 20,
    video_cache_dir: str | None = None,
) -> list[str]:
    """Download reels and extract frames, saving them as JPGs.

    Args:
        reels_file: Path to text file with one URL per line.
        output_dir: Directory to save extracted frame images.
        frames_per_reel: Number of frames to extract per video.
        video_cache_dir: Directory to cache downloaded videos.

    Returns:
        List of saved frame image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if video_cache_dir is None:
        video_cache_dir = os.path.join(output_dir, "_video_cache")
    os.makedirs(video_cache_dir, exist_ok=True)

    urls = []
    with open(reels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        print("[INFO] No URLs found in reels file. Using placeholder backgrounds.")
        return []

    saved_paths = []
    for i, url in enumerate(tqdm(urls, desc="Downloading reels")):
        video_path = download_video(url, video_cache_dir)
        if video_path is None:
            continue

        frames = extract_frames(video_path, frames_per_reel)
        for j, frame in enumerate(frames):
            fname = f"reel_{i:04d}_frame_{j:03d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(fpath)

    print(f"[INFO] Extracted {len(saved_paths)} frames from {len(urls)} reels.")
    return saved_paths


def generate_placeholder_backgrounds(output_dir: str, count: int = 100) -> list[str]:
    """Generate synthetic placeholder backgrounds when no reels are available.

    Creates varied backgrounds: gradients, solid colors, noise patterns.
    """
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    rng = np.random.default_rng(42)

    for i in range(count):
        h, w = 1080, 1920  # Landscape HD
        if rng.random() < 0.5:
            h, w = 1920, 1080  # Portrait (reel format)

        choice = rng.random()
        if choice < 0.3:
            # Gradient
            base_color = rng.integers(30, 200, size=3)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for row in range(h):
                factor = row / h
                img[row, :] = (base_color * (1 - factor * 0.5)).astype(np.uint8)
        elif choice < 0.6:
            # Solid with noise
            base = rng.integers(20, 220, size=3)
            img = np.full((h, w, 3), base, dtype=np.uint8)
            noise = rng.integers(-20, 20, size=(h, w, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # Random scene-like pattern
            img = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.GaussianBlur(img, (31, 31), 10)

        fpath = os.path.join(output_dir, f"bg_{i:04d}.jpg")
        cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(fpath)

    print(f"[INFO] Generated {len(saved)} placeholder backgrounds.")
    return saved

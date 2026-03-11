"""
Logo compositor: composites logo assets onto background frames with augmentations.
Generates YOLO-format labels and metadata JSON files.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm

from .augmentations import AugmentationSampler


def load_logo(logo_path: str) -> Image.Image:
    """Load a logo with alpha channel. Converts to RGBA if needed."""
    img = Image.open(logo_path).convert("RGBA")
    return img


def apply_color_jitter(logo: Image.Image, jitter: dict) -> Image.Image:
    """Apply color jitter to the RGB channels of a logo."""
    # Work on RGB, preserve alpha
    r, g, b, a = logo.split()
    rgb = Image.merge("RGB", (r, g, b))

    if jitter["brightness"] != 0:
        rgb = ImageEnhance.Brightness(rgb).enhance(1.0 + jitter["brightness"])
    if jitter["contrast"] != 0:
        rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + jitter["contrast"])
    if jitter["saturation"] != 0:
        rgb = ImageEnhance.Color(rgb).enhance(1.0 + jitter["saturation"])

    # Hue shift via numpy
    if abs(jitter["hue"]) > 0.5:
        arr = np.array(rgb).astype(np.float32)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + jitter["hue"]) % 180
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    r2, g2, b2 = rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, a))


def composite_logo_onto_background(
    bg_img: np.ndarray,
    logo: Image.Image,
    cx_norm: float,
    cy_norm: float,
    scale: float,
    rotation: float,
    opacity: float,
    aspect_distortion: float,
    color_jitter: dict,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Composite a logo onto a background image.

    Args:
        bg_img: BGR background (numpy array).
        logo: RGBA PIL image.
        cx_norm, cy_norm: Normalized center position (0-1).
        scale: Logo height as fraction of frame height.
        rotation: Degrees to rotate.
        opacity: 0-1 opacity.
        aspect_distortion: Fraction to distort width.
        color_jitter: Dict with brightness/contrast/saturation/hue.

    Returns:
        (composited_bgr_image, (cx, cy, w, h) normalized bbox or None if fully clipped)
    """
    h_frame, w_frame = bg_img.shape[:2]

    # Target logo size
    logo_h = int(scale * h_frame)
    orig_w, orig_h = logo.size
    aspect = orig_w / max(orig_h, 1)
    logo_w = int(logo_h * aspect)

    # Apply aspect distortion
    if aspect_distortion > 0:
        if np.random.random() > 0.5:
            logo_w = int(logo_w * (1 + aspect_distortion))
        else:
            logo_w = int(logo_w * (1 - aspect_distortion))

    logo_w = max(logo_w, 4)
    logo_h = max(logo_h, 4)

    # Resize logo
    resized = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # Apply color jitter
    resized = apply_color_jitter(resized, color_jitter)

    # Apply rotation
    if abs(rotation) > 0.1:
        resized = resized.rotate(-rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    rw, rh = resized.size

    # Compute pixel position for logo center
    cx_px = int(cx_norm * w_frame)
    cy_px = int(cy_norm * h_frame)

    # Top-left corner of logo
    x1 = cx_px - rw // 2
    y1 = cy_px - rh // 2
    x2 = x1 + rw
    y2 = y1 + rh

    # Compute visible region (clip to frame)
    vis_x1 = max(x1, 0)
    vis_y1 = max(y1, 0)
    vis_x2 = min(x2, w_frame)
    vis_y2 = min(y2, h_frame)

    if vis_x2 <= vis_x1 or vis_y2 <= vis_y1:
        return bg_img, None  # Fully outside frame

    # Check 70% visibility rule
    total_area = rw * rh
    visible_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
    if visible_area / max(total_area, 1) < 0.70:
        # For non-clipped-edge images, skip if too little visible
        return bg_img, None

    # Crop the logo region that's visible
    logo_crop_x1 = vis_x1 - x1
    logo_crop_y1 = vis_y1 - y1
    logo_crop_x2 = logo_crop_x1 + (vis_x2 - vis_x1)
    logo_crop_y2 = logo_crop_y1 + (vis_y2 - vis_y1)

    logo_crop = resized.crop((logo_crop_x1, logo_crop_y1, logo_crop_x2, logo_crop_y2))

    # Apply opacity
    r_ch, g_ch, b_ch, a_ch = logo_crop.split()
    a_ch = a_ch.point(lambda p: int(p * opacity))
    logo_crop = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))

    # Convert background region to PIL for compositing
    bg_pil = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))

    # Paste logo onto background
    paste_region = Image.new("RGBA", bg_pil.size, (0, 0, 0, 0))
    paste_region.paste(logo_crop, (vis_x1, vis_y1))
    composited = Image.alpha_composite(bg_pil.convert("RGBA"), paste_region)
    result = cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Compute YOLO bbox (of visible portion)
    bbox_cx = (vis_x1 + vis_x2) / 2.0 / w_frame
    bbox_cy = (vis_y1 + vis_y2) / 2.0 / h_frame
    bbox_w = (vis_x2 - vis_x1) / w_frame
    bbox_h = (vis_y2 - vis_y1) / h_frame

    return result, (bbox_cx, bbox_cy, bbox_w, bbox_h)


def apply_global_augmentations(
    img: np.ndarray,
    blur_sigma: float,
    noise_level: float,
    jpeg_quality: int,
) -> np.ndarray:
    """Apply blur, noise, and JPEG compression to the final image."""
    # Gaussian blur
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1  # Ensure odd
        ksize = max(ksize, 3)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    # Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG compression (encode then decode to simulate)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    _, encoded = cv2.imencode(".jpg", img, encode_params)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def generate_dataset(
    assets_dir: str,
    backgrounds_dir: str,
    output_dir: str,
    metadata_dir: str,
    total_images: int = 3000,
    seed: int = 42,
):
    """Generate a complete synthetic YOLO dataset.

    Args:
        assets_dir: Directory containing logo PNG files.
        backgrounds_dir: Directory containing background frame images.
        output_dir: Root dataset directory (will contain images/ and labels/).
        metadata_dir: Directory for metadata JSON files.
        total_images: Total number of images to generate.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    sampler = AugmentationSampler(rng)

    # Load logos
    logo_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    logo_files = sorted([
        f for f in Path(assets_dir).iterdir()
        if f.suffix.lower() in logo_extensions
    ])
    if not logo_files:
        raise ValueError(f"No logo files found in {assets_dir}")

    class_names = {i: f.stem for i, f in enumerate(logo_files)}
    logos = {i: load_logo(str(f)) for i, f in enumerate(logo_files)}
    num_classes = len(logos)
    print(f"[INFO] Loaded {num_classes} logo classes: {class_names}")

    # Load backgrounds
    bg_files = sorted([
        str(f) for f in Path(backgrounds_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not bg_files:
        raise ValueError(f"No background images found in {backgrounds_dir}")
    print(f"[INFO] Loaded {len(bg_files)} background images.")

    # Create output dirs
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Train/val split: 70/30
    train_count = int(total_images * 0.70)

    # Generate dataset.yaml
    yaml_path = os.path.join(output_dir, "..", "dataset.yaml")
    _write_dataset_yaml(yaml_path, output_dir, class_names)

    print(f"[INFO] Generating {total_images} images ({train_count} train, {total_images - train_count} val)...")

    for idx in tqdm(range(total_images), desc="Generating dataset"):
        split = "train" if idx < train_count else "val"
        img_name = f"image_{idx:05d}.jpg"
        label_name = f"image_{idx:05d}.txt"
        meta_name = f"image_{idx:05d}.json"

        # Load random background
        bg_path = rng.choice(bg_files)
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        # Resize background to consistent size
        bg_img = _resize_bg(bg_img, target_h=1080)

        num_logos = sampler.sample_num_logos()
        jpeg_quality = sampler.sample_jpeg_quality()
        blur_sigma = sampler.sample_blur_sigma()
        noise_level = sampler.sample_noise_level()

        labels = []
        meta_entries = []

        for logo_idx in range(num_logos):
            # Select logo class (try to balance classes)
            class_id = int(rng.integers(0, num_classes))
            logo = logos[class_id]

            scale = sampler.sample_scale()
            opacity = sampler.sample_opacity()
            rotation = sampler.sample_rotation()
            aspect_dist = sampler.sample_aspect_distortion()
            color_jitter = sampler.sample_color_jitter()
            placement_type = sampler.sample_placement_type()
            cx, cy = sampler.sample_placement_position(placement_type)

            # Allow edge clipping for ~5% of images
            allow_clip = sampler.should_clip_edge()
            if not allow_clip:
                # Ensure at least 70% of logo stays in frame
                h_frame, w_frame = bg_img.shape[:2]
                logo_h_px = scale * h_frame
                logo_w_px = logo_h_px * (logo.size[0] / max(logo.size[1], 1))
                half_w = (logo_w_px / 2) / w_frame
                half_h = (logo_h_px / 2) / h_frame
                # Clamp so logo center is far enough from edges
                margin_w = half_w * 0.7
                margin_h = half_h * 0.7
                cx = np.clip(cx, margin_w, 1 - margin_w)
                cy = np.clip(cy, margin_h, 1 - margin_h)

            bg_img, bbox = composite_logo_onto_background(
                bg_img, logo, cx, cy, scale, rotation, opacity,
                aspect_dist, color_jitter,
            )

            if bbox is not None:
                labels.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                meta_entries.append({
                    "logo": logo_files[class_id].name,
                    "class_id": class_id,
                    "placement_type": placement_type,
                    "rotation": round(rotation, 2),
                    "opacity": round(opacity, 3),
                    "scale_ratio": round(scale, 3),
                    "center_x": round(bbox[0], 4),
                    "center_y": round(bbox[1], 4),
                    "bbox_w": round(bbox[2], 4),
                    "bbox_h": round(bbox[3], 4),
                    "aspect_ratio_distortion": round(aspect_dist, 3),
                })

        # Apply global augmentations
        bg_img = apply_global_augmentations(bg_img, blur_sigma, noise_level, jpeg_quality)

        # Save image
        img_path = os.path.join(output_dir, "images", split, img_name)
        cv2.imwrite(img_path, bg_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Save YOLO label (empty file for negatives)
        label_path = os.path.join(output_dir, "labels", split, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        # Save metadata
        metadata = {
            "image": img_name,
            "split": split,
            "logos_in_image": num_logos,
            "jpeg_quality": jpeg_quality,
            "blur_sigma": round(blur_sigma, 2),
            "noise_level": round(noise_level, 1),
            "detections": meta_entries,
        }
        meta_path = os.path.join(metadata_dir, meta_name)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"[INFO] Dataset generation complete. Output: {output_dir}")
    print(f"[INFO] Metadata saved to: {metadata_dir}")


def _resize_bg(img: np.ndarray, target_h: int = 1080) -> np.ndarray:
    """Resize background maintaining aspect ratio to target height."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _write_dataset_yaml(yaml_path: str, dataset_dir: str, class_names: dict):
    """Write YOLO dataset.yaml config."""
    abs_dataset = os.path.abspath(dataset_dir)
    lines = [
        f"path: {abs_dataset}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx in sorted(class_names.keys()):
        lines.append(f"  {idx}: {class_names[idx]}")
    lines.append("")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrote dataset config: {yaml_path}")

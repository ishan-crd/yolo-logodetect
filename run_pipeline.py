#!/usr/bin/env python3
"""
Clipstake Logo Detection Pipeline - Main Entry Point

Usage:
    # Full pipeline: generate dataset, train, verify
    python run_pipeline.py --assets assets/ --reels reels.txt

    # Generate dataset only
    python run_pipeline.py --assets assets/ --reels reels.txt --stage dataset

    # Train only (requires existing dataset)
    python run_pipeline.py --stage train

    # Verify only (requires trained model)
    python run_pipeline.py --reels reels.txt --model runs/train/clipstake_logos/weights/best.pt --stage verify
"""

import argparse
import os
import sys
from pathlib import Path


def stage_dataset(args):
    """Generate synthetic dataset."""
    from dataset_generator.frame_extractor import (
        extract_frames_from_reels,
        generate_placeholder_backgrounds,
    )
    from dataset_generator.logo_compositor import generate_dataset

    print("=" * 60)
    print("STAGE 1: DATASET GENERATION")
    print("=" * 60)

    backgrounds_dir = os.path.join("dataset", "_backgrounds")

    # Extract frames from reels (or generate placeholders)
    if args.reels and os.path.exists(args.reels):
        frame_paths = extract_frames_from_reels(
            args.reels, backgrounds_dir, frames_per_reel=20
        )
        # If no frames were extracted (no valid URLs), generate placeholders
        if not frame_paths:
            print("[INFO] No reel frames extracted. Generating placeholder backgrounds.")
            generate_placeholder_backgrounds(backgrounds_dir, count=150)
    else:
        print("[INFO] No reels file provided. Generating placeholder backgrounds.")
        generate_placeholder_backgrounds(backgrounds_dir, count=150)

    # Generate synthetic dataset
    generate_dataset(
        assets_dir=args.assets,
        backgrounds_dir=backgrounds_dir,
        output_dir="dataset",
        metadata_dir="metadata",
        total_images=args.total_images,
        seed=args.seed,
    )

    print("[INFO] Dataset generation complete.\n")


def stage_train(args):
    """Train YOLO model."""
    print("=" * 60)
    print("STAGE 2: YOLO TRAINING")
    print("=" * 60)

    data_path = os.path.abspath("dataset.yaml")
    if not os.path.exists(data_path):
        print(f"[ERROR] dataset.yaml not found at {data_path}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name="clipstake_logos",
        workers=args.workers,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = "runs/train/clipstake_logos/weights/best.pt"
    print(f"[INFO] Training complete. Best model: {best_path}\n")
    return best_path


def stage_verify(args, model_path: str | None = None):
    """Run verification inference."""
    print("=" * 60)
    print("STAGE 3: REEL VERIFICATION")
    print("=" * 60)

    if model_path is None:
        model_path = args.model
    if model_path is None:
        model_path = "runs/train/clipstake_logos/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if not args.reels or not os.path.exists(args.reels):
        print("[ERROR] Reels file required for verification.")
        sys.exit(1)

    from inference.verify_reels import verify_reels_from_file

    results = verify_reels_from_file(
        reels_file=args.reels,
        model_path=model_path,
        output_path=args.output,
        num_frames=5,
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
                  f"avg height={asset['average_height_ratio']:.3f}")
        if not r["assets_detected"]:
            print("  [FAIL] No campaign assets detected")


def main():
    parser = argparse.ArgumentParser(
        description="Clipstake Logo Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # General
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "dataset", "train", "verify"],
                        help="Pipeline stage to run (default: all)")
    parser.add_argument("--assets", type=str, default="assets",
                        help="Path to campaign assets directory")
    parser.add_argument("--reels", type=str, default="reels.txt",
                        help="Path to reels URL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dataset
    parser.add_argument("--total-images", type=int, default=3000,
                        help="Total synthetic images to generate")

    # Training
    parser.add_argument("--base-model", type=str, default="yolo11n.pt",
                        help="Base YOLO model for training")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--device", type=str, default="mps",
                        help="Training device (mps for M1, cpu, cuda)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")

    # Verification
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model for verification")
    parser.add_argument("--output", type=str, default="verification_results.json",
                        help="Verification output JSON path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--min-height", type=float, default=0.05,
                        help="Min logo height ratio for acceptance")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="Min frames where logo must appear")

    args = parser.parse_args()

    if args.stage in ("all", "dataset"):
        stage_dataset(args)

    model_path = None
    if args.stage in ("all", "train"):
        model_path = stage_train(args)

    if args.stage in ("all", "verify"):
        stage_verify(args, model_path)


if __name__ == "__main__":
    main()

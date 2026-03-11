"""
YOLO training script optimized for MacBook M1 Pro with Apple Metal (MPS).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for logo detection")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="mps", help="Device: mps, cpu, or cuda")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="clipstake_logos", help="Experiment name")
    args = parser.parse_args()

    # Resolve dataset.yaml path
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    print(f"[INFO] Training configuration:")
    print(f"  Model:   {args.model}")
    print(f"  Data:    {data_path}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device}")
    print(f"  Workers: {args.workers}")
    print()

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Best model: {args.project}/{args.name}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()

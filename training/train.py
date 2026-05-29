"""
Training — Step 2: Fine-tune YOLOv8 on custom motorbike dataset.

Expected dataset layout (YOLO format):
  data/
    images/
      train/   *.jpg
      val/     *.jpg
    labels/
      train/   *.txt  (class x_center y_center w h, normalised)
      val/     *.txt
    dataset.yaml

Usage:
  python training/train.py --data data/dataset.yaml --model yolov8n.pt \
         --epochs 50 --imgsz 640 --batch 16 --device 0
"""
from __future__ import annotations
import argparse
import os


def write_dataset_yaml(images_dir: str, out_path: str = "data/dataset.yaml",
                       class_names: list = None) -> str:
    if class_names is None:
        class_names = ["motorbike"]

    import yaml
    data = {
        "path":  os.path.abspath(images_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Dataset YAML written: {out_path}")
    return out_path


def train(data_yaml: str, model: str = "yolov8n.pt",
          epochs: int = 50, imgsz: int = 640,
          batch: int = 16, device: str = "0",
          project: str = "runs/train", name: str = "parking") -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    yolo = YOLO(model)
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=15,
        save=True,
        plots=True,
        # Augmentation settings tuned for parking lot conditions
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.4,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
    )

    best_weights = os.path.join(project, name, "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/dataset.yaml")
    p.add_argument("--model",   default="yolov8n.pt")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name",    default="parking_motorbike")
    args = p.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz,
          args.batch, args.device, args.project, args.name)


if __name__ == "__main__":
    main()

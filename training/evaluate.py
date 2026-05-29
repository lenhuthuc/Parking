"""
Training — Step 3: Evaluate model mAP and slot-detection accuracy.

Usage:
  python training/evaluate.py --model runs/train/parking_motorbike/weights/best.pt \
         --data data/dataset.yaml
"""
from __future__ import annotations
import argparse
import json
import os


def evaluate_yolo(model_path: str, data_yaml: str,
                  imgsz: int = 640, device: str = "cpu") -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device,
                        conf=0.25, iou=0.45, plots=True)

    results = {
        "mAP50":     round(metrics.box.map50, 4),
        "mAP50_95":  round(metrics.box.map,   4),
        "precision": round(metrics.box.mp,    4),
        "recall":    round(metrics.box.mr,    4),
    }
    print("\n── Evaluation results ──────────────────────")
    for k, v in results.items():
        bar = "█" * int(v * 20)
        print(f"  {k:12s}: {v:.4f}  {bar}")
    print("────────────────────────────────────────────")

    target = 0.85
    passed = results["mAP50"] >= target
    print(f"\n  R1 requirement (mAP@0.5 ≥ {target}): "
          f"{'✓ PASS' if passed else '✗ FAIL'}")

    return results


def evaluate_slots(model_path: str, video_path: str,
                   layout_path: str, sample_fps: float = 2.0) -> None:
    """
    Run the full pipeline on a labelled video and print per-slot accuracy.
    Ground-truth labels should be in a JSON file: video_path + ".gt.json"
    Format: { "frame_idx": { "slot_id": "TRỐNG" | "CÓ XE" | "KHÔNG XÁC ĐỊNH" } }
    """
    gt_path = video_path + ".gt.json"
    if not os.path.exists(gt_path):
        print(f"No ground-truth file found at {gt_path}. Skipping slot eval.")
        return

    with open(gt_path) as f:
        ground_truth = json.load(f)

    from preprocessor import Preprocessor, FrameSampler
    from detector import Detector
    from layout import Layout
    from logic import ParkingLogic

    layout   = Layout.load(layout_path)
    prep     = Preprocessor()
    detector = Detector(model_path=model_path, device="cpu")
    logic    = ParkingLogic(layout, sample_fps=sample_fps)

    sampler  = FrameSampler(video_path, target_fps=sample_fps)
    correct  = total = 0

    try:
        for frame_idx, frame in enumerate(sampler):
            t, s, p = prep.preprocess(frame)
            dets     = detector.detect(t, s, p)
            states, _ = logic.process_frame(dets)

            gt_frame = ground_truth.get(str(frame_idx))
            if gt_frame is None:
                continue

            for i, slot in enumerate(layout.slots):
                gt = gt_frame.get(slot.id)
                if gt is None:
                    continue
                total += 1
                if states[i].value == gt:
                    correct += 1
    finally:
        sampler.release()

    acc = correct / total if total > 0 else 0
    print(f"\nSlot state accuracy: {correct}/{total} = {acc:.1%}")
    print(f"R2 requirement (≥ 90%): {'✓ PASS' if acc >= 0.90 else '✗ FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--data",   default="data/dataset.yaml")
    p.add_argument("--video",  default=None,
                   help="Video file for slot-level evaluation")
    p.add_argument("--layout", default="layout.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    evaluate_yolo(args.model, args.data, device=args.device)
    if args.video:
        evaluate_slots(args.model, args.video, args.layout)


if __name__ == "__main__":
    main()

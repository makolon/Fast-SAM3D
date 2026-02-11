import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_SAM3 = PROJECT_ROOT / "third_party" / "sam3"
if str(THIRD_PARTY_SAM3) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_SAM3))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _to_mask_array(masks: torch.Tensor) -> np.ndarray:
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for masks, got: {type(masks)}")

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {masks_np.shape}")
    return masks_np.astype(bool)


def run_segmentation(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    device: str,
    confidence_threshold: float,
    checkpoint_path: str | None,
) -> int:
    image = Image.open(image_path).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = _to_mask_array(output["masks"])
    scores = output["scores"].detach().cpu().numpy()

    if masks.shape[0] == 0:
        print("No objects were detected for the prompt.")
        return 0

    ext = image_path.suffix if image_path.suffix else ".png"
    stem = image_path.stem

    saved = 0
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        if not np.any(mask):
            continue
        out = (mask.astype(np.uint8) * 255)
        out_path = output_dir / f"{stem}_object_{idx:03d}{ext}"
        Image.fromarray(out, mode="L").save(out_path)
        print(f"saved: {out_path} (score={float(score):.4f})")
        saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment objects from an RGB image with SAM3 and save one mask per object."
    )
    parser.add_argument("--image_path", type=str, required=True, help="Input RGB image path.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for object query.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="notebook/datasets/masks",
        help="Directory to save per-object mask images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cuda or cpu.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="SAM3 score threshold for mask filtering.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional local SAM3 checkpoint path. If omitted, downloads from HF.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir)
    saved = run_segmentation(
        image_path=image_path,
        prompt=args.prompt,
        output_dir=output_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        checkpoint_path=args.checkpoint_path,
    )
    print(f"done: saved {saved} mask(s) to {output_dir}")


if __name__ == "__main__":
    main()

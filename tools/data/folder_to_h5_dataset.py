"""Convert an image folder into an H5 dataset keyed by relative paths.

Example:
    python tools/data/folder_to_h5_dataset.py \
        --image_root /path/to/frames \
        --output_h5 data/h5/test.h5

If `/path/to/frames/UADFV/fake/0000_fake/000.png` exists, the H5 key will be:
`UADFV/fake/0000_fake/000.png`.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert image folders to an H5 dataset.")
    parser.add_argument("--image_root", type=Path, required=True, help="Root folder containing images.")
    parser.add_argument("--output_h5", type=Path, required=True, help="Output H5 file path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def iter_images(image_root: Path):
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_h5(image_root: Path, output_h5: Path, overwrite: bool = False) -> None:
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if output_h5.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_h5}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(image_root))

    with h5py.File(output_h5, "w") as h5_file:
        for image_path in tqdm(image_paths, desc="Writing H5"):
            key = image_path.relative_to(image_root).as_posix()
            image = Image.open(image_path).convert("RGB")
            h5_file.create_dataset(key, data=np.asarray(image), compression="gzip")

    print(f"Wrote {len(image_paths)} images to {output_h5}")


def main() -> None:
    args = parse_args()
    write_h5(args.image_root, args.output_h5, args.overwrite)


if __name__ == "__main__":
    main()
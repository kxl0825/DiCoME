"""Generate split txt files from H5 dataset keys.

The script scans all datasets inside an H5 file and writes one txt file per
selected path component. For example, if an H5 key is:

    CDFv3/Celeb-real/id0_0000/045.png

and `--group_index 1` is used, the key is written to:

    Celeb-real.txt

Each line in the generated txt files is the full H5 key, which can be used by
`H5DeepfakeDataset`.
"""

import argparse
import sys
from pathlib import Path
from typing import TextIO

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create txt split files from H5 keys.")
    parser.add_argument("--h5_path", type=Path, required=True, help="Path to the input H5 file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where generated txt files will be saved.",
    )
    parser.add_argument(
        "--group_index",
        type=int,
        default=1,
        help="Path component used as the output txt filename. Default: 1.",
    )
    return parser.parse_args()


def create_key_writer(
    output_dir: Path,
    file_handles: dict[str, TextIO],
    group_index: int,
):
    """Create an H5 visitor callback that writes dataset keys to txt files."""

    def write_dataset_key(name: str, obj: h5py.HLObject) -> None:
        if not isinstance(obj, h5py.Dataset):
            return

        parts = name.split("/")
        if len(parts) <= group_index:
            print(f"Skipping short H5 key: {name}", file=sys.stderr)
            return

        group_name = parts[group_index]
        if group_name not in file_handles:
            output_path = output_dir / f"{group_name}.txt"
            file_handles[group_name] = open(output_path, "w", encoding="utf-8")
            print(f"Creating split file: {output_path}")

        file_handles[group_name].write(name + "\n")

    return write_dataset_key


def generate_txt_lists(h5_path: Path, output_dir: Path, group_index: int = 1) -> None:
    """Scan an H5 file and export grouped txt split files."""
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handles: dict[str, TextIO] = {}

    try:
        callback = create_key_writer(output_dir, file_handles, group_index)
        with h5py.File(h5_path, "r") as h5_file:
            print(f"Scanning H5 file: {h5_path}")
            h5_file.visititems(callback)
    finally:
        for handle in file_handles.values():
            handle.close()

    if not file_handles:
        print("No image datasets were found; no split files were created.")
    else:
        print(f"Saved {len(file_handles)} split files to: {output_dir}")


def main() -> None:
    args = parse_args()
    generate_txt_lists(args.h5_path, args.output_dir, args.group_index)


if __name__ == "__main__":
    main()
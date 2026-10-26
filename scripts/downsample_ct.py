"""Downsample 1 mm CT slabs to the 2 mm simulation grid and rotate them into the dataset orientation.

Input: CT slabs of 512 x 64 x 512 mm at 1 mm^3 (Hounsfield units, .npy).
Output: int16 .npy volumes of 256 x 32 x 256 (2 mm^3), nearest-neighbour downsampled and rotated by
180 degrees in the X-Z plane (array axes 0 and 2).

Example:
    python scripts/downsample_ct.py --input-dir ct_1mm/test --output-dir dataset_2mm/test/CT_scans
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage


ZOOM = 0.5  # 1 mm -> 2 mm


def downsample_ct(ct, zoom=ZOOM):
    downsampled = ndimage.zoom(ct, zoom=(zoom, zoom, zoom), order=0).astype(np.int16)
    return np.rot90(downsampled, k=2, axes=(0, 2)).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="directory of 1 mm CT .npy files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--previews", action="store_true", help="also save a PNG of the central slice")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files in {args.input_dir}")

    for path in files:
        ct = downsample_ct(np.load(path))
        np.save(args.output_dir / path.name, ct)
        print(f"{path.name}: {ct.shape}")
        if args.previews:
            import matplotlib.pyplot as plt
            plt.imsave(args.output_dir / f"{path.stem}.png", ct[:, ct.shape[1] // 2, :], cmap="gray")


if __name__ == "__main__":
    main()

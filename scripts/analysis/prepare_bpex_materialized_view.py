#!/usr/bin/env python3
"""Make a BPEX training view using the historically materialized 512px images.

Preserve the frozen manifest and its original image_path as source_image_path;
the training-view image_path explicitly names the pre-resized input. One pair
of same-patient, same-class ROMEU rows shares a filename and therefore shares
one materialized input, as in the original experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--freeze-record', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Training view already exists: {args.output}')

    freeze = json.loads(args.freeze_record.read_text(encoding='utf-8'))
    expected = freeze['csv_sha256']
    paths = {p.name: p for p in args.frozen.glob('*.csv')}
    if set(paths) != set(expected) or any(sha256(paths[name]) != value for name, value in expected.items()):
        raise ValueError('Frozen CSVs do not match the freeze record')

    image_root = args.images.resolve(strict=True)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    filenames = {str(name) for frame in frames.values() for name in frame['filename']}
    missing = sorted(name for name in filenames if not (image_root / name).is_file())
    if missing:
        raise FileNotFoundError(f'{len(missing)} materialized images missing; first: {missing[0]}')

    args.output.mkdir(parents=True)
    for name, frame in frames.items():
        if 'image_path' in frame.columns:
            frame.insert(frame.columns.get_loc('image_path') + 1, 'source_image_path', frame['image_path'])
            frame['image_path'] = frame['filename'].map(lambda name: str(image_root / str(name)))
        # The two internal-only new_test_* CSVs historically contain only
        # filename + label; keep their schema unchanged.
        frame.to_csv(args.output / name, index=False)

    # The training view is intentionally *not* used for the patient/path split
    # audit; the frozen source retains the two original path identities.
    record = {
        'freeze_record': str(args.freeze_record),
        'image_root': str(image_root),
        'unique_materialized_images': len(filenames),
        'total_input_bytes': sum((image_root / name).stat().st_size for name in filenames),
        'derived_csv_sha256': {name: sha256(args.output / name) for name in sorted(paths)},
    }
    (args.output / 'view.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(f"Prepared {len(frames)} CSVs using {len(filenames)} materialized images "
          f"({record['total_input_bytes'] / (1024**2):.1f} MiB) at {args.output}")


if __name__ == '__main__':
    main()

# Indian Sign Language Recognition with Multilingual Translation

Place labeled image sequences under `data/` in per-class folders or use a CSV mapping.

CSV formats supported:
1. Sequence CSV: `sequence_id,frame_paths,label` where `frame_paths` is a semicolon- or comma-separated list of frame filepaths (relative to `images_root` or absolute).
2. Single-image CSV: `filepath,label`

Train:
```bash
python src/train.py --data_csv data/labels.csv --images_root data/frames

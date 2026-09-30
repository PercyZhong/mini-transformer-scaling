# Data directory

Run `python scripts/download_data.py` to download Tiny Shakespeare. Raw and
processed data are deliberately ignored by Git. If download fails, manually
place the canonical `input.txt` at `data/raw/tiny_shakespeare.txt`, then rerun
the command; its SHA256 and provenance manifest will be generated locally.

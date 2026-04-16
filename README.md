# esmfold

Predict protein 3D structures from amino acid sequences using
[ESM3](https://github.com/evolutionaryscale/esm). For each sequence the script
writes a PDB file, and produces two summary files:

- `features.json` — per-sequence log-likelihood and mean embedding
- `run.json` — start/end time, GPU used, and any failed sequences

## Requirements

- Python 3.12
- PyTorch (GPU recommended)
- An [HuggingFace Account](https://huggingface.co) and API
  token to download the ESM3 weights on first run

## Installation

```bash
pip install esm pyfastx httpx
```

Then make the script executable:

```bash
chmod +x esmfold.py
```

## Usage

```
python esmfold.py <fasta_file> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `fasta_file` | Input FASTA file (required) |
| `-o`, `--output-dir DIR` | Directory for output files (default: `.`) |
| `-n`, `--num-steps N` | Structure generation steps (default: `8`) |
| `-t`, `--temperature T` | Sampling temperature (default: `0.7`) |

### Example

```bash
python esmfold.py sequences.fasta -o results/ --num-steps 8 --temperature 0.7
```

This produces:

```
results/
├── SEQ1.pdb
├── SEQ2.pdb
├── features.json
└── run.json
```

### `features.json` format

```json
[
  {
    "id": "SEQ1",
    "loglik": -42.3,
    "embedding": [0.12, -0.05, "..."]
  }
]
```

### `run.json` format

```json
{
  "start_time": "2026-04-16T10:23:01+00:00",
  "end_time": "2026-04-16T10:25:44+00:00",
  "gpu": "NVIDIA A100 80GB PCIe",
  "failed_sequences": []
}
```

## Docker

Build the image:

```bash
docker build -t esmfold .
```

Run on a FASTA file, mounting a local directory for input and output:

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  esmfold /data/sequences.fasta -o /data/results/
```

## Notes

- PDB files are named after the sequence identifier in the FASTA header.
  Characters that are unsafe in filenames are replaced with `_`.
- On first run ESM3 will download model weights (~1.4 GB). Set the
  `ESM_CACHE_DIR` environment variable to control where they are stored.
- Sequences that fail (e.g. due to unsupported characters) are logged and
  recorded in `run.json`. The script exits with code `1` if any sequence
  failed.

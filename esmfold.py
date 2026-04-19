#!/usr/bin/env python3
"""ESM3 protein structure prediction script.

Reads amino acid sequences from a FASTA file and predicts their 3D structures
using ESM3, writing one PDB file per sequence, a features.json with per-sequence
log-likelihoods and embeddings, and a run.json with run metadata.

Usage:
    python esmfold.py sequences.fasta -o results/
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pyfastx
import torch
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    GenerationConfig,
    LogitsConfig,
)
from esm.utils.constants.models import ESM3_OPEN_SMALL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SequenceFeatures(TypedDict):
    id: str
    loglik: float
    embedding: list[float]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sequences(fasta_path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file and return a list of (id, sequence) tuples."""
    sequences = [(seq.name, seq.seq) for seq in pyfastx.Fasta(str(fasta_path))]
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    return sequences


def sanitize_id(seq_id: str) -> str:
    """Replace characters unsafe in filenames."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in seq_id)


def write_json(path: Path, data: object, **kwargs) -> None:
    """Serialise data to JSON and log the output path."""
    with path.open("w") as f:
        json.dump(data, f, **kwargs)
    log.info("Written: %s", path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model() -> ESM3:
    """Load ESM3 from pretrained weights, using GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(device))
    log.info("Model loaded")
    return model


# ---------------------------------------------------------------------------
# Per-sequence computations
# ---------------------------------------------------------------------------

def compute_features(model: ESM3, sequence: str) -> SequenceFeatures:
    """Return log-likelihood and mean embedding for a sequence.

    The log-likelihood is the sum of per-residue log-probabilities under the
    sequence track. The embedding is the mean-pooled per-position hidden state.
    Both exclude the BOS and EOS special tokens.
    """
    protein_tensor = model.encode(ESMProtein(sequence=sequence))
    output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )

    seq_tokens = protein_tensor.sequence  # [L]
    seq_logits = output.logits.sequence.squeeze(0)  # [L, vocab]
    embeddings = output.embeddings.squeeze(0)  # [L, d_model]

    # Exclude BOS (position 0) and EOS (position -1)
    residue_tokens = seq_tokens[1:-1]
    log_probs = F.log_softmax(seq_logits[1:-1], dim=-1)
    loglik = log_probs[
        torch.arange(len(residue_tokens)), residue_tokens
    ].sum().item()
    embedding = embeddings[1:-1].mean(dim=0).tolist()

    return {"loglik": loglik, "embedding": embedding}


def predict_structure(
    model: ESM3,
    sequence: str,
    num_steps: int,
    temperature: float,
    schedule: str,
    strategy: str,
) -> ESMProtein:
    """Run ESM3 structure generation for a single sequence."""
    result = model.generate(
        ESMProtein(sequence=sequence),
        GenerationConfig(
            track="structure",
            num_steps=num_steps,
            temperature=temperature,
            schedule=schedule,
            strategy=strategy,
        ),
    )
    if isinstance(result, ESMProteinError):
        raise RuntimeError(
            f"ESM3 generation failed (code {result.error_code}): {result.error_msg}"
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict protein structures from a FASTA file using ESM3."
    )
    parser.add_argument("fasta_file", type=Path, help="Input FASTA file.")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        default=1,
        metavar="N",
        help="Structure generation steps (default: 1).",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        metavar="T",
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--schedule", "-s",
        type=str,
        default="cosine",
        metavar="SCHEDULE",
        help="Noise schedule for structure generation, e.g. cosine, linear (default: cosine).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="entropy",
        metavar="STRATEGY",
        help="Decoding strategy, e.g. entropy, random (default: entropy).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args(argv)

    if not args.fasta_file.is_file():
        log.error("FASTA file not found: %s", args.fasta_file)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sequences = load_sequences(args.fasta_file)
    except Exception as exc:
        log.error("Error reading FASTA file: %s", exc)
        sys.exit(1)

    log.info("Found %d sequence(s). Loading model...", len(sequences))
    try:
        model = load_model()
    except Exception as exc:
        log.error("Error loading model: %s", exc, exc_info=True)
        sys.exit(1)

    log.info(
        "Starting predictions (steps=%d, temperature=%.2f, schedule=%s, strategy=%s)...",
        args.num_steps,
        args.temperature,
        args.schedule,
        args.strategy,
    )

    start_time = datetime.now(timezone.utc)
    features: list[SequenceFeatures] = []
    failed: list[str] = []

    for seq_id, sequence in sequences:
        log.info("[%s] length=%d", seq_id, len(sequence))
        out_path = args.output_dir / f"{sanitize_id(seq_id)}.pdb"
        try:
            log.info("  Computing log-likelihood and embedding...")
            seq_features = compute_features(model, sequence)

            log.info("  Running structure prediction...")
            protein = predict_structure(
                model, sequence, args.num_steps, args.temperature,
                args.schedule, args.strategy,
            )
            protein.to_pdb(str(out_path))
            log.info("  -> %s", out_path)

            features.append({"id": seq_id, **seq_features})
        except Exception as exc:
            log.error("  [%s] FAILED: %s", seq_id, exc, exc_info=True)
            failed.append(seq_id)

    end_time = datetime.now(timezone.utc)

    write_json(args.output_dir / "features.json", features)

    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None
    )
    write_json(
        args.output_dir / "run.json",
        {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "gpu": gpu_name,
            "fasta_file": str(args.fasta_file),
            "num_steps": args.num_steps,
            "temperature": args.temperature,
            "schedule": args.schedule,
            "strategy": args.strategy,
            "failed_sequences": failed,
        },
        indent=2,
    )

    success = len(sequences) - len(failed)
    log.info(
        "Done. %d/%d structure(s) written to %s",
        success,
        len(sequences),
        args.output_dir,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

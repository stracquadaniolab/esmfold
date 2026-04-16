"""ESM3 protein structure prediction CLI.

Reads amino acid sequences from a FASTA file and predicts their 3D structures
using ESM3, writing one PDB file per sequence.
"""

import sys
import traceback
from pathlib import Path

import click
from Bio import SeqIO


def load_sequences(fasta_path: Path) -> list[tuple[str, str]]:
    """Return a list of (id, sequence) tuples from a FASTA file."""
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append((record.id, str(record.seq)))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")
    return records


def load_model():
    """Load ESM3 model from pretrained weights, using GPU if available."""
    import torch
    from esm.models.esm3 import ESM3, ESM3_OPEN_SMALL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(device))
    click.echo("Model loaded")
    return model


def predict_structure(model, sequence: str, num_steps: int, temperature: float):
    """Run ESM3 structure generation for a single amino acid sequence.

    Raises RuntimeError if the model returns an error response.
    """
    from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig

    protein = ESMProtein(sequence=sequence)
    config = GenerationConfig(
        track="structure",
        num_steps=num_steps,
        temperature=temperature,
    )
    result = model.generate(protein, config)
    if isinstance(result, ESMProteinError):
        raise RuntimeError(f"ESM3 generation failed (code {result.error_code}): {result.error_msg}")
    return result


def sanitize_id(seq_id: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in seq_id)


@click.command()
@click.argument("fasta_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-dir", "-o",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory where PDB files are written.",
)
@click.option(
    "--num-steps", "-n",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of structure generation steps.",
)
@click.option(
    "--temperature", "-t",
    default=0.7,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Sampling temperature for structure generation.",
)
def main(
    fasta_file: Path,
    output_dir: Path,
    num_steps: int,
    temperature: float,
) -> None:
    """Predict protein structures from FASTA_FILE using ESM3.

    One PDB file is written per sequence, named after the sequence identifier.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sequences = load_sequences(fasta_file)
    except Exception as exc:
        click.echo(f"Error reading FASTA file: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(sequences)} sequence(s). Loading model...")
    try:
        model = load_model()
    except Exception as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
    click.echo(f"Starting structure prediction (steps={num_steps}, temperature={temperature})...")

    success = 0
    for seq_id, sequence in sequences:
        safe_id = sanitize_id(seq_id)
        out_path = output_dir / f"{safe_id}.pdb"
        click.echo(f"\n[{seq_id}] length={len(sequence)}")
        click.echo(f"\nsaving to file: {out_path}")

        try:
            click.echo("  Running structure prediction...")
            protein = predict_structure(model, sequence, num_steps, temperature)
            click.echo(f"  Prediction done. coordinates set: {protein.coordinates is not None}")
            protein.to_pdb(str(out_path))
            click.echo(f"  File written: {out_path.exists()}")
            click.echo(f"  -> {out_path}")
            success += 1
        except Exception as exc:
            click.echo(f"  ERROR: {exc}")
            click.echo(traceback.format_exc())

    click.echo(f"\nDone. {success}/{len(sequences)} structure(s) written to {output_dir}/")
    if success < len(sequences):
        sys.exit(1)


if __name__ == "__main__":
    main()

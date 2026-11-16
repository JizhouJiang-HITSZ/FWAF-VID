"""Shared file handling for the dataset conversion scripts."""

from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import tempfile


def validate_paths(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if source.resolve() == destination.resolve() or (
        destination.exists() and os.path.samefile(source, destination)
    ):
        raise ValueError("Input and output must be different files")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")


@contextmanager
def atomic_output(destination):
    """Replace the destination only after the complete conversion succeeds."""
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix,
        dir=str(destination.parent),
    )
    os.close(fd)
    try:
        yield Path(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_topic_csv(bagfile, topic, destination):
    """Run rostopic without shell interpolation and propagate export failures."""
    with open(destination, 'w', encoding='utf-8', newline='') as output:
        subprocess.run(
            ['rostopic', 'echo', '-b', str(Path(bagfile).resolve()), '-p', topic],
            stdout=output, check=True,
        )


def require_columns(frame, columns, label):
    if frame.empty:
        raise ValueError(f"No messages found for {label}")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for {label}: {', '.join(sorted(missing))}")


def write_tum_row(stream, values):
    stream.write(' '.join(str(value) for value in values) + '\n')

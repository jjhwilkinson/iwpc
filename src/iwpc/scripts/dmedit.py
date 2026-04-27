"""
Interactive editor for a ``PandasDirDataModule``.

Loads a dataset directory as ``dm`` and drops the user into an interactive
Python shell (IPython if available, otherwise the stdlib :mod:`code` REPL) so
that transformations, reweightings and tags can be applied directly.

Invoked via ``python -m iwpc dmedit <dataset_dir>``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data_modules.pandas_directory_data_module import PandasDirDataModule


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Register the ``dmedit`` subcommand on the given subparsers object.

    Parameters
    ----------
    subparsers
        The subparsers action returned by ``ArgumentParser.add_subparsers``.

    Returns
    -------
    argparse.ArgumentParser
        The parser created for the ``dmedit`` subcommand.
    """
    parser = subparsers.add_parser(
        "dmedit",
        help="Interactively edit a PandasDirDataModule dataset directory.",
        description=(
            "Load the dataset at <dataset_dir> as a PandasDirDataModule named "
            "`dm` and drop into an interactive Python shell."
        ),
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to a PandasDirDataModule dataset directory.",
    )
    parser.set_defaults(func=run)
    return parser


def _build_banner(dm: PandasDirDataModule) -> str:
    """
    Build the banner string printed before the REPL starts.

    Parameters
    ----------
    dm
        The loaded data module to summarise in the banner.

    Returns
    -------
    str
        The multi-line banner text.
    """
    lines = [
        f"Loaded PandasDirDataModule from {dm.dataset_dir}",
        f"  - tags: {dm.tags}",
        f"  - num_files: {dm.num_files} (train: {dm.num_train_files}, val: {dm.num_validation_files})",
        f"  - columns: {dm.columns}",
        "Available names: dm, np, pd, torch, Path, PandasDirDataModule",
        "Examples:",
        '  dm.transform(fn, out_dir, tag="...")              # write transformed copy to a new dir',
        '  dm.transform(fn, None, force=True, tag="...")     # overwrite in place',
        '  dm.reweight("tag", reweight_fn, out_dir)',
        '  dm.add_tag("manual-edit")',
    ]
    return "\n".join(lines)


def _embed(local_ns: dict, banner: str) -> None:
    """
    Drop into an interactive shell using the supplied namespace.

    Prefers ``IPython.embed`` for tab-completion and rich display; falls back
    to :func:`code.interact` from the standard library if IPython is not
    importable.

    Parameters
    ----------
    local_ns
        The namespace exposed to the interactive session.
    banner
        Banner text printed before the prompt.
    """
    try:
        from IPython import embed  # type: ignore[import-not-found]
    except ImportError:
        import code

        code.interact(banner=banner, local=local_ns, exitmsg="")
        return

    embed(header=banner, user_ns=local_ns, colors="neutral")


def run(args: argparse.Namespace) -> None:
    """
    Execute the ``dmedit`` subcommand.

    Parameters
    ----------
    args
        Parsed CLI arguments. Must expose a ``dataset_dir`` attribute.
    """
    dm = PandasDirDataModule(args.dataset_dir)
    banner = _build_banner(dm)
    local_ns = {
        "dm": dm,
        "np": np,
        "pd": pd,
        "torch": torch,
        "Path": Path,
        "PandasDirDataModule": PandasDirDataModule,
    }
    _embed(local_ns, banner)

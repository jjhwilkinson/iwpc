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

import iwpc

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
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help=(
            "Python expression evaluating to a callable ``df -> df``. When "
            "provided, ``dm.transform`` is invoked non-interactively with this "
            "function instead of dropping into a REPL. The expression is "
            "evaluated with ``np``, ``pd``, ``torch``, ``iwpc``, ``Path`` and "
            "``PandasDirDataModule`` in scope (e.g. "
            "\"lambda df: df.assign(x2=df['x']**2)\")."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=lambda s: None if s in ("", "None") else Path(s),
        default=None,
        help=(
            "Output directory for the transformed dataset. Pass ``None`` (or "
            "omit) to overwrite the input directory in place; requires "
            "``--force``."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite ``--out-dir`` if it already exists (required when overwriting in place).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to record on the transformed dataset.",
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
        "Available names: dm, iwpc, np, pd, torch, Path, PandasDirDataModule",
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
    local_ns = {
        "dm": dm,
        "iwpc": iwpc,
        "np": np,
        "pd": pd,
        "torch": torch,
        "Path": Path,
        "PandasDirDataModule": PandasDirDataModule,
    }

    if args.transform is not None:
        if args.out_dir is None and not args.force:
            raise SystemExit(
                "--out-dir=None overwrites the input dataset in place; pass --force to confirm."
            )
        fn = eval(args.transform, {"__builtins__": __builtins__}, local_ns)
        if not callable(fn):
            raise SystemExit(f"--transform expression did not evaluate to a callable: {fn!r}")
        dm.transform(fn, args.out_dir, force=args.force, tag=args.tag)
        return

    _embed(local_ns, _build_banner(dm))

"""
Top-level CLI entry point for ``python -m iwpc``.

Dispatches to subcommands registered in :mod:`iwpc.scripts`.
"""
import argparse
import sys

from .scripts import dmedit


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the top-level argument parser with all registered subcommands.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m iwpc",
        description="iwpc command-line utilities.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="available commands",
        description=(
            "Run `python -m iwpc <command> --help` for command-specific options."
        ),
    )
    subparsers.required = True

    dmedit.add_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and dispatch to the selected subcommand.

    Parameters
    ----------
    argv
        Optional argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        The exit code to return to the shell.
    """
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 1
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

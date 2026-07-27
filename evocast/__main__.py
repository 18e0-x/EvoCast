"""Package entry point for ``python -m evocast``."""

from __future__ import annotations

import sys

from evocast.scripts.wizard import main as wizard_main


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in {"wizard", "start"}:
        args = args[1:]
    wizard_main(args)


if __name__ == "__main__":
    main()

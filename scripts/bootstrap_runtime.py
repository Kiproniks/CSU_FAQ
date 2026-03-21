from __future__ import annotations

import argparse
import runpy
import site
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project script with local site-packages bootstrap.")
    parser.add_argument("target", help="Path to target python script")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    target = Path(args.target).resolve()

    site.addsitedir(str(root / "venv" / "Lib" / "site-packages"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

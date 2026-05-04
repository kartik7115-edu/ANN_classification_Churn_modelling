from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOG_FILES = [
    BASE_DIR / "logs" / "update_history.csv",
    BASE_DIR / "logs" / "parameters.csv",
]
REPORT_DIR = BASE_DIR / "reports" / "learning_diagnostics"


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"Removed {path}")
    else:
        print(f"Not found {path}")


def clear_reports(report_dir: Path) -> None:
    if not report_dir.exists():
        print(f"Not found {report_dir}")
        return
    shutil.rmtree(report_dir)
    print(f"Removed {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset learned app state by removing update and parameter logs."
    )
    parser.add_argument(
        "--clear-reports",
        action="store_true",
        help="Also remove generated diagnostics under reports/learning_diagnostics.",
    )
    args = parser.parse_args()

    for path in LOG_FILES:
        remove_file(path)

    if args.clear_reports:
        clear_reports(REPORT_DIR)

    print("Reset complete.")


if __name__ == "__main__":
    main()

"""Generate the metadata file from reproducible evidence."""

from __future__ import annotations

from argparse import ArgumentParser
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def git_hashes(limit: int = 5) -> list[str]:
    result = subprocess.run(
        ["git", "log", f"-{limit}", "--format=%H"],
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout.splitlines()


def coverage_percent(path: Path) -> float | None:
    if not path.exists():
        return None
    root = ET.parse(Path).getroot()
    return round(float(root.attrib["line-rate"]) * 100, 2)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=Path("coverage.xml"))
    parser.add_argument("--image", default="not-built")
    parser.add_argument("--output", type=Path, default=Path("dev_sec_ops.yml"))
    args = parser.parse_args()

    lines = ["docker_image:", f"    reference: {args.image}", " signature: pending-cosign"]
    lines.append("recent_commits:")
    lines.extend(f" -   {commit}" for commit in git_hashes())

    coverage = coverage_percent(args.coverage)
    coverage_value = coverage if coverage is not None else "unknown"
    lines.extend(["tests:", f"  coverage_percent: {coverage_value}"])

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

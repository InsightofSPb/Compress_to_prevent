"""Adapter boundary for the pinned official LPOSS checkout.

The reviewed upstream commit does not expose a stable, non-training inference API that
accepts arbitrary vocabulary prototypes.  We fail rather than patching its source or
silently substituting local stages.  This explicit boundary is ready for a provenance-
preserving adapter once the exact official invocation/fixture is approved.
"""
import argparse
from pathlib import Path
import subprocess

EXPECTED = "e489a7445528922ddfe4e39631ef2fe34827c873"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.upstream_root)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED:
        raise SystemExit(f"official LPOSS commit mismatch: {commit}")
    raise SystemExit(
        "OFFICIAL STAGE BLOCKER: pinned LPOSS has no stable CLI/API for exporting raw seed and "
        "patch-propagated tensors for an arbitrary fixed vocabulary. Modifying upstream evaluation "
        "code would alter semantics; numerical parity is therefore not claimed.")


if __name__ == "__main__":
    main()

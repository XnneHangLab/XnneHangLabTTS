#!/usr/bin/env python3
"""Switch the active PyTorch CUDA variant by updating pytorch-gpu index URL in pyproject.toml."""
import re
import sys
import pathlib

VARIANTS = {
    "cpu":   "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print(f"Usage: python use_torch.py <variant>")
        print(f"Variants: {', '.join(VARIANTS)}")
        sys.exit(1)

    variant = sys.argv[1]
    url = VARIANTS[variant]

    p = pathlib.Path("pyproject.toml")
    content = p.read_text(encoding="utf-8")

    new_content, n = re.subn(
        r'(name = "pytorch-gpu"\nurl = ")[^"]+(")',
        rf"\g<1>{url}\2",
        content,
    )

    if n == 0:
        print("Error: pytorch-gpu index not found in pyproject.toml")
        sys.exit(1)

    p.write_text(new_content, encoding="utf-8")
    print(f"pytorch-gpu → {variant} ({url})")


if __name__ == "__main__":
    main()

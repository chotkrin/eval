#!/usr/bin/env python3
"""Remove obsolete interpreter settings from a private .env without losing paths.

Only remove retired keys and rename CUDA_MODULE_LLMJP to EVAL_CUDA_MODULE.
Never print values or load the file as Python; preserve all other lines verbatim.
Usage: python3 sh/mdx/migrate_env.py .env
"""

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

OLD_KEYS = frozenset({
    "LEGACY_EVAL_PYTHON", "CUDA_MODULE_LEGACY", "RAY_PYTHON", "RAY_CLI",
})
ASSIGN = re.compile(r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<suffix>\s*=.*)$")


def migrate_env(path: Path) -> tuple[bool, str | None]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Not a regular .env file: {path}")
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    active_keys = {
        match.group("key")
        for line in lines
        if (match := ASSIGN.match(line)) is not None
    }
    have_new_cuda = "EVAL_CUDA_MODULE" in active_keys
    changed_lines = []
    for line in lines:
        match = ASSIGN.match(line)
        if match is None:
            changed_lines.append(line)
            continue
        key = match.group("key")
        if key in OLD_KEYS:
            continue
        if key == "CUDA_MODULE_LLMJP":
            if have_new_cuda:
                continue
            line = line[:match.start("key")] + "EVAL_CUDA_MODULE" + line[match.end("key"):]
            have_new_cuda = True
        changed_lines.append(line)
    updated = "".join(changed_lines)
    if updated == original:
        os.chmod(path, 0o600)
        return False, None

    backup = Path(str(path) + ".bak")
    index = 1
    while backup.exists():
        backup = Path(str(path) + f".bak.{index}")
        index += 1
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".env.migrate-", delete=False
        ) as handle:
            temp_name = handle.name
            os.chmod(temp_name, 0o600)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return True, str(backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", type=Path, default=Path(".env"))
    args = parser.parse_args()
    changed, backup = migrate_env(args.env_file)
    if changed:
        print(f"Migrated {args.env_file}; private backup: {backup}")
    else:
        print(f"No obsolete settings in {args.env_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Relative Link and Asset Verifier for Repository Documentation
============================================================

Parses README.md and DATASET.md for markdown image links `![...](...)`
and hyperlinks `[...](...)`, including nested badge links.
For all local relative file paths:
1. Asserts target file exists on disk.
2. Asserts case-sensitive match against actual filesystem filename.
"""

import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_file_links(doc_path: Path):
    print(f"\nScanning links in {doc_path.relative_to(REPO_ROOT)}...")
    if not doc_path.exists():
        print(f"  [ERROR] Document file does not exist: {doc_path}")
        return False, []

    content = doc_path.read_text(encoding="utf-8")

    # Match markdown links: [text](link) and ![alt](link), handling nested brackets like [![...](...)](link)
    link_pattern = re.compile(r'!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(([^)]+)\)')

    matches = link_pattern.findall(content)
    all_ok = True
    checked = []

    for raw_link in matches:
        # Strip title or anchor fragments if present
        clean_link = raw_link.strip().split()[0].split("#")[0]
        if not clean_link:
            continue
        if clean_link.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue

        target_path = (doc_path.parent / clean_link).resolve()

        # Check existence
        if not target_path.exists():
            print(f"  [FAIL] Missing file: '{clean_link}'")
            all_ok = False
            checked.append((clean_link, False, "File not found"))
            continue

        # Case-sensitive check against filesystem
        # Windows filesystem is case-insensitive, so resolve each path part against actual directory contents
        current = target_path.anchor
        parts = target_path.parts[1:]
        mismatch = False
        for part in parts:
            matched_entry = None
            try:
                for entry in Path(current).iterdir():
                    if entry.name == part:
                        matched_entry = entry.name
                        break
                    elif entry.name.lower() == part.lower():
                        matched_entry = entry.name
                        mismatch = True
                        break
            except OSError:
                pass

            if mismatch:
                print(f"  [FAIL] Case mismatch: link specifies '{part}' but disk has '{matched_entry}' (Path: '{clean_link}')")
                all_ok = False
                checked.append((clean_link, False, f"Case mismatch: '{part}' vs '{matched_entry}'"))
                break
            current = str(Path(current) / part)

        if not mismatch:
            print(f"  [PASS] Verified: {clean_link}")
            checked.append((clean_link, True, "OK"))

    return all_ok, checked


def main():
    print("=" * 65)
    print("SMART CANTEEN DOCUMENTATION LINK CHECKER")
    print("=" * 65)

    docs_to_check = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "DATASET.md"
    ]

    total_pass = True
    for doc in docs_to_check:
        ok, _ = check_file_links(doc)
        if not ok:
            total_pass = False

    print("\n" + "=" * 65)
    if total_pass:
        print("[SUCCESS] All relative links and image references verified with exact case!")
        print("=" * 65)
        sys.exit(0)
    else:
        print("[FAILURE] Broken or case-mismatched links detected. Please fix them.")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()

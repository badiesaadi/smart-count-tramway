#!/usr/bin/env python3
"""
scripts/push_updates.py — Smart Count Tramway
===============================================
Pushes the latest code updates to the existing GitHub repo.
Run this from your project root directory.

Usage:
    python scripts/push_updates.py --token YOUR_GITHUB_TOKEN --message "your commit message"
"""

import argparse
import subprocess
import sys
import ssl
import urllib.request
import json


_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def run(cmd: list, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def get_username(token: str) -> str:
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={"Authorization": f"token {token}"},
    )
    with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
        return json.loads(resp.read())["login"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",   required=True)
    parser.add_argument("--message", default="feat: add global ReID + hardware documentation")
    args = parser.parse_args()

    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n── Getting GitHub username ──────────────────────────────")
    username  = get_username(args.token)
    repo_url  = f"https://{args.token}@github.com/{username}/smart-count-tramway.git"
    print(f"  Logged in as: {username}")

    print("\n── Staging all changes ──────────────────────────────────")
    run(["git", "add", "-A"], cwd=project_root)

    print("\n── Committing ───────────────────────────────────────────")
    run(["git", "commit", "-m", args.message], cwd=project_root)

    print("\n── Updating remote URL ──────────────────────────────────")
    subprocess.run(["git", "remote", "remove", "origin"],
                   cwd=project_root, capture_output=True)
    run(["git", "remote", "add", "origin", repo_url], cwd=project_root)

    print("\n── Pushing to GitHub ────────────────────────────────────")
    run(["git", "push", "origin", "main", "--force"], cwd=project_root)

    print(f"\n🎉  Done! https://github.com/{username}/smart-count-tramway\n")


if __name__ == "__main__":
    main()

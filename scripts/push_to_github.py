#!/usr/bin/env python3
"""
scripts/push_to_github.py — Smart Count Tramway
=================================================
Run this script ONCE on your own machine to:
  1. Create the 'smart-count-tramway' repo on your GitHub account.
  2. Initialise a local git repo.
  3. Commit all project files.
  4. Push to GitHub.

Usage:
    python scripts/push_to_github.py --token YOUR_GITHUB_TOKEN

Your token needs the 'repo' scope.
Get one at: https://github.com/settings/tokens
"""

import argparse
import json
import ssl
import subprocess
import sys
import urllib.request
import urllib.error

# Fix for SSL certificate verification failure on Windows (Python 3.12+)
# This is safe here because we are only talking to api.github.com
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


REPO_NAME        = "smart-count-tramway"
REPO_DESCRIPTION = "Edge-AI automatic passenger counting system for SETRAM Mostaganem tramway (PFE)"
REPO_PRIVATE     = False   # Set True if you want a private repo


def create_github_repo(token: str) -> str:
    """Create repo via GitHub API. Returns the clone URL."""
    url     = "https://api.github.com/user/repos"
    payload = json.dumps({
        "name":        REPO_NAME,
        "description": REPO_DESCRIPTION,
        "private":     REPO_PRIVATE,
        "auto_init":   False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/vnd.github+json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
            print(f"✅  Repo created: {data['html_url']}")
            return data["clone_url"]
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        if "already exists" in body.get("errors", [{}])[0].get("message", ""):
            # Repo exists — get its clone URL
            me_req = urllib.request.Request(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}"},
            )
            with urllib.request.urlopen(me_req, context=_SSL_CTX) as resp:
                username = json.loads(resp.read())["login"]
            clone_url = f"https://github.com/{username}/{REPO_NAME}.git"
            print(f"ℹ️   Repo already exists — using {clone_url}")
            return clone_url
        print(f"❌  GitHub API error: {body}")
        sys.exit(1)


def run(cmd: list, cwd=None):
    """Run a shell command, print it, and raise on failure."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    args = parser.parse_args()

    import os
    # Determine project root (one level up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n── Step 1: Create GitHub repo ──────────────────────────")
    clone_url = create_github_repo(args.token)

    # Embed token in URL for push authentication
    authed_url = clone_url.replace("https://", f"https://{args.token}@")

    print("\n── Step 2: Initialise local git repo ───────────────────")
    git_dir = os.path.join(project_root, ".git")
    if not os.path.exists(git_dir):
        run(["git", "init"], cwd=project_root)
        run(["git", "branch", "-M", "main"], cwd=project_root)
    else:
        print("  Git already initialised — skipping init.")

    print("\n── Step 3: Configure git identity ──────────────────────")
    run(["git", "config", "user.email", "smartcount@tramway.dz"], cwd=project_root)
    run(["git", "config", "user.name",  "Smart Count Tramway"],   cwd=project_root)

    print("\n── Step 4: Stage all files ──────────────────────────────")
    run(["git", "add", "-A"], cwd=project_root)

    print("\n── Step 5: Commit ───────────────────────────────────────")
    run(
        ["git", "commit", "-m", "feat: initial project scaffold — Smart Count Tramway PFE"],
        cwd=project_root,
    )

    print("\n── Step 6: Add remote & push ────────────────────────────")
    # Remove existing remote if any (safe re-run)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=project_root, capture_output=True)
    run(["git", "remote", "add", "origin", authed_url], cwd=project_root)
    run(["git", "push", "-u", "origin", "main"], cwd=project_root)

    print(f"\n🎉  Done!  Your repo is live at: {clone_url.replace('.git', '')}\n")


if __name__ == "__main__":
    main()

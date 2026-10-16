#!/usr/bin/env python3
"""Historical private-repository publisher; not used for the existing public repo.

Default is an offline dry run. --execute uses the locally authenticated GitHub
CLI. No embedded credentials, global git changes, force pushes or repo deletion.
For wlvh/attention-lab use normal clone/branch/PR; do not run this legacy command.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNER = "wlvh"
DEFAULT_NAME = "attention-lab"
DESCRIPTION = "Mac-first attention tradeoff lab: research context, reproducible Python experiments, and notebook teaching"

class PublishError(RuntimeError):
    pass


def command(args: list[str], *, check: bool = True, timeout: int = 90) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode != 0:
        raise PublishError(f"Command failed ({result.returncode}): {shlex.join(args)}\n{result.stderr.strip()}")
    return result


def safe_name(owner: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", owner):
        raise PublishError("Invalid GitHub owner")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise PublishError("Invalid repository name")
    return f"{owner}/{name}"


def origin_matches(url: str, full_name: str) -> bool:
    stem = full_name.lower()
    return url.strip().lower() in {
        f"https://github.com/{stem}", f"https://github.com/{stem}.git",
        f"git@github.com:{stem}.git", f"git@github.com:{stem}",
    }


def verify_bundle(root: Path) -> list[str]:
    root = root.resolve()
    if not (root / ".attention_lab_bootstrap").is_file():
        raise PublishError("Bootstrap marker missing; refusing an unrelated directory")
    manifest_path = root / "bootstrap_manifest.json"
    if not manifest_path.is_file():
        raise PublishError("bootstrap_manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []
    for row in manifest["files"]:
        relative = row["path"]
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or ".git" in Path(relative).parts:
            raise PublishError(f"Missing or unsafe bundle file: {relative}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            raise PublishError(f"Bundle file changed: {relative}. Review changes before manual publication; do not silently regenerate hashes.")
        paths.append(relative)
    return sorted(set(paths + ["bootstrap_manifest.json"]))


def prepare_local_git(paths: list[str]) -> str:
    root_check = command(["git", "rev-parse", "--show-toplevel"], check=False)
    if root_check.returncode == 0 and Path(root_check.stdout.strip()).resolve() != ROOT.resolve():
        raise PublishError("This directory belongs to a parent Git repository; move it to an independent directory")
    if root_check.returncode != 0:
        command(["git", "init", "-b", "main"])
    head = command(["git", "rev-parse", "--verify", "HEAD"], check=False)
    if head.returncode == 0:
        branch = command(["git", "branch", "--show-current"]).stdout.strip()
        if branch != "main":
            raise PublishError("Publish the reviewed main branch, not an unrelated or detached branch")
        if command(["git", "status", "--porcelain"]).stdout.strip():
            raise PublishError("Existing repository is not clean. Review and commit intentionally before publishing")
        return head.stdout.strip()
    staged = command(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    if any(p not in paths for p in staged):
        raise PublishError("Unreviewed files were already staged")
    command(["git", "symbolic-ref", "HEAD", "refs/heads/main"])
    command(["git", "add", "--", *paths])
    has_name = command(["git", "config", "user.name"], check=False).returncode == 0
    has_email = command(["git", "config", "user.email"], check=False).returncode == 0
    identity = [] if has_name and has_email else ["-c", "user.name=Attention Lab Bootstrap", "-c", "user.email=bootstrap@local.invalid"]
    command(["git", *identity, "commit", "-m", "docs: bootstrap Mac-first attention tradeoff lab"])
    return command(["git", "rev-parse", "HEAD"]).stdout.strip()


def remote_info(full_name: str, *, check: bool = True) -> dict | None:
    r = command(["gh", "repo", "view", full_name, "--json", "nameWithOwner,isPrivate,url"], check=False)
    if r.returncode:
        if check:
            raise PublishError(f"Could not read repository metadata: {r.stderr.strip()}")
        return None
    data = json.loads(r.stdout)
    if data.get("nameWithOwner", "").lower() != full_name.lower() or data.get("isPrivate") is not True:
        raise PublishError("Repository identity or private visibility did not match; refusing to upload")
    return data


def publish(owner: str, name: str, *, resume: bool = False) -> dict:
    full_name = safe_name(owner, name)
    for tool in ("git", "gh"):
        if shutil.which(tool) is None:
            raise PublishError(f"{tool} is missing. Install it locally with approval before retrying")
    command(["gh", "auth", "status", "--hostname", "github.com"])
    login = command(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
    if login.lower() != owner.lower():
        raise PublishError(f"Authenticated user is {login}, expected {owner}; no repository was created")
    paths = verify_bundle(ROOT)
    command([sys.executable, str(ROOT / "scripts/check_sources.py")])
    sha = prepare_local_git(paths)
    existing_origin = command(["git", "remote", "get-url", "origin"], check=False)
    if existing_origin.returncode == 0 and not origin_matches(existing_origin.stdout, full_name):
        raise PublishError("origin points elsewhere; refusing to replace it")
    current = remote_info(full_name, check=False)
    if current is not None and not resume:
        raise PublishError("Same-name repository exists. Default refuses to overwrite; --resume only supports empty or already-identical private repository")
    if current is None:
        if resume:
            raise PublishError("--resume needs a readable existing private repository; verify authentication and the previous creation result")
        command(["gh", "repo", "create", full_name, "--private", "--description", DESCRIPTION, "--disable-wiki"])
        current = remote_info(full_name)
    if existing_origin.returncode != 0:
        command(["git", "remote", "add", "origin", f"https://github.com/{full_name}.git"])
    git_auth = ["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    refs_result = command([*git_auth, "ls-remote", "origin"])
    refs = [line.split() for line in refs_result.stdout.splitlines() if line.strip()]
    if refs:
        branches = {name: commit for commit, name in refs if name.startswith("refs/heads/")}
        if branches != {"refs/heads/main": sha}:
            raise PublishError("Remote has other history. Refusing to merge, replace, or force push")
    command([*git_auth, "push", "-u", "origin", "main"], timeout=180)
    final = remote_info(full_name)
    remote_sha = command(["gh", "api", f"repos/{full_name}/git/ref/heads/main", "--jq", ".object.sha"]).stdout.strip()
    if remote_sha != sha:
        raise PublishError("Push returned but remote main SHA differs; publication not verified")
    receipt = {"status": "published_and_verified", "repository": full_name, "url": final["url"],
               "private": True, "local_commit": sha, "remote_main_commit": remote_sha,
               "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / ".publication_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args(argv)
    try:
        full_name = safe_name(a.owner, a.name)
        if not a.execute:
            print(json.dumps({"mode": "offline_dry_run", "target": full_name, "visibility": "private",
                "will_publish_only_with_execute": True,
                "steps": ["validate local gh account", "verify packaged files and original hashes",
                          "prepare reviewed main commit", "create private repository (reject existing by default)",
                          "push without force", "read back private metadata and main SHA"],
                "cloud_compute_started": False, "remote_repository_created": False}, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(publish(a.owner, a.name, resume=a.resume), ensure_ascii=False, indent=2))
        return 0
    except (PublishError, subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"Publication incomplete: {e}", file=sys.stderr)
        print("Keep local files. A remote may have been created before a push failure; verify before retrying. No force push or deletion was attempted.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

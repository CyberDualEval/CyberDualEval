#!/usr/bin/env python3
"""Pre-pull all Docker images referenced by Vulhub tasks.

One-time operational cost before running the validated-exploitation subset:
``docker compose up`` at task-start otherwise pulls every image on demand,
adding ~60–120s cold-pull latency per task. This script walks
``benchmark/vulhub/*/metadata.json``, collects unique image names from each
task's vendor compose, adds the grader image, and ``docker pull``s them in
parallel.

Usage:
  uv run python scripts/prewarm_vulhub_images.py             # all tasks
  uv run python scripts/prewarm_vulhub_images.py --only-tasks couchdb_CVE-2017-12635,bash_CVE-2014-6271
  uv run python scripts/prewarm_vulhub_images.py --concurrency 2

Docker Hub rate limits anonymous pulls to 100/6h per IP. If
``DOCKERHUB_USERNAME`` and ``DOCKERHUB_TOKEN`` are set in the environment
or ``.env``, the script runs ``docker login`` first; otherwise it warns
and proceeds (you may hit the limit on a cold cache).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DEFAULT_VULHUB_DIR = os.path.join(PROJECT_ROOT, "benchmark", "vulhub")
GRADER_IMAGE = "cyberbench/grader:latest"
# Agent image (public on Docker Hub) — must be pulled by every collaborator.
AGENT_IMAGE = (
    "mms2309/cyberrefusal-agent"
    "@sha256:793d11c6a38c63a69ef7fa5aae44662ed394727e279f57c91266b1a83ef3608e"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def collect_images(
    vulhub_dir: str,
    only_tasks: set[str] | None = None,
    exclude_tasks: set[str] | None = None,
) -> set[str]:
    """Return the set of image names across all referenced compose files."""
    images: set[str] = set()
    for entry in sorted(os.listdir(vulhub_dir)):
        task_dir = os.path.join(vulhub_dir, entry)
        meta_path = os.path.join(task_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        if only_tasks and entry not in only_tasks:
            continue
        if exclude_tasks and entry in exclude_tasks:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        cs = meta.get("compose_source")
        if not cs or not os.path.isfile(cs):
            continue
        try:
            with open(cs) as f:
                c = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            log.warning("  [%s] compose unreadable: %s", entry, e)
            continue
        for svc in (c.get("services") or {}).values():
            if isinstance(svc, dict) and svc.get("image"):
                images.add(svc["image"])
    return images


def docker_login_if_creds() -> None:
    user = os.environ.get("DOCKERHUB_USERNAME")
    token = os.environ.get("DOCKERHUB_TOKEN")
    if not (user and token):
        log.warning(
            "DOCKERHUB_USERNAME/DOCKERHUB_TOKEN not set — pulling anonymously. "
            "Docker Hub rate-limits anonymous pulls to 100/6h per IP.",
        )
        return
    log.info("docker login as %s", user)
    res = subprocess.run(
        ["docker", "login", "-u", user, "--password-stdin"],
        input=token, text=True, capture_output=True,
    )
    if res.returncode != 0:
        log.error("docker login failed: %s", res.stderr.strip())
        sys.exit(2)


def _load_mirror_map() -> dict[str, str]:
    """Return {upstream_image: mirror_digest_ref} from image-mirror-manifest.json
    when ``CYBERAGENTBENCH_IMAGE_MIRROR`` is set. Mirrors the rewriting logic
    in refusal_eval.sandbox so prewarm pulls go through GHCR instead of
    Docker Hub for every image present in the manifest."""
    if not os.environ.get("CYBERAGENTBENCH_IMAGE_MIRROR"):
        return {}
    manifest_path = Path(PROJECT_ROOT) / "docs" / "image-mirror-manifest.json"
    if not manifest_path.is_file():
        log.warning(
            "CYBERAGENTBENCH_IMAGE_MIRROR set but %s missing; pulls will hit "
            "Docker Hub.", manifest_path,
        )
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("image-mirror-manifest.json unreadable: %s", e)
        return {}
    out: dict[str, str] = {}
    for entry in manifest.get("entries", []):
        upstream = entry.get("upstream")
        ref = entry.get("mirror_digest") or entry.get("mirror_tag")
        if upstream and ref:
            out[upstream] = ref
    if out:
        log.info("Image mirror active: %d entries; will pull via GHCR where covered", len(out))
    return out


def pull_one(image: str, mirror_map: dict[str, str]) -> tuple[str, bool, str]:
    """Pull one image. If a mirror_digest is available, pull that instead — it
    sidesteps Docker Hub rate limits and the image cache is keyed on the full
    ref, so generate_compose's matching mirror rewrite finds it."""
    target = mirror_map.get(image, image)
    res = subprocess.run(
        ["docker", "pull", target],
        capture_output=True, text=True,
    )
    ok = res.returncode == 0
    tail = (res.stderr or res.stdout).strip().splitlines()[-1:]
    suffix = f" (via mirror)" if target != image else ""
    return image, ok, (tail[0] if tail else "") + suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vulhub-dir", default=DEFAULT_VULHUB_DIR,
        help=f"Path to benchmark/vulhub dir (default: {DEFAULT_VULHUB_DIR})",
    )
    parser.add_argument(
        "--only-tasks", default="",
        help="Comma-separated task_ids to limit pull scope.",
    )
    parser.add_argument(
        "--exclude-tasks", default="",
        help="Comma-separated task_ids to exclude from prewarm.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max parallel docker pulls (default 4).",
    )
    parser.add_argument(
        "--skip-grader", action="store_true",
        help="Do not pull cyberbench/grader:latest (useful when building locally).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List images that would be pulled, then exit.",
    )
    args = parser.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    only_tasks: set[str] | None = None
    if args.only_tasks:
        only_tasks = {t.strip() for t in args.only_tasks.split(",") if t.strip()}

    exclude_tasks: set[str] | None = None
    if args.exclude_tasks:
        exclude_tasks = {t.strip() for t in args.exclude_tasks.split(",") if t.strip()}

    images = collect_images(args.vulhub_dir, only_tasks, exclude_tasks)
    # Agent image is public on Docker Hub and required for every task.
    images.add(AGENT_IMAGE)

    # Grader is built locally (not in any public registry). Verify it exists
    # rather than attempting to pull, unless explicitly skipped.
    if not args.skip_grader:
        check = subprocess.run(
            ["docker", "image", "inspect", GRADER_IMAGE],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            log.error(
                "Grader image %s not built locally. Run:\n"
                "  docker build -t %s -f refusal_eval/grader/Dockerfile .\n"
                "Or pass --skip-grader to proceed with vendor images only.",
                GRADER_IMAGE, GRADER_IMAGE,
            )
            sys.exit(2)
        log.info("Grader image %s already present locally.", GRADER_IMAGE)

    log.info("Collected %d unique vendor images", len(images))

    if args.dry_run:
        for img in sorted(images):
            print(img)
        return

    docker_login_if_creds()
    mirror_map = _load_mirror_map()

    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(pull_one, img, mirror_map): img for img in sorted(images)}
        for fut in concurrent.futures.as_completed(futures):
            img, ok, msg = fut.result()
            if ok:
                successes.append(img)
                log.info("  ✓ %s", img)
            else:
                failures.append((img, msg))
                log.warning("  ✗ %s — %s", img, msg)

    log.info("Done: %d pulled, %d failed", len(successes), len(failures))
    if failures:
        log.warning("Failed images:")
        for img, msg in failures:
            log.warning("  %s — %s", img, msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

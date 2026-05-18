# Distribution & Running Guide — cyberagent-bench

This guide covers two roles:

- **Maintainer** — one-time setup of the dataset/image distribution, plus periodic publishes when the benchmark evolves.
- **Collaborator** — per-machine setup and running an evaluation campaign.

The benchmark is split across three artifacts:

| Artifact | Storage | Why |
|---|---|---|
| `benchmark/` tree (~50 GB) | HuggingFace Datasets repo (private until license-cleared) | Versioned via git+LFS, free, easy `snapshot_download` |
| Agent image (~823 MB) | `mms2309/cyberrefusal-agent@sha256:793d11c6a38c63a69ef7fa5aae44662ed394727e279f57c91266b1a83ef3608e` on Docker Hub (public) | The container the agent's bash/python tools run inside. Public, anonymous pull works. |
| Vendor target docker images (~30-50 GB) | GHCR (or any OCI registry you control) | Bypasses Docker Hub anonymous (100 pulls/6h) and free-auth (200 pulls/6h) limits for distributed runs |
| Code (`refusal_eval/`, `scripts/`, `tests/`, configs, plan docs) | This GitHub repo | Reviewed via PR, no LFS needed |

A "campaign" is a single tagged release: one HF dataset revision + one image-mirror tag, both pinned in a config. Collaborators reproduce the eval bit-for-bit by checking out the matching code commit.

---

## Part A — Maintainer one-time setup

### A.1 Prerequisites

- HuggingFace account with an org (or your personal namespace). Generate a token with **write** scope at https://huggingface.co/settings/tokens. Run `huggingface-cli login` once to cache it.
- GitHub account with an org (or personal namespace). Generate a Personal Access Token (classic) with `write:packages` and `read:packages`. Run `echo $GHCR_TOKEN | docker login ghcr.io -u <you> --password-stdin` once.
- `docker` daemon up; sufficient disk for prewarm (plan ~80 GB headroom on the maintainer machine).
- Optional but recommended: `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` in `.env` so the prewarm sweep during mirroring isn't anonymous-rate-limited.

### A.2 Publish `benchmark/` to HuggingFace

```bash
uv run python scripts/publish_benchmark_to_hf.py \
    --repo-id CyberDualEval/CyberDualEval \
    --revision-tag v2026.05-passE-vulhub
```

The script:

- **Pre-flight: bundles each task's `source/` into a deterministic `source.tar.gz` sibling** (via `scripts/bundle_task_sources.py`). HuggingFace caps a dataset at ~100k files; raw `benchmark/` is ~2.4M files (cybergym ARVO source dominates). Tarballing brings it under the cap. The bundler is idempotent — only stale tarballs are rebuilt — so reruns are cheap. Pass `--skip-bundle` if you've already bundled and just want to re-upload (the script still verifies tarballs are current and aborts if any are stale).
- Creates the dataset repo as **private** (default; pass `--public` once a license review clears it).
- Uploads `benchmark/` excluding raw `source/` trees, curation artifacts (`.curation/`, lockfiles, `__pycache__/`, `.DS_Store`), and HF's reserved `.cache/` namespace.
- Tags the revision so collaborators can pin it.

Re-running is safe — `upload_large_folder` does content-hash-based diffing, so unchanged files aren't re-uploaded. To bump a tag, just re-run with a new `--revision-tag`.

Determinism note: bundles use a fixed gzip mtime + sorted entry order + normalized owner, so re-bundling unchanged source produces byte-identical tarballs. Without this, every publish would churn LFS storage even when no source actually changed.

### A.3 Mirror vendor images to GHCR

```bash
uv run python scripts/mirror_images_to_ghcr.py \
    --namespace ghcr.io/<your-org>/cyberagent-bench \
    --tag v2026.05 \
    --manifest docs/image-mirror-manifest.json \
    --concurrency 2
```

The script:

- Walks every `benchmark/<task>/metadata.json`, follows `compose_source`, extracts every `image:` reference.
- For each image: `docker pull` upstream, capture `sha256` digest, retag to `<namespace>/<safe-name>:<tag>` AND `<namespace>/<safe-name>@<digest>`, push both.
- Writes `docs/image-mirror-manifest.json` mapping each upstream ref to its digest-pinned mirror ref. **Commit this manifest to the repo** — it's how collaborator runs reproduce the exact image set.

The first run is the heavy one (~30-50 GB push). Subsequent runs only push images whose digests changed. If you bump task selection or vendor source, just re-run with the same tag — the manifest gets updated and existing images are skipped.

### A.4 Ensure the eval framework picks up the mirror

When `CYBERAGENTBENCH_IMAGE_MIRROR` is set in the environment, `generate_compose()` rewrites every vendor `image:` in the materialized compose to its mirror digest using `docs/image-mirror-manifest.json`. Without it set, the vendor compose references upstream Docker Hub directly. *(See "Wiring the mirror into `generate_compose`" below — this is a small follow-up edit you should land before publishing the first campaign.)*

### A.5 Tag the campaign

Tag the code repo with the same campaign tag, so a collaborator's clone-by-tag matches the published HF revision and image mirror:

```bash
git tag -a v2026.05-passE-vulhub -m "Pass E Vulhub campaign"
git push origin v2026.05-passE-vulhub
```

---

## Part B — Collaborator setup

You will need: ~100 GB of free disk, docker, Python 3.12, an OpenAI API key (and Anthropic / AWS Bedrock if you want to compare).

### B.1 Prerequisites

```bash
# macOS
brew install git git-lfs uv docker
git lfs install --skip-repo

# Ubuntu / Debian
sudo apt-get install -y git git-lfs docker.io
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs install --skip-repo
```

### B.2 Clone the code repo at the campaign tag

```bash
git clone https://github.com/<maintainer>/cyberagent-bench.git
cd cyberagent-bench
git checkout v2026.05-passE-vulhub   # or whichever campaign you're running
uv sync
```

### B.3 Authenticate with HuggingFace and pull the benchmark

```bash
huggingface-cli login   # paste a read-scope token

uv run python scripts/sync_benchmark_from_hf.py \
    --repo-id CyberDualEval/CyberDualEval \
    --revision v2026.05-passE-vulhub
```

This populates `benchmark/` from the pinned HF revision and then **extracts every `source.tar.gz` back into `source/`** so the eval harness can read raw files. The on-disk footprint after extraction is ~50 GB; tarballs are deleted post-extract by default (pass `--keep-tarballs` to retain them, e.g. for integrity-check reruns). If `benchmark/` already exists from a previous campaign, pass `--force` to back it up to `benchmark.bak-<timestamp>/` before replacing.

### B.4 Build the grader image locally

```bash
docker build -t cyberbench/grader:latest -f refusal_eval/grader/Dockerfile .
docker build -t cyberbench/controller:latest -f refusal_eval/controller/Dockerfile .
```

The grader and controller are local-only sidecars (not published to any registry). You only build them once per machine.

### B.5 Authenticate with GHCR and prewarm vendor images

```bash
echo $GHCR_READ_TOKEN | docker login ghcr.io -u <your-github-username> --password-stdin

export CYBERAGENTBENCH_IMAGE_MIRROR=ghcr.io/<maintainer-org>/cyberagent-bench
uv run python scripts/prewarm_vulhub_images.py
```

Setting `CYBERAGENTBENCH_IMAGE_MIRROR` makes `generate_compose()` rewrite vendor image references to the mirror at materialization time, so docker compose pulls from GHCR (no Docker Hub rate limits) and uses the digest-pinned references in `docs/image-mirror-manifest.json`. Prewarm hydrates the local docker cache once; back-to-back model evaluations reuse it.

If the maintainer hasn't set up the mirror yet, leave `CYBERAGENTBENCH_IMAGE_MIRROR` unset; vendor images come from Docker Hub. Set `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` in `.env` to avoid the 100/6h anonymous limit.

### B.6 Set API keys

```bash
# .env at repo root
OPENAI_API_KEY=                 # fill locally
ANTHROPIC_API_KEY=              # optional
AWS_ACCESS_KEY_ID=              # optional, Bedrock
AWS_SECRET_ACCESS_KEY=          # optional, Bedrock
DOCKERHUB_USERNAME=...               # optional, for upstream-fallback prewarm
DOCKERHUB_TOKEN=...                  # optional
```

### B.7 Run the evaluation

```bash
uv run python scripts/run_models.py config-all-bare-gpt5.4.yaml
```

This config:

- Runs `gpt-5.1` then `gpt-5.4` against the full benchmark (cybench, vulhub, cybergym, exploitdb).
- Uses bare framing only.
- `validated: false` honors per-task `accuracy.skip` and `validation:` blocks as authored.
- `task_order: validated_first` runs graded tasks before mocked ones, so environment regressions surface early.
- Auto-prewarms vulhub images once before the model loop.

Eval logs land under `logs/<timestamp>/`. Each `inspect eval` invocation produces a `.eval` file in `logs/`.

### B.8 Share results back

```bash
# Compress and post to a shared bucket / S3 / drive
tar czf eval-<your-machine>-$(date +%Y%m%d).tar.gz logs/
```

Coordinate with the maintainer for the upload destination (an S3 bucket, GCS bucket, or a shared HF dataset repo for results).

---

## Part C — Operational notes (maintainer)

### C.1 Wiring the mirror into `generate_compose`

The mirror manifest is a static file (`docs/image-mirror-manifest.json`) that maps `upstream_ref -> mirror_digest_ref`. When `CYBERAGENTBENCH_IMAGE_MIRROR` is set in the environment, `refusal_eval/sandbox.py::generate_compose` rewrites each vendor `image:` field to the mirror digest at materialization time. This is a ~20 line edit:

```python
# Pseudocode for the patch in refusal_eval/sandbox.py
_MIRROR_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "docs" / "image-mirror-manifest.json"

def _load_mirror_map() -> dict[str, str]:
    if not os.environ.get("CYBERAGENTBENCH_IMAGE_MIRROR"):
        return {}
    if not _MIRROR_MANIFEST_PATH.is_file():
        return {}
    data = json.loads(_MIRROR_MANIFEST_PATH.read_text())
    return {e["upstream"]: e["mirror_digest"] for e in data.get("entries", [])}

# inside generate_compose, after challenge load:
mirror = _load_mirror_map()
for svc_config in challenge.get("services", {}).values():
    if mirror and isinstance(svc_config, dict) and svc_config.get("image"):
        upstream = svc_config["image"]
        if upstream in mirror:
            svc_config["image"] = mirror[upstream]
```

Land this once, gate it on the env var, and you're done.

### C.2 Adding new tasks

Pass D, Pass E, or any future curation pass appends new tasks to `benchmark/`. To roll them into a new campaign:

1. Run `scripts/publish_benchmark_to_hf.py` with a new `--revision-tag`.
2. Run `scripts/mirror_images_to_ghcr.py` with a new `--tag` — it skips already-mirrored digests, only pushing new ones.
3. Tag the code repo with the matching campaign tag.

### C.3 Updating an existing campaign in place

Don't. If a fix is needed mid-campaign, mint a new tag (`v2026.05.1`) and notify collaborators. Mutating a published tag breaks reproducibility for anyone whose runs are already in flight.

### C.4 License audit checklist before going public

Before flipping the HF dataset to `--public`:

- **CyBench:** review HackTheBox license terms for redistributed challenge content.
- **Vulhub:** MIT-licensed at the repo level, but individual vendor sources have their own licenses; the cloned `source/` trees inside each task should be reviewed.
- **CyberGym:** ARVO is research-licensed; verify redistribution allowance.
- **ExploitDB:** generally redistributable but each `reference/<id>.py` carries its own author attribution.
- Strip or redact any `validation_evidence.json` containing operational details you don't want public (PoC outputs, internal hostnames).

---

## Part D — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `git lfs` not found during HF clone | Missing `git-lfs` install | `brew install git-lfs` / `apt install git-lfs`, then `git lfs install --skip-repo` |
| `403 Forbidden` from HF | Token expired or wrong scope | Regenerate read-scope token; `huggingface-cli login` |
| `unauthorized: authentication required` from GHCR | Not logged in | `docker login ghcr.io -u <you> --password-stdin` |
| `toomanyrequests` from Docker Hub | Anonymous prewarm hit 100/6h | Set DOCKERHUB_USERNAME/TOKEN in `.env`, or wait 6h, or ensure mirror is configured |
| `cyberbench/grader:latest not found` | Local build skipped | `docker build -t cyberbench/grader:latest -f refusal_eval/grader/Dockerfile .` |
| Compose up hangs on a vulhub task | Vendor image failed to pull | Check `docker logs`, retry with manual `docker pull <image>` |
| Eval results show all `phase2_accuracy.skipped=True` | `validated: false` set somewhere (or capability-only run intended) | Confirm config has `validated: true` (default) — `false` means "bypass all in-eval validation" |

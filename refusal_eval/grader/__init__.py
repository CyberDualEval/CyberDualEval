"""Sidecar grader for the validated-exploitation subset.

The grader runs in its own Docker container on the shared network alongside
the agent and vendor target services. It observes target-side state before
Phase 3 (``/baseline``) and after Phase 3 (``/verify``) and reports whether
a causal state change occurred — following the CVE-Bench (Zhu et al. 2025)
in-container-grader pattern, adapted here as a sidecar because Vulhub ships
pre-built vendor images that we cannot modify.
"""

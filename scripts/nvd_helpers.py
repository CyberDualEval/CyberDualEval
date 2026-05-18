"""Shared NVD API helpers for benchmark preparation scripts.

Covers:
- NVD description fetch (with on-disk cache)
- NVD CPE product fetch (with on-disk cache)

Both helpers use the NVD REST API 2.0. The description cache is keyed by
CVE ID and stores the English description string. The CPE cache is keyed
by CVE ID and stores a list of {vendor, product, version} dicts.

Used by prepare_exploitdb.py, prepare_exploitdb_source.py, and prepare_vulhub.py.
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


# ---------------------------------------------------------------------------
# Cache helpers (shared between description and CPE caches)
# ---------------------------------------------------------------------------


def load_cache(cache_path: str) -> dict:
    """Load a JSON cache file. Returns empty dict if missing or malformed."""
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Malformed cache at %s: %s", cache_path, e)
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    """Atomically save a JSON cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# NVD description fetch
# ---------------------------------------------------------------------------


def fetch_nvd_description(
    cve_id: str,
    api_key: str | None,
    cache: dict[str, str],
    cache_path: str,
) -> str | None:
    """Fetch a single CVE description from NVD.

    Returns the English description text, or None if unavailable.
    Caches both hits and negatives (cached as empty string).
    """
    if cve_id in cache:
        return cache[cve_id] or None  # "" means cached negative

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    retries = 3
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.get(
                NVD_API_URL,
                params={"cveId": cve_id},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 404:
                cache[cve_id] = ""
                save_cache(cache_path, cache)
                return None
            if resp.status_code in (429, 500, 502, 503):
                log.warning(
                    "  NVD %d for %s, retry %d/%d in %.0fs",
                    resp.status_code, cve_id, attempt + 1, retries, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            log.warning("  NVD network error for %s: %s (retry %d/%d)", cve_id, e, attempt + 1, retries)
            time.sleep(delay)
            delay *= 2
            continue
        except requests.exceptions.RequestException as e:
            log.warning("  NVD request error for %s: %s", cve_id, e)
            cache[cve_id] = ""
            save_cache(cache_path, cache)
            return None

        # Parse response
        try:
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                cache[cve_id] = ""
                save_cache(cache_path, cache)
                return None

            descriptions = vulns[0].get("cve", {}).get("descriptions", [])
            # Prefer English
            for desc in descriptions:
                if desc.get("lang") == "en":
                    cache[cve_id] = desc["value"]
                    save_cache(cache_path, cache)
                    return desc["value"]
            # Fallback to any language
            if descriptions:
                cache[cve_id] = descriptions[0]["value"]
                save_cache(cache_path, cache)
                return descriptions[0]["value"]
        except (KeyError, IndexError) as e:
            log.warning("  NVD parse error for %s: %s", cve_id, e)

        cache[cve_id] = ""
        save_cache(cache_path, cache)
        return None

    log.warning("  NVD retries exhausted for %s", cve_id)
    return None


def get_nvd_descriptions(
    cves: list[str],
    api_key: str | None,
    cache: dict[str, str],
    cache_path: str,
    rate_delay: float,
) -> str | None:
    """Fetch NVD descriptions for one or more CVEs, combining results.

    Returns combined description text, or None if no descriptions found.
    Sleeps `rate_delay` between *uncached* API calls.
    """
    results: list[tuple[str, str]] = []
    for cve_id in cves:
        was_cached = cve_id in cache
        desc = fetch_nvd_description(cve_id, api_key, cache, cache_path)
        if desc:
            results.append((cve_id, desc))
        if not was_cached:
            time.sleep(rate_delay)

    if not results:
        return None
    if len(results) == 1:
        return results[0][1]
    return "\n\n".join(f"{cve_id}: {desc}" for cve_id, desc in results)


# ---------------------------------------------------------------------------
# NVD CPE product fetch
# ---------------------------------------------------------------------------


def fetch_cpe_products(
    cve_id: str,
    cache: dict,
    cache_path: str,
    api_key: str | None,
    rate_delay: float,
) -> list[dict]:
    """Get [{vendor, product, version, version_source}] from NVD CPE data.

    Filters to application-type CPEs (part == "a"). For rows with a concrete
    `version` in the CPE string, emits that directly. For rows where
    `version == "*"` but NVD exposes range fields, picks one concrete
    version string per populated range bound (priority: end_inclusive >
    end_exclusive > start_inclusive > start_exclusive).

    `version_source` in each dict is one of: "concrete", "end_inclusive",
    "end_exclusive", "start_inclusive", "start_exclusive" — informational,
    logged by callers for debugging.

    Caches both hits and empty results under the CVE ID.
    """
    if cve_id in cache:
        return cache[cve_id]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    url = f"{NVD_API_URL}?cveId={cve_id}"
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                log.warning("  NVD %d, retrying in %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                log.error("  NVD error for %s: %s", cve_id, e)
                cache[cve_id] = []
                save_cache(cache_path, cache)
                return []
            time.sleep(2 ** (attempt + 1))
    else:
        cache[cve_id] = []
        save_cache(cache_path, cache)
        return []

    time.sleep(rate_delay)

    products: list[dict] = []
    try:
        data = resp.json()
        vuln = data["vulnerabilities"][0]["cve"]
        for cfg in vuln.get("configurations", []):
            for node in cfg.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    cpe = match.get("criteria", "")
                    parts = cpe.split(":")
                    if len(parts) < 6 or parts[2] != "a":
                        continue
                    vendor, product = parts[3], parts[4]
                    version = parts[5]
                    if version and version != "*":
                        products.append({
                            "vendor": vendor,
                            "product": product,
                            "version": version,
                            "version_source": "concrete",
                        })
                        continue
                    # version == "*" — derive from range bounds
                    for field, src in (
                        ("versionEndIncluding", "end_inclusive"),
                        ("versionEndExcluding", "end_exclusive"),
                        ("versionStartIncluding", "start_inclusive"),
                        ("versionStartExcluding", "start_exclusive"),
                    ):
                        v = match.get(field)
                        if v:
                            products.append({
                                "vendor": vendor,
                                "product": product,
                                "version": v,
                                "version_source": src,
                            })
                            break
    except (KeyError, IndexError):
        pass

    # Deduplicate by (vendor, product, version)
    seen = set()
    unique: list[dict] = []
    for p in products:
        key = (p["vendor"], p["product"], p["version"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    cache[cve_id] = unique
    save_cache(cache_path, cache)
    return unique


# ---------------------------------------------------------------------------
# NVD reference URLs fetch (for source discovery)
# ---------------------------------------------------------------------------


def fetch_references(
    cve_id: str,
    cache: dict,
    cache_path: str,
    api_key: str | None,
    rate_delay: float,
) -> list[dict]:
    """Return NVD `references` array for a CVE: [{"url": str, "tags": [str]}].

    Cached under `"{cve_id}:refs"` in the shared NVD cache file.
    """
    cache_key = f"{cve_id}:refs"
    if cache_key in cache:
        return cache[cache_key]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    refs: list[dict] = []
    try:
        resp = requests.get(
            NVD_API_URL,
            params={"cveId": cve_id},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if vulns:
                for ref in vulns[0].get("cve", {}).get("references", []):
                    url = ref.get("url")
                    if url:
                        refs.append({"url": url, "tags": ref.get("tags", [])})
        elif resp.status_code in (429, 503):
            log.warning("  NVD %d on references for %s", resp.status_code, cve_id)
    except requests.RequestException as e:
        log.warning("  NVD references error for %s: %s", cve_id, e)

    time.sleep(rate_delay)
    cache[cache_key] = refs
    save_cache(cache_path, cache)
    return refs


# ---------------------------------------------------------------------------
# NVD CWE fetch (for XSS/CSRF filtering in Vulhub)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NVD CVSS metrics fetch
# ---------------------------------------------------------------------------


def fetch_cvss(
    cve_id: str,
    api_key: str | None,
    cache: dict,
    cache_path: str,
) -> dict | None:
    """Return CVSS metrics for a CVE, preferring v3.1 > v3.0 > v2.0.

    Result shape (or None if unavailable):
        {
            "version": "3.1" | "3.0" | "2.0",
            "base": float,            # baseScore
            "impact": float,          # impactScore (v3: 0–6.0, v2: 0–10.0)
            "exploitability": float,  # exploitabilityScore (v3: 0–3.9, v2: 0–10.0)
            "vector": str,            # vectorString, e.g. "AV:N/AC:L/..."
        }

    Cached under `"{cve_id}:cvss"` in the shared NVD cache file. Negative
    lookups are cached as None.
    """
    cache_key = f"{cve_id}:cvss"
    if cache_key in cache:
        return cache[cache_key]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = requests.get(
            NVD_API_URL,
            params={"cveId": cve_id},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            cache[cache_key] = None
            save_cache(cache_path, cache)
            return None
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            cache[cache_key] = None
            save_cache(cache_path, cache)
            return None
        metrics = vulns[0].get("cve", {}).get("metrics", {})
    except requests.RequestException as e:
        log.warning("  NVD CVSS error for %s: %s", cve_id, e)
        cache[cache_key] = None
        save_cache(cache_path, cache)
        return None

    # Prefer v3.1 > v3.0 > v2.0
    for metric_key, version_label in (
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ):
        entries = metrics.get(metric_key, [])
        if not entries:
            continue
        # Prefer the NVD-primary entry if present
        primary = next((e for e in entries if e.get("type") == "Primary"), entries[0])
        cvss = primary.get("cvssData", {})
        base = cvss.get("baseScore")
        vector = cvss.get("vectorString")
        impact = primary.get("impactScore")
        exploit = primary.get("exploitabilityScore")
        if base is None or impact is None or exploit is None:
            continue
        result = {
            "version": version_label,
            "base": float(base),
            "impact": float(impact),
            "exploitability": float(exploit),
            "vector": vector or "",
        }
        cache[cache_key] = result
        save_cache(cache_path, cache)
        return result

    cache[cache_key] = None
    save_cache(cache_path, cache)
    return None


def fetch_cwes(
    cve_id: str,
    api_key: str | None,
    cache: dict,
    cache_path: str,
) -> list[str]:
    """Return the list of CWE IDs associated with a CVE (e.g., ['CWE-79']).

    Uses a dedicated `_cwes` key per CVE in the shared description cache to
    avoid a second API round-trip. Returns empty list on miss.
    """
    cwe_key = f"{cve_id}:cwes"
    if cwe_key in cache:
        return cache[cwe_key]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = requests.get(
            NVD_API_URL,
            params={"cveId": cve_id},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            cache[cwe_key] = []
            save_cache(cache_path, cache)
            return []
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            cache[cwe_key] = []
            save_cache(cache_path, cache)
            return []
        cwes: list[str] = []
        for weakness in vulns[0].get("cve", {}).get("weaknesses", []):
            for desc in weakness.get("description", []):
                val = desc.get("value", "")
                if val.startswith("CWE-"):
                    cwes.append(val)
        cwes = sorted(set(cwes))
        cache[cwe_key] = cwes
        save_cache(cache_path, cache)
        return cwes
    except requests.RequestException:
        cache[cwe_key] = []
        save_cache(cache_path, cache)
        return []
